use ffi::Gles2;

use super::*;
use std::sync::Arc;

/// A handle to a GLES texture
///
/// The texture can be used with the same [`GlesRenderer`] it was created with, or one using a
/// shared [`EGLContext`].
#[derive(Debug, Clone)]
pub struct GlesTexture(pub(super) Arc<GlesTextureInternal>);

/// A texture object sharing the pixels and synchronization of a [`GlesTexture`].
///
/// Sampling parameters belong to this view, independently of its source texture.
/// The view must be used in the context that created it, or a shared context.
/// Callers manage bindless residency established in additional or externally managed contexts.
#[derive(Debug, Clone)]
pub struct GlesTextureView(pub(super) Arc<GlesTextureViewInner>);

#[derive(Debug)]
pub(super) struct GlesTextureViewInner {
    pub source: GlesTexture,
    pub texture: ffi::types::GLuint,
    pub cleanup: Arc<ViewCleanupOwner>,
}

#[derive(Debug)]
pub(super) struct ViewCleanupOwner {
    pub id: ContextId<GlesTexture>,
    pending: Mutex<Option<Vec<ffi::types::GLuint>>>,
    shared: Sender<CleanupResource>,
}

impl ViewCleanupOwner {
    fn delete(&self, texture: ffi::types::GLuint) {
        let mut pending = self.pending.lock().unwrap();
        if let Some(pending) = pending.as_mut() {
            pending.push(texture);
        } else {
            let _ = self.shared.send(CleanupResource::Texture(texture));
        }
    }
}

#[derive(Debug)]
pub(super) struct ViewCleanup(pub Arc<ViewCleanupOwner>);

impl ViewCleanup {
    pub fn new(shared: Sender<CleanupResource>) -> Self {
        Self(Arc::new(ViewCleanupOwner {
            id: ContextId::new(),
            pending: Mutex::new(Some(Vec::new())),
            shared,
        }))
    }

    pub fn cleanup(&self, egl: &EGLContext, gl: &Gles2) -> Result<(), MakeCurrentError> {
        let mut pending = self.0.pending.lock().unwrap();
        let Some(pending) = pending.as_mut().filter(|pending| !pending.is_empty()) else {
            return Ok(());
        };
        if !egl.is_current() {
            unsafe {
                egl.make_current()?;
            }
        }
        unsafe {
            gl.DeleteTextures(pending.len() as i32, pending.as_ptr());
        }
        pending.clear();
        Ok(())
    }
}

impl Drop for ViewCleanup {
    fn drop(&mut self) {
        // This field drops after the owned EGL context, whose residencies are now gone.
        // Close the queue atomically so late view drops use the surviving share group.
        let pending = self.0.pending.lock().unwrap().take().unwrap_or_default();
        for texture in pending {
            let _ = self.0.shared.send(CleanupResource::Texture(texture));
        }
    }
}

impl GlesTextureView {
    /// Whether this view shares pixels with the given source texture.
    pub fn is_view_of(&self, source: &GlesTexture) -> bool {
        Arc::ptr_eq(&self.0.source.0, &source.0)
    }
    /// OpenGL texture name. Ownership remains with this view.
    pub fn tex_id(&self) -> ffi::types::GLuint {
        self.0.texture
    }

    /// The texture whose pixels this view shares.
    pub fn source(&self) -> &GlesTexture {
        &self.0.source
    }

    /// Whether sampling requires `samplerExternalOES`.
    pub fn is_external(&self) -> bool {
        self.0.source.0.is_external
    }

    /// Whether the texture contains an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.0.source.0.has_alpha
    }
}

impl Drop for GlesTextureViewInner {
    fn drop(&mut self) {
        *self.source.0.view_references.lock().unwrap() -= 1;
        self.cleanup.delete(self.texture);
    }
}

impl GlesTexture {
    /// Whether sampling this texture requires `samplerExternalOES`.
    pub fn is_external(&self) -> bool {
        self.0.is_external
    }
    /// Create a GlesTexture from a raw gl texture id.
    ///
    /// It is expected to not be external or y_inverted.
    ///
    /// Ownership over the texture is taken by the renderer, you should not free the texture yourself.
    ///
    /// # Safety
    ///
    /// The renderer cannot make sure `tex` is a valid texture id.
    pub unsafe fn from_raw(
        renderer: &GlesRenderer,
        internal_format: Option<ffi::types::GLenum>,
        opaque: bool,
        tex: ffi::types::GLuint,
        size: Size<i32, BufferCoord>,
    ) -> GlesTexture {
        GlesTexture(Arc::new(GlesTextureInternal {
            texture: tex,
            sync: RwLock::default(),
            view_references: Mutex::new(0),
            format: internal_format,
            has_alpha: !opaque,
            is_external: false,
            y_inverted: false,
            size,
            egl_images: None,
            destruction_callback_sender: renderer.gles_cleanup().sender.clone(),
        }))
    }

    /// OpenGL texture id of this texture
    ///
    /// This id will become invalid, when the GlesTexture is dropped and does not transfer ownership.
    pub fn tex_id(&self) -> ffi::types::GLuint {
        self.0.texture
    }

    /// Whether the texture is upside down
    pub fn is_y_inverted(&self) -> bool {
        self.0.y_inverted
    }

    /// Whether this is the only snapshot reference to this texture (strong or weak).
    ///
    /// Texture views share live pixels rather than retaining a snapshot, so they
    /// do not prevent updating the source's contents.
    ///
    /// Note that this tracks only references to this Smithay object (that you can get by cloning
    /// it). If you make a reference via OpenGL directly somehow, you need to keep track of it on
    /// your own.
    pub fn is_unique_reference(&mut self) -> bool {
        {
            let views = self.0.view_references.lock().unwrap();
            Arc::strong_count(&self.0) == 1 + *views && Arc::weak_count(&self.0) == 0
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct TextureSync {
    pub read_sync: Mutex<ReadSync>,
    write_sync: Mutex<Option<ffi::types::GLsync>>,
    pub view_image: Option<EGLImage>,
}

#[derive(Debug, Default)]
pub(super) struct ReadSync {
    normal: Option<ffi::types::GLsync>,
    pub views: Option<Arc<ViewReadFence>>,
}

#[derive(Debug)]
pub(super) struct ViewReadFence {
    pub context: ContextId<GlesTexture>,
    pub sync: ffi::types::GLsync,
    pub sender: Sender<CleanupResource>,
}

impl Drop for ViewReadFence {
    fn drop(&mut self) {
        let _ = self.sender.send(CleanupResource::Sync(self.sync));
    }
}

unsafe fn wait_for_syncpoint(sync: &mut Option<ffi::types::GLsync>, gl: &Gles2) {
    if let Some(sync_obj) = *sync {
        match gl.ClientWaitSync(sync_obj, 0, 0) {
            ffi::ALREADY_SIGNALED | ffi::CONDITION_SATISFIED => {
                let _ = sync.take();
                gl.DeleteSync(sync_obj);
            }
            _ => {
                gl.WaitSync(sync_obj, 0, ffi::TIMEOUT_IGNORED);
            }
        };
    }
}

impl TextureSync {
    pub(super) fn wait_for_upload(&self, gl: &Gles2) {
        unsafe {
            wait_for_syncpoint(&mut self.write_sync.lock().unwrap(), gl);
        }
    }

    pub(super) fn update_read(&self, gl: &Gles2) {
        let mut read_sync = self.read_sync.lock().unwrap();
        if let Some(old) = read_sync.normal.take() {
            unsafe {
                gl.WaitSync(old, 0, ffi::TIMEOUT_IGNORED);
                gl.DeleteSync(old);
            };
        }
        read_sync.normal = Some(unsafe { gl.FenceSync(ffi::SYNC_GPU_COMMANDS_COMPLETE, 0) });
    }

    pub(super) fn wait_for_all(&mut self, gl: &Gles2) {
        unsafe {
            let reads = self.read_sync.get_mut().unwrap();
            wait_for_syncpoint(&mut reads.normal, gl);
            if let Some(fence) = reads.views.take() {
                gl.WaitSync(fence.sync, 0, ffi::TIMEOUT_IGNORED);
            }
            wait_for_syncpoint(self.write_sync.get_mut().unwrap(), gl);
        }
    }

    pub(super) fn update_write(&mut self, gl: &Gles2) {
        let write_sync = self.write_sync.get_mut().unwrap();
        if let Some(old) = write_sync.take() {
            unsafe {
                gl.WaitSync(old, 0, ffi::TIMEOUT_IGNORED);
                gl.DeleteSync(old);
            };
        }

        *write_sync = Some(unsafe { gl.FenceSync(ffi::SYNC_GPU_COMMANDS_COMPLETE, 0) });
    }
}

#[derive(Debug)]
pub(super) struct GlesTextureInternal {
    pub(super) texture: ffi::types::GLuint,
    pub(super) sync: RwLock<TextureSync>,
    pub(super) view_references: Mutex<usize>,
    pub(super) format: Option<ffi::types::GLenum>,
    pub(super) has_alpha: bool,
    pub(super) is_external: bool,
    pub(super) y_inverted: bool,
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) egl_images: Option<Vec<EGLImage>>,
    pub(super) destruction_callback_sender: Sender<CleanupResource>,
}
unsafe impl Send for GlesTextureInternal {}
unsafe impl Sync for GlesTextureInternal {}

impl Drop for GlesTextureInternal {
    fn drop(&mut self) {
        let _ = self
            .destruction_callback_sender
            .send(CleanupResource::Texture(self.texture));
        let mut sync = self.sync.write().unwrap();
        if let Some(sync) = sync.read_sync.get_mut().unwrap().normal.take() {
            let _ = self
                .destruction_callback_sender
                .send(CleanupResource::Sync(sync as *const _));
        }
        if let Some(sync) = sync.write_sync.get_mut().unwrap().take() {
            let _ = self
                .destruction_callback_sender
                .send(CleanupResource::Sync(sync as *const _));
        }
        if let Some(images) = self.egl_images.take() {
            for image in images {
                let _ = self
                    .destruction_callback_sender
                    .send(CleanupResource::EGLImage(image));
            }
        }
        if let Some(image) = sync.view_image.take() {
            let _ = self
                .destruction_callback_sender
                .send(CleanupResource::EGLImage(image));
        }
    }
}

impl Texture for GlesTexture {
    fn width(&self) -> u32 {
        self.0.size.w as u32
    }
    fn height(&self) -> u32 {
        self.0.size.h as u32
    }
    fn size(&self) -> Size<i32, BufferCoord> {
        self.0.size
    }
    fn format(&self) -> Option<Fourcc> {
        let fmt = gl_internal_format_to_fourcc(self.0.format?);
        if self.0.has_alpha {
            fmt
        } else {
            fmt.and_then(get_opaque)
        }
    }
}

/// Texture mapping of a GLES2 texture
#[derive(Debug)]
pub struct GlesMapping {
    pub(super) pbo: ffi::types::GLuint,
    pub(super) format: ffi::types::GLenum,
    pub(super) layout: ffi::types::GLenum,
    pub(super) has_alpha: bool,
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) mapping: AtomicPtr<std::ffi::c_void>,
    pub(super) destruction_callback_sender: Sender<CleanupResource>,
}

impl Texture for GlesMapping {
    fn width(&self) -> u32 {
        self.size.w as u32
    }
    fn height(&self) -> u32 {
        self.size.h as u32
    }
    fn size(&self) -> Size<i32, BufferCoord> {
        self.size
    }
    fn format(&self) -> Option<Fourcc> {
        let fmt = gl_read_format_to_fourcc(self.format, self.layout);
        if self.has_alpha {
            fmt
        } else {
            fmt.and_then(get_opaque)
        }
    }
}
impl TextureMapping for GlesMapping {
    fn flipped(&self) -> bool {
        true
    }
    fn format(&self) -> Fourcc {
        Texture::format(self).expect("Should never happen")
    }
}

impl Drop for GlesMapping {
    fn drop(&mut self) {
        let _ = self.destruction_callback_sender.send(CleanupResource::Mapping(
            self.pbo,
            self.mapping.load(Ordering::SeqCst),
        ));
    }
}
