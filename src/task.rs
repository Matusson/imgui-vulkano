//! Task implementation for imgui rendering in the task graph.
//! Split into upload (outside render pass) and draw (inside render pass) tasks.

use crate::renderer::VulkanoRenderer;
use crate::Vertex;
use std::cell::RefCell;
use std::sync::Arc;
use vulkano::render_pass::Subpass;
use vulkano_taskgraph::command_buffer::RecordingCommandBuffer;
use vulkano_taskgraph::{Id, Task, TaskContext, TaskResult};
use vulkano::buffer::Buffer;
use vulkano::image::Image;
use bytemuck::{Pod, Zeroable};
use vulkano::pipeline::Pipeline;

/// Draw data for a single frame (internal, passed from upload to draw task).
#[derive(Clone)]
pub(crate) struct ImguiDrawData {
    pub(crate) fb_width: f32,
    pub(crate) fb_height: f32,
    pub(crate) display_size: [f32; 2],
    pub(crate) framebuffer_scale: [f32; 2],
}

#[derive(Clone)]
pub(crate) struct ImguiDrawCommand {
    pub(crate) count: usize,
    pub(crate) texture_id: imgui::TextureId,
    pub(crate) clip_rect: [f32; 4],
    pub(crate) idx_offset: usize,
    pub(crate) vtx_offset: usize,
}

use vulkano_taskgraph::descriptor_set::{SamplerId, SampledImageId};

/// Holds frame data passed from the upload task to the draw task.
///
/// This struct must be included in your context type that implements `HasImguiContext`.
/// It stores the vertex/index data and draw commands for the current frame, allowing
/// the upload task to pass this data to the draw task.
///
/// The vertex, index, and draw command buffers are reused across frames to avoid repeated allocations.
pub struct ImguiFrameData {
    draw_data: RefCell<Option<ImguiDrawData>>,
    vertices: RefCell<Vec<Vertex>>,
    indices: RefCell<Vec<u16>>,
    draw_commands: RefCell<Vec<(usize, usize, Vec<ImguiDrawCommand>)>>,
}

impl ImguiFrameData {
    /// Creates a new `ImguiFrameData` instance.
    pub fn new() -> Self {
        Self {
            draw_data: RefCell::new(None),
            vertices: RefCell::new(Vec::new()),
            indices: RefCell::new(Vec::new()),
            draw_commands: RefCell::new(Vec::new()),
        }
    }

    /// Sets the draw data for this frame.
    pub(crate) fn set(&self, data: ImguiDrawData) {
        *self.draw_data.borrow_mut() = Some(data);
    }

    /// Takes the draw data for this frame, leaving None.
    pub(crate) fn take(&self) -> Option<ImguiDrawData> {
        self.draw_data.borrow_mut().take()
    }

    pub(crate) fn vertices_mut(&self) -> std::cell::RefMut<'_, Vec<Vertex>> {
        self.vertices.borrow_mut()
    }

    pub(crate) fn indices_mut(&self) -> std::cell::RefMut<'_, Vec<u16>> {
        self.indices.borrow_mut()
    }

    pub(crate) fn draw_commands_mut(&self) -> std::cell::RefMut<'_, Vec<(usize, usize, Vec<ImguiDrawCommand>)>> {
        self.draw_commands.borrow_mut()
    }
}

impl Default for ImguiFrameData {
    fn default() -> Self {
        Self::new()
    }
}

/// Push constants structure matching the shader layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ImguiPushConstants {
    matrix: [[f32; 4]; 4],
    sampled_image_id: SampledImageId,
    sampler_id: SamplerId,
}

/// Trait for providing imgui context and UI building callback.
///
/// This trait is required to use the imgui task graph integration. It provides access
/// to the imgui context and renderer, and defines how UI is built each frame.
///
/// Platform integration is handled by the user outside
/// the task graph. Call your platform's `prepare_frame()` before executing the graph.
///
/// # Implementation Approaches
///
/// You have two options for implementing this trait:
///
/// The easiest approach is to use the provided [`ImguiContext`](crate::ImguiContext) struct,
/// which implements this trait for you. You should wrap it with your application struct.
///
/// Alternatively, you can also implement the trait directly on your own struct, though
/// you need to handle some boilerplate in this case.
///
/// The typical setup flow when using this trait:
///
/// 1. Create imgui context, platform, and renderer
/// 2. Create your struct implementing `HasImguiContext`
/// 3. Create physical buffers with `ImguiBuffers::new()`
/// 4. Build task graph with `setup_resources()` and `setup_imgui_tasks()`
/// 5. Compile task graph
/// 6. Call `ImguiTaskNodes::setup_after_compile()`
/// 7. Execute task graph each frame
///
/// See the crate-level documentation for examples.
pub trait HasImguiContext {
    /// Returns references to the imgui context and renderer.
    fn imgui_components(&self) -> (&RefCell<imgui::Context>, &RefCell<VulkanoRenderer>);

    /// Returns a reference to the frame data storage.
    ///
    /// This is used internally to pass draw data from the upload task to the draw task.
    /// You must include an `ImguiFrameData` field in your context and return a reference to it.
    fn imgui_frame_data(&self) -> &ImguiFrameData;

    /// Build your custom UI for this frame.
    ///
    /// This method is called each frame during the upload task. Use the provided
    /// `Ui` object to create windows, draw widgets, etc.
    ///
    /// Platform integration (prepare_frame/prepare_render) should be handled
    /// by the user before/after task graph execution. This method only builds the UI.
    fn build_ui(&self, ui: &imgui::Ui);

    /// Optional hook called after `build_ui` but before rendering.
    ///
    /// This allows platform-specific integration (like winit's `prepare_render`)
    /// to be called at the correct time without making the library platform-specific.
    ///
    /// The default implementation does nothing.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn after_build_ui(&self, ui: &imgui::Ui) {
    ///     // For winit integration:
    ///     self.platform.borrow_mut().prepare_render(ui, &self.window);
    /// }
    /// ```
    fn after_build_ui(&self, _ui: &imgui::Ui) {
        // Default implementation does nothing
    }
}

/// Upload task that prepares imgui draw data and uploads buffers (runs outside render pass).
pub struct ImguiUploadTask<W> {
    pub v_vertex_buffer: Id<Buffer>,
    pub v_index_buffer: Id<Buffer>,
    pub v_vertex_staging: Id<Buffer>,
    pub v_index_staging: Id<Buffer>,
    pub _phantom: std::marker::PhantomData<W>,
}

impl<W: HasImguiContext + Send + Sync + 'static> Task for ImguiUploadTask<W> {
    type World = W;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer,
        _tcx: &mut TaskContext,
        world: &Self::World,
    ) -> TaskResult {
        // Get imgui context from world
        let (ctx_ref, _) = world.imgui_components();

        // Build UI
        let mut imgui_ctx = ctx_ref.borrow_mut();
        let ui = imgui_ctx.new_frame();
        world.build_ui(&ui);

        // Call platform-specific hook (e.g., prepare_render for winit)
        world.after_build_ui(&ui);

        // Get draw data
        let draw_data = imgui_ctx.render();

        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];

        if fb_width <= 0.0 || fb_height <= 0.0 {
            world.imgui_frame_data().set(ImguiDrawData {
                fb_width: 0.0,
                fb_height: 0.0,
                display_size: draw_data.display_size,
                framebuffer_scale: draw_data.framebuffer_scale,
            });
            return Ok(());
        }

        // Get reusable buffers and clear them for this frame
        let frame_data = world.imgui_frame_data();
        let mut all_vertices = frame_data.vertices_mut();
        let mut all_indices = frame_data.indices_mut();
        let mut draw_commands_list = frame_data.draw_commands_mut();
        all_vertices.clear();
        all_indices.clear();
        draw_commands_list.clear();

        for draw_list in draw_data.draw_lists() {
            let vtx_offset = all_vertices.len();
            let idx_offset = all_indices.len();

            for v in draw_list.vtx_buffer() {
                all_vertices.push(Vertex::from(*v));
            }

            for &idx in draw_list.idx_buffer() {
                all_indices.push(idx);
            }

            let mut commands = Vec::new();
            for cmd in draw_list.commands() {
                if let imgui::DrawCmd::Elements { count, cmd_params } = cmd {
                    commands.push(ImguiDrawCommand {
                        count,
                        texture_id: cmd_params.texture_id,
                        clip_rect: cmd_params.clip_rect,
                        idx_offset: cmd_params.idx_offset,
                        vtx_offset: cmd_params.vtx_offset,
                    });
                }
            }

            draw_commands_list.push((vtx_offset, idx_offset, commands));
        }

        // Validate buffer sizes
        let vertex_byte_size = all_vertices.len() as u64 * size_of::<Vertex>() as u64;
        let index_byte_size = all_indices.len() as u64 * size_of::<u16>() as u64;

        let vertex_buffer_size = _tcx.buffer(self.v_vertex_buffer)?.buffer().size();
        let index_buffer_size = _tcx.buffer(self.v_index_buffer)?.buffer().size();

        if vertex_byte_size > vertex_buffer_size {
            panic!(
                "ImGui vertex data ({} bytes) exceeds vertex buffer size ({} bytes). \
                 Consider increasing vertex_buffer_size in ImguiBufferConfig.",
                vertex_byte_size, vertex_buffer_size
            );
        }

        if index_byte_size > index_buffer_size {
            panic!(
                "ImGui index data ({} bytes) exceeds index buffer size ({} bytes). \
                 Consider increasing index_buffer_size in ImguiBufferConfig.",
                index_byte_size, index_buffer_size
            );
        }

        // Upload buffers via staging buffers
        if !all_vertices.is_empty() {
            // Write to staging buffer
            let vertex_size = size_of::<Vertex>() as u64;
            let vertex_byte_len = all_vertices.len() as u64 * vertex_size;
            _tcx.write_buffer::<[Vertex]>(self.v_vertex_staging, 0..vertex_byte_len)?
                .copy_from_slice(&all_vertices);

            // Copy from staging to device buffer
            unsafe {
                cbf.copy_buffer(&vulkano_taskgraph::command_buffer::CopyBufferInfo {
                    src_buffer: self.v_vertex_staging,
                    dst_buffer: self.v_vertex_buffer,
                    ..Default::default()
                })?;
            }
        }

        if !all_indices.is_empty() {
            // Write to staging buffer
            let index_size = std::mem::size_of::<u16>() as u64;
            let index_byte_len = all_indices.len() as u64 * index_size;
            _tcx.write_buffer::<[u16]>(self.v_index_staging, 0..index_byte_len)?
                .copy_from_slice(&all_indices);

            // Copy from staging to device buffer
            unsafe {
                cbf.copy_buffer(&vulkano_taskgraph::command_buffer::CopyBufferInfo {
                    src_buffer: self.v_index_staging,
                    dst_buffer: self.v_index_buffer,
                    ..Default::default()
                })?;
            }
        }

        drop(all_vertices);
        drop(all_indices);
        drop(draw_commands_list);

        // Store draw data for the draw task
        world.imgui_frame_data().set(ImguiDrawData {
            fb_width,
            fb_height,
            display_size: draw_data.display_size,
            framebuffer_scale: draw_data.framebuffer_scale,
        });

        Ok(())
    }
}

/// Draw task that renders imgui UI (runs inside render pass).
pub struct ImguiDrawTask<W> {
    pub v_vertex_buffer: Id<Buffer>,
    pub v_index_buffer: Id<Buffer>,
    pub target: Id<Image>,
    pub subpass: Option<Arc<Subpass>>,
    pub _phantom: std::marker::PhantomData<W>,
}

impl<W: HasImguiContext + Send + Sync + 'static> Task for ImguiDrawTask<W> {
    type World = W;

    fn clear_values(&self, _clear_values: &mut vulkano_taskgraph::ClearValues<'_>, _world: &Self::World) {

    }

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer,
        _tcx: &mut TaskContext,
        world: &Self::World,
    ) -> TaskResult {
        // Get draw data from context (prepared by upload task)
        let draw_data = match world.imgui_frame_data().take() {
            Some(data) => data,
            None => return Ok(()), // No draw data available
        };

        let frame_data = world.imgui_frame_data();
        let draw_commands = frame_data.draw_commands_mut();
        if draw_commands.is_empty() {
            return Ok(());
        }

        let (_, renderer_ref) = world.imgui_components();
        let renderer = renderer_ref.borrow();
        let pipeline = renderer.pipeline().clone();

        // Bind pipeline and buffers
        unsafe {
            cbf.bind_pipeline_graphics(&pipeline)?
                .bind_vertex_buffers(0, &[self.v_vertex_buffer], &[0], &[], &[])?
                .bind_index_buffer(self.v_index_buffer, 0, None, vulkano::buffer::IndexType::U16)?;
        }

        // Setup viewport
        let viewport = vulkano::pipeline::graphics::viewport::Viewport {
            offset: [0.0, 0.0],
            extent: [draw_data.fb_width, draw_data.fb_height],
            min_depth: 0.0,
            max_depth: 1.0,
        };

        cbf.set_viewport(0, &[viewport])?;

        // Setup orthographic projection
        let matrix = [
            [2.0 / draw_data.display_size[0], 0.0, 0.0, 0.0],
            [0.0, 2.0 / draw_data.display_size[1], 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0],
        ];

        // Render each draw list
        for (vtx_offset, idx_offset, commands) in draw_commands.iter() {
            for cmd in commands {
                let texture = renderer.lookup_texture(cmd.texture_id)
                    .expect("Texture not found - this is a programming error");

                let push_constants = ImguiPushConstants {
                    matrix,
                    sampled_image_id: texture.sampled_image_id,
                    sampler_id: texture.sampler_id,
                };

                cbf.push_constants(
                    pipeline.layout(),
                    0,
                    &push_constants,
                )?;

                // Set scissor
                let clip_rect = cmd.clip_rect;
                let scissor = vulkano::pipeline::graphics::viewport::Scissor {
                    offset: [
                        (clip_rect[0] * draw_data.framebuffer_scale[0]).max(0.0) as u32,
                        (clip_rect[1] * draw_data.framebuffer_scale[1]).max(0.0) as u32,
                    ],
                    extent: [
                        ((clip_rect[2] - clip_rect[0]) * draw_data.framebuffer_scale[0]) as u32,
                        ((clip_rect[3] - clip_rect[1]) * draw_data.framebuffer_scale[1]) as u32,
                    ],
                };

                cbf.set_scissor(0, &[scissor])?;

                cbf.draw_indexed(
                    cmd.count as u32,
                    1,
                    (idx_offset + cmd.idx_offset) as u32,
                    (vtx_offset + cmd.vtx_offset) as i32,
                    0,
                )?;
            }
        }

        Ok(())
    }
}

/// Setup implementation for the draw task (called after graph compilation).
impl<W: HasImguiContext> ImguiDrawTask<W> {
    pub fn setup_after_compile(
        &mut self,
        subpass: Arc<Subpass>,
        bindless_context: &vulkano_taskgraph::descriptor_set::BindlessContext,
        world: &W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.subpass = Some(subpass.clone());

        // Create pipeline in renderer
        let (_, renderer_ref) = world.imgui_components();
        let mut renderer = renderer_ref.borrow_mut();
        renderer.create_pipeline(subpass, bindless_context)?;

        Ok(())
    }
}
