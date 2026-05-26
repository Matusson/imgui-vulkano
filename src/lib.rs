//! # imgui-vulkano-task-renderer
//!
//! An [imgui-rs](https://github.com/imgui-rs/imgui-rs) renderer based on [vulkano-taskgraph](https://github.com/vulkano-rs/vulkano).
//!
//!
//! ## Quick Start
//!
//! ```rust
//! // 1. Create Vulkan resources with bindless context
//! let resources = Resources::new(
//!     &device,
//!     &ResourcesCreateInfo {
//!         bindless_context: Some(&BindlessContextCreateInfo {
//!             global_set: &GlobalDescriptorSetCreateInfo::new(),
//!             local_set: None,
//!             ..Default::default()
//!         }),
//!         ..Default::default()
//!     },
//! )?;
//!
//! let flight_id = resources.create_flight(2)?;
//! let bindless_context = resources.bindless_context()
//!     .expect("Resources should have bindless context");
//!
//! // 2. Create ImGui context, platform, and renderer
//! let mut imgui_ctx = Context::create();
//! imgui_ctx.set_ini_filename(None);
//!
//! // Initialize platform (winit is used in examples)
//! let mut platform = WinitPlatform::new(&mut imgui_ctx);
//! platform.attach_window(imgui_ctx.io_mut(), &window, HiDpiMode::Rounded);
//!
//! let renderer = VulkanoRenderer::new(
//!     &mut imgui_ctx,
//!     device.clone(),
//!     queue.clone(),
//!     &resources,
//!     flight_id,
//!     bindless_context,
//!     Some(2.2), // Optional gamma correction (use Some(2.2) for sRGB, None or Some(1.0) for linear)
//! )?;
//!
//! // 3. Create your context implementing HasImguiContext
//! // You can use the ImguiContext helper struct to reduce boilerplate, or
//! // you can implement it from scratch. See examples for more details.
//! struct RenderContext {
//!     imgui: ImguiContext,
//!     // ... your application state
//! }
//!
//! impl HasImguiContext for RenderContext {
//!     fn imgui_components(&self) -> (&RefCell<Context>, &RefCell<VulkanoRenderer>) {
//!         self.imgui.imgui_components()  // Delegate to ImguiContext
//!     }
//!
//!     fn imgui_frame_data(&self) -> &ImguiFrameData {
//!         self.imgui.imgui_frame_data()  // Delegate to ImguiContext
//!     }
//!
//!     // This function is where you implement your GUI drawing code.
//!     fn build_ui(&self, ui: &imgui::Ui) {
//!         ui.window("Hello world")
//!             .size([300.0, 110.0], Condition::FirstUseEver)
//!             .build(|| {
//!                 ui.text("Hello from imgui-vulkano-task-renderer!");
//!             });
//!     }
//!
//!     // Optional: Add platform-specific integration (e.g., winit's prepare_render)
//!     fn after_build_ui(&self, ui: &imgui::Ui) {
//!         // For winit integration:
//!         // self.platform.borrow_mut().prepare_render(ui, &self.window);
//!     }
//! }
//!
//! // Create the ImguiContext helper
//! let imgui_context = ImguiContext::new(
//!     imgui_ctx,
//!     platform,
//!     renderer,
//!     window.clone(),
//! );
//!
//! let context = RenderContext {
//!     imgui: imgui_context,
//!     // ... your application state
//! };
//!
//! // 4. Create physical buffers (once, at the start)
//! let imgui_buffers = ImguiBuffers::new(&resources)?;
//!
//! // 5. Build task graph
//! let mut task_graph = TaskGraph::new(&resources);
//!
//! // Create virtual swapchain and get target image
//! let v_swapchain_id = task_graph.add_swapchain(&swapchain_info);
//! let target_image = v_swapchain_id.current_image_id();
//!
//! // Setup ImGui tasks
//! let v_imgui_buffers = imgui_buffers.setup_resources(&mut task_graph);
//! let imgui_task_nodes = v_imgui_buffers
//!     .setup_imgui_tasks(&mut task_graph, target_image, None, None)?;
//!
//! // 6. Compile task graph
//! let mut executable = unsafe {
//!     task_graph.compile(&CompileInfo {
//!         queues: &[&queue],
//!         present_queue: Some(&queue),
//!         flight_id,
//!         ..Default::default()
//!     })?
//! };
//!
//! // 7. Setup pipelines after compilation
//! imgui_task_nodes.setup_after_compile(&mut executable, &resources, &context)?;
//!
//! // 8. Render each frame
//! loop {
//!     // Map virtual resources to physical resources
//!     let resource_map = resource_map!(
//!         &executable,
//!         v_swapchain_id => swapchain_id,
//!     )?;
//!
//!     v_imgui_buffers.map_buffers(&mut resource_map, &imgui_buffers)?;
//!
//!     // Execute task graph
//!     unsafe {
//!         executable.execute(resource_map, &context, || {
//!             window.pre_present_notify()
//!         })?
//!     };
//! }
//! ```
//!
//! This is a rough implementation guide and is not meant to be compiled. For full samples, check the examples:
//! ```bash
//! cargo run --example hello_world
//! ```
//! ```bash
//! cargo run --example custom_textures
//! ```
//! Additionally, you may only use parts of the library if you wish. For example, you may want to manage buffers yourself,
//! or you might want to wrap the default task implementations to add extra logic.
//! ## Pre-release warning
//! This crate is currently pre-release and the API might introduce breaking changes without warning. This will
//! be the case at least until `vulkano-taskgraph` is fully released, as right now it requires a git dependency. I'm
//! also not certain on API design just yet, and it might evolve over time. Once `vulkano` publishes a stable
//! release with `vulkano-taskgraph`, this crate will also get a stable release.
//!
//! So far, I only tested it with reasonably basic UIs. It's possible there are issues, and it's possible the library
//! is not as flexible as it should be. Please raise an issue or submit a PR if you see problems.
//!
//!
//! ## License
//! Licensed under [MIT](http://opensource.org/licenses/MIT) license.

mod buffers;
mod context;
mod renderer;
mod shader;
mod task;
mod vertex;

// Re-export public API
pub use buffers::{ImguiBuffers, ImguiBufferConfig, ImguiVirtualBuffers, ImguiTaskNodes, DEFAULT_VERTEX_BUFFER_SIZE, DEFAULT_INDEX_BUFFER_SIZE};
pub use context::ImguiContext;
pub use renderer::{VulkanoRenderer, RendererError, Texture};
pub use task::{HasImguiContext, ImguiFrameData, ImguiUploadTask, ImguiDrawTask};
pub use vertex::Vertex;
