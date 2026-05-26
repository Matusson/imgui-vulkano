mod common;

use imgui::*;
use imgui_vulkano_task_renderer::{HasImguiContext, ImguiContext, VulkanoRenderer, ImguiBuffers, ImguiVirtualBuffers, ImguiTaskNodes};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::cell::RefCell;
use std::sync::Arc;
use vulkano::{
    device::{Device, Queue},
    image::ImageUsage,
    instance::Instance,
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};
use vulkano_taskgraph::{
    graph::{CompileInfo, ExecutableTaskGraph, TaskGraph},
    resource::{Flight, Resources},
    resource_map, Id,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct RenderContext {
    imgui: ImguiContext,

    // Task graph and virtual resources
    executable: Option<ExecutableTaskGraph<RenderContext>>,
    v_swapchain_id: Option<Id<Swapchain>>,
    v_imgui_buffers: Option<ImguiVirtualBuffers>,
    imgui_task_nodes: Option<ImguiTaskNodes>,
}

impl HasImguiContext for RenderContext {
    fn imgui_components(&self) -> (&RefCell<Context>, &RefCell<VulkanoRenderer>) {
        self.imgui.imgui_components()
    }

    fn imgui_frame_data(&self) -> &imgui_vulkano_task_renderer::ImguiFrameData {
        self.imgui.imgui_frame_data()
    }

    fn build_ui(&self, ui: &Ui) {
        ui.window("Hello world")
            .size([400.0, 110.0], Condition::FirstUseEver)
            .build(|| {
                ui.text("Hello world!");
                ui.text("こんにちは世界！");
                ui.text("This is imgui-rs with vulkano-rs, in task graph!");
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });
    }

    fn after_build_ui(&self, ui: &Ui) {
        // Platform-specific integration (winit's prepare_render)
        self.imgui.platform().borrow_mut().prepare_render(ui, self.imgui.window());
    }
}

// SAFETY: RenderContext is only accessed from the main thread during rendering.
// The RefCell interior mutability is safe as long as we don't access it concurrently.
// The ExecutableTaskGraph is only used during rendering and doesn't escape the main thread.
unsafe impl Send for RenderContext {}
unsafe impl Sync for RenderContext {}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    render_context: Option<RenderContext>,
    window: Option<Arc<Window>>,
    surface: Option<Arc<Surface>>,
    flight_id: Id<Flight>,

    // Physical resource IDs
    swapchain_id: Option<Id<Swapchain>>,
    imgui_buffers: Option<ImguiBuffers>, // This example uses only one set of buffers. Check
    // custom_textures example for demonstration of multiple buffers.

    recreate_swapchain: bool,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let ctx = common::VulkanContext::new(event_loop);

        App {
            instance: ctx.instance,
            device: ctx.device,
            queue: ctx.queue,
            resources: ctx.resources,
            render_context: None,
            window: None,
            surface: None,
            flight_id: ctx.flight_id,
            swapchain_id: None,
            imgui_buffers: None,
            recreate_swapchain: false,
        }
    }

    fn initialize_imgui(&mut self) {
        let mut imgui = Context::create();
        imgui.set_ini_filename(None);

        // Initialize platform, winit is used for examples
        let window = self.window.as_ref().unwrap();
        let mut platform = WinitPlatform::new(&mut imgui);
        platform.attach_window(imgui.io_mut(), window, HiDpiMode::Rounded);

        let hidpi_factor = platform.hidpi_factor();
        common::setup_fonts(&mut imgui, hidpi_factor);

        let bindless_context = self.resources.bindless_context()
            .expect("Resources should have bindless context");

        let renderer = VulkanoRenderer::new(
            &mut imgui,
            self.device.clone(),
            self.queue.clone(),
            &self.resources,
            self.flight_id,
            bindless_context,
            Some(2.2f32),
        )
        .expect("Failed to create renderer");

        let imgui_context = ImguiContext::new(
            imgui,
            platform,
            renderer,
            self.window.clone().unwrap(),
        );

        self.render_context = Some(RenderContext {
            imgui: imgui_context,
            executable: None,
            v_swapchain_id: None,
            v_imgui_buffers: None,
            imgui_task_nodes: None,
        });
    }

    fn rebuild_swapchain_and_task_graph(&mut self) {
        let window = self.window.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let window_size = window.inner_size();

        let rcx = self.render_context.as_mut().unwrap();
        rcx.executable = None;

        // Wait for GPU to be idle to ensure all resources are released
        self.resources.wait_idle().unwrap();

        // Select appropriate surface format
        let (image_format, _color_space) = common::select_surface_format(&self.device, surface);

        let swapchain_info = SwapchainCreateInfo {
            min_image_count: 3,
            image_format,
            image_extent: window_size.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
            composite_alpha: vulkano::swapchain::CompositeAlpha::Opaque,
            present_mode: PresentMode::Fifo,
            ..Default::default()
        };

        // Create or recreate physical swapchain
        let swapchain_id = if let Some(old_swapchain_id) = self.swapchain_id {
            self.resources
                .recreate_swapchain(old_swapchain_id, |create_info| SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..*create_info
                })
                .expect("Failed to recreate swapchain")
        } else {
            self.resources
                .create_swapchain(&surface, &swapchain_info)
                .expect("Failed to create swapchain")
        };

        self.swapchain_id = Some(swapchain_id);

        // Create physical imgui buffers if they don't exist
        if self.imgui_buffers.is_none() {
            self.imgui_buffers = Some(
                ImguiBuffers::new(&self.resources)
                    .expect("Failed to create imgui buffers")
            );
        }


        // Build task graph
        let mut task_graph = TaskGraph::new(&self.resources);

        // Get swapchain info
        let swapchain_state = self.resources.swapchain(swapchain_id).unwrap();
        let swapchain_info = SwapchainCreateInfo {
            image_format: swapchain_state.images()[0].format(),
            image_extent: swapchain_state.images()[0].extent()[0..2]
                .try_into()
                .unwrap(),
            image_usage: swapchain_state.images()[0].usage(),
            ..Default::default()
        };

        // Create virtual swapchain with matching parameters
        let v_swapchain_id = task_graph.add_swapchain(&swapchain_info);

        // Get the swapchain's current image ID
        let swapchain_image = v_swapchain_id.current_image_id();

        // Setup ImGui rendering (resources + tasks)
        let buffers = self.imgui_buffers.as_ref().unwrap();
        let v_imgui_buffers = buffers.setup_resources(&mut task_graph);
        let imgui_task_nodes = v_imgui_buffers
            .setup_imgui_tasks(&mut task_graph, swapchain_image, None, None)
            .expect("Failed to setup imgui tasks");

        // Compile task graph
        let mut executable = unsafe {
            task_graph
                .compile(&CompileInfo {
                    queues: &[&self.queue],
                    present_queue: Some(&self.queue),
                    flight_id: self.flight_id,
                    ..Default::default()
                })
                .expect("Failed to compile task graph")
        };

        // Setup imgui draw task pipeline after compilation
        // This is safe to call multiple times during the lifetime of the application
        imgui_task_nodes
            .setup_after_compile(&mut executable, &self.resources, rcx)
            .expect("Failed to setup imgui draw task");

        // Store the virtual resource IDs in RenderContext
        rcx.v_swapchain_id = Some(v_swapchain_id);
        rcx.v_imgui_buffers = Some(v_imgui_buffers);
        rcx.imgui_task_nodes = Some(imgui_task_nodes);
        rcx.executable = Some(executable);
        self.recreate_swapchain = false;
    }

    fn render_frame(&mut self) {
        if self.recreate_swapchain {
            self.rebuild_swapchain_and_task_graph();
        }

        // Wait for the previous frame to finish before starting a new one
        let flight = self.resources.flight(self.flight_id).unwrap();
        flight.wait(None).unwrap();

        let rcx = self.render_context.as_ref().unwrap();
        let executable = rcx.executable.as_ref().unwrap();

        // Prepare imgui frame
        let window = self.window.as_ref().unwrap();
        {
            let mut imgui_ctx = rcx.imgui.imgui_ctx().borrow_mut();
            let platform = rcx.imgui.imgui_platform().borrow_mut();
            platform
                .prepare_frame(imgui_ctx.io_mut(), window)
                .expect("Failed to prepare frame");
        }

        // Map virtual resources to physical resources
        let swapchain_id = self.swapchain_id.unwrap();
        let v_swapchain_id = rcx.v_swapchain_id.unwrap();
        let v_buffers = rcx.v_imgui_buffers.as_ref().unwrap();
        let buffers = self.imgui_buffers.as_ref().unwrap();

        let mut resource_map = resource_map!(
            &executable,
            v_swapchain_id => swapchain_id,
        )
        .expect("Failed to create resource map");

        v_buffers
            .map_buffers(&mut resource_map, buffers)
            .expect("Failed to map imgui buffers");

        // Execute task graph
        match unsafe { executable.execute(resource_map, rcx, || window.pre_present_notify()) } {
            Ok(()) => {}
            Err(e) => {
                eprintln!("Failed to execute task graph: {e:?}");
                self.recreate_swapchain = true;
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("imgui-vulkano-task-renderer: hello_world"),
                )
                .unwrap(),
        );

        let surface = Surface::from_window(&self.instance, &window)
            .expect("Failed to create surface");

        self.window = Some(window);
        self.surface = Some(surface);
        self.initialize_imgui();
        self.rebuild_swapchain_and_task_graph();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(rcx) = &self.render_context {
            rcx.imgui.imgui_platform().borrow_mut().handle_event::<()>(
                rcx.imgui.imgui_ctx().borrow_mut().io_mut(),
                rcx.imgui.window(),
                &winit::event::Event::WindowEvent {
                    window_id: _window_id,
                    event: event.clone(),
                },
            );
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new(&event_loop);
    event_loop
        .run_app(&mut app)
        .expect("Failed to run event loop");
}
