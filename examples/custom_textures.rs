// Custom Textures example for imgui-vulkano-task-renderer
// Demonstrates how to upload and display custom textures

mod common;

use imgui::*;
use imgui_vulkano_task_renderer::{HasImguiContext, ImguiFrameData, VulkanoRenderer, ImguiBuffers, ImguiVirtualBuffers, ImguiTaskNodes};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::cell::RefCell;
use std::error::Error;
use std::sync::Arc;
use vulkano::{
    buffer::{BufferCreateInfo, BufferUsage},
    device::{Device, Queue},
    format::Format,
    image::{ImageCreateInfo, ImageType, ImageUsage as ImageUsageFlags},
    instance::Instance,
    memory::allocator::{AllocationCreateInfo, DeviceLayout},
    image::sampler::{Sampler, SamplerCreateInfo},
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};
use vulkano::image::view::ImageView;
use vulkano_taskgraph::{
    command_buffer::CopyBufferToImageInfo,
    descriptor_set::BindlessContext,
    graph::{CompileInfo, ExecutableTaskGraph, TaskGraph},
    resource::{AccessTypes, Flight, ImageLayoutType, Resources, HostAccessType},
    resource_map, Id,
};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[derive(Default)]
struct CustomTexturesApp {
    my_texture_id: Option<TextureId>,
    peppers: Option<TestTexture>,
}

struct TestTexture {
    texture_id: TextureId,
    size: [f32; 2],
}

impl CustomTexturesApp {
    fn register_textures(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        resources: &Arc<Resources>,
        flight_id: Id<Flight>,
        renderer: &mut VulkanoRenderer,
        bindless_context: &BindlessContext,
    ) -> Result<(), Box<dyn Error>> {
        const WIDTH: usize = 100;
        const HEIGHT: usize = 100;

        if self.my_texture_id.is_none() {
            // Generate dummy texture
            let mut data = Vec::with_capacity(WIDTH * HEIGHT * 4);
            for i in 0..WIDTH {
                for j in 0..HEIGHT {
                    // Insert RGBA values
                    data.push(i as u8);
                    data.push(j as u8);
                    data.push((i + j) as u8);
                    data.push(255_u8);
                }
            }

            // Create image using Resources
            let image_id = resources.create_image(
                &ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_SRGB,
                    extent: [WIDTH as u32, HEIGHT as u32, 1],
                    usage: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
                    ..Default::default()
                },
                &AllocationCreateInfo::default(),
            )?;

            // Create staging buffer using Resources
            let data_size = data.len() as u64;
            let staging_buffer_id = resources.create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: vulkano::memory::allocator::MemoryTypeFilter::PREFER_HOST
                        | vulkano::memory::allocator::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_unsized::<[u8]>(data_size).unwrap(),
            )?;

            // Wait for the flight before using it
            resources.flight(flight_id).wait(None).unwrap();

            // Upload texture data using taskgraph::execute
            unsafe {
                vulkano_taskgraph::execute(
                    &queue,
                    &resources,
                    flight_id,
                    |cbf, tcx| {
                        // Write data to staging buffer
                        tcx.write_buffer::<[u8]>(staging_buffer_id, ..)
                            .copy_from_slice(&data);

                        // Copy staging buffer to image
                        cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                            src_buffer: staging_buffer_id,
                            dst_image: image_id,
                            regions: &[vulkano_taskgraph::command_buffer::BufferImageCopy {
                                buffer_offset: 0,
                                image_subresource: vulkano::image::ImageSubresourceLayers {
                                    aspects: vulkano::image::ImageAspects::COLOR,
                                    mip_level: 0,
                                    base_array_layer: 0,
                                    layer_count: Some(1),
                                },
                                image_offset: [0, 0, 0],
                                image_extent: [WIDTH as u32, HEIGHT as u32, 1],
                                ..Default::default()
                            }],
                            ..CopyBufferToImageInfo::new()
                        });

                        Ok(())
                    },
                    [(staging_buffer_id, HostAccessType::Write)],
                    [(staging_buffer_id, AccessTypes::COPY_TRANSFER_READ)],
                    [(image_id, AccessTypes::COPY_TRANSFER_WRITE, ImageLayoutType::Optimal)],
                )
                .unwrap();
            }

            // Get the image handle from Resources
            let image = resources.image(image_id).image().clone();

            // Create view and sampler
            let image_view = ImageView::new_default(&image)?;
            let sampler = Sampler::new(
                &device,
                &SamplerCreateInfo::simple_repeat_linear(),
            )?;

            // Register with bindless context to get descriptor IDs
            let global_set = bindless_context.global_set();
            let sampler_id = global_set.add_sampler(sampler);
            let sampled_image_id = global_set.add_sampled_image(
                image_view,
                vulkano::image::ImageLayout::ShaderReadOnlyOptimal,
            );

            // Register texture with renderer using bindless IDs
            let texture_id = renderer.register_texture(sampled_image_id, sampler_id);

            self.my_texture_id = Some(texture_id);
        }

        if self.peppers.is_none() {
            self.peppers = Some(TestTexture::new(
                device,
                queue,
                resources,
                flight_id,
                renderer,
                bindless_context,
            )?);
        }

        Ok(())
    }

    fn show_textures(&self, ui: &Ui) {
        ui.window("Hello textures")
            .size([400.0, 600.0], Condition::FirstUseEver)
            .build(|| {
                ui.text("Hello textures!");
                if let Some(my_texture_id) = self.my_texture_id {
                    ui.text("Some generated texture");
                    imgui::Image::new(my_texture_id, [100.0, 100.0]).build(ui);
                }

                if let Some(peppers) = &self.peppers {
                    ui.text("Test image loaded from file:");
                    peppers.show(ui);
                }
            });
    }
}

impl TestTexture {
    fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        resources: &Arc<Resources>,
        flight_id: Id<Flight>,
        renderer: &mut VulkanoRenderer,
        bindless_context: &BindlessContext,
    ) -> Result<Self, Box<dyn Error>> {
        let lenna_bytes = include_bytes!("resources/peppers.jpg");
        let img = image::load_from_memory_with_format(lenna_bytes, image::ImageFormat::Jpeg)?;
        let rgba_img = img.to_rgba8();
        let (width, height) = rgba_img.dimensions();
        let image_data = rgba_img.into_raw();

        // Create image
        let image_id = resources.create_image(
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [width, height, 1],
                usage: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )?;

        // Create staging buffer
        let data_size = image_data.len() as u64;
        let staging_buffer_id = resources.create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: vulkano::memory::allocator::MemoryTypeFilter::PREFER_HOST
                    | vulkano::memory::allocator::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[u8]>(data_size).unwrap(),
        )?;

        // Wait for the flight before using it
        resources.flight(flight_id).wait(None).unwrap();

        // Upload texture dat
        unsafe {
            vulkano_taskgraph::execute(
                &queue,
                &resources,
                flight_id,
                |cbf, tcx| {
                    // Write data to staging buffer
                    tcx.write_buffer::<[u8]>(staging_buffer_id, ..)
                        .copy_from_slice(&image_data);

                    // Copy staging buffer to image
                    cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                        src_buffer: staging_buffer_id,
                        dst_image: image_id,
                        regions: &[vulkano_taskgraph::command_buffer::BufferImageCopy {
                            buffer_offset: 0,
                            image_subresource: vulkano::image::ImageSubresourceLayers {
                                aspects: vulkano::image::ImageAspects::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: Some(1),
                            },
                            image_offset: [0, 0, 0],
                            image_extent: [width, height, 1],
                            ..Default::default()
                        }],
                        ..CopyBufferToImageInfo::new()
                    });

                    Ok(())
                },
                [(staging_buffer_id, HostAccessType::Write)],
                [(staging_buffer_id, AccessTypes::COPY_TRANSFER_READ)],
                [(image_id, AccessTypes::COPY_TRANSFER_WRITE, ImageLayoutType::Optimal)],
            )
            .unwrap();
        }

        // Get the image handle
        let image = resources.image(image_id).image().clone();

        // Create view and sampler
        let image_view = ImageView::new_default(&image)?;
        let sampler = Sampler::new(
            &device,
            &SamplerCreateInfo::simple_repeat_linear(),
        )?;

        // Register with bindless context to get descriptor IDs
        let global_set = bindless_context.global_set();
        let sampler_id = global_set.add_sampler(sampler);
        let sampled_image_id = global_set.add_sampled_image(
            image_view,
            vulkano::image::ImageLayout::ShaderReadOnlyOptimal,
        );

        // Register texture with renderer
        let texture_id = renderer.register_texture(sampled_image_id, sampler_id);

        Ok(TestTexture {
            texture_id,
            size: [width as f32, height as f32],
        })
    }

    fn show(&self, ui: &Ui) {
        imgui::Image::new(self.texture_id, self.size).build(ui);
    }
}

/// Render context that holds imgui components
struct RenderContext {
    imgui_ctx: RefCell<Context>,
    imgui_platform: RefCell<WinitPlatform>,
    renderer: RefCell<VulkanoRenderer>,
    window: Arc<Window>,
    imgui_frame_data: ImguiFrameData,
    app: RefCell<CustomTexturesApp>,

    // Task graph and virtual resources (owned by RenderContext)
    executable: Option<ExecutableTaskGraph<RenderContext>>,
    v_swapchain_id: Option<Id<Swapchain>>,
    v_imgui_buffers: Option<ImguiVirtualBuffers>,
    imgui_task_nodes: Option<ImguiTaskNodes>,
}

impl HasImguiContext for RenderContext {
    fn imgui_components(&self) -> (&RefCell<imgui::Context>, &RefCell<VulkanoRenderer>) {
        (&self.imgui_ctx, &self.renderer)
    }

    fn imgui_frame_data(&self) -> &ImguiFrameData {
        &self.imgui_frame_data
    }

    fn build_ui(&self, ui: &imgui::Ui) {
        let app = self.app.borrow();
        app.show_textures(ui);
    }

    fn after_build_ui(&self, ui: &imgui::Ui) {
        // Platform-specific integration (winit's prepare_render)
        self.imgui_platform.borrow_mut().prepare_render(ui, &self.window);
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
    imgui_buffers: Vec<ImguiBuffers>, // Multiple buffers, one for each frame-in-flight

    recreate_swapchain: bool,
    textures_registered: bool,
    frame_count: usize,
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
            imgui_buffers: Vec::new(),
            recreate_swapchain: false,
            textures_registered: false,
            frame_count: 0,
        }
    }

    fn initialize_imgui(&mut self) {
        let mut imgui = Context::create();
        imgui.set_ini_filename(None);

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
            Some(2.2f32)
        )
        .expect("Failed to create renderer");

        self.render_context = Some(RenderContext {
            imgui_ctx: RefCell::new(imgui),
            imgui_platform: RefCell::new(platform),
            renderer: RefCell::new(renderer),
            window: self.window.clone().unwrap(),
            imgui_frame_data: ImguiFrameData::new(),
            app: RefCell::new(CustomTexturesApp::default()),
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
            min_image_count: 2,
            image_format,
            image_extent: window_size.into(),
            image_usage: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_DST,
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

        // Create physical imgui buffers (one per frame in flight) if they don't exist
        // Using multiple buffer sets allows CPU/GPU parallelism
        const FRAMES_IN_FLIGHT: usize = 3;
        if self.imgui_buffers.is_empty() {
            for _ in 0..FRAMES_IN_FLIGHT {
                let buffers = ImguiBuffers::new(&self.resources)
                    .expect("Failed to create imgui buffers");
                self.imgui_buffers.push(buffers);
            }
        }

        // Build task graph
        let mut task_graph = TaskGraph::new(&self.resources);

        // Create virtual swapchain with matching parameters
        let v_swapchain_id = task_graph.add_swapchain(&swapchain_info);

        // Get the swapchain's current image ID
        let swapchain_image = v_swapchain_id.current_image_id();

        // Setup ImGui rendering (resources + tasks)
        // Use the first buffer to create virtual resources (all buffers have the same configuration)
        let buffers = &self.imgui_buffers[0];
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

        // Register custom textures with bindless context after pipeline creation
        if !self.textures_registered {
            let bindless_context = self.resources.bindless_context()
                .expect("Resources should have bindless context");
            let (_, renderer_ref) = rcx.imgui_components();
            let mut renderer = renderer_ref.borrow_mut();
            let mut app = rcx.app.borrow_mut();

            app.register_textures(
                self.device.clone(),
                self.queue.clone(),
                &self.resources,
                self.flight_id,
                &mut renderer,
                bindless_context,
            )
            .expect("Failed to register textures");

            self.textures_registered = true;
        }

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
        let flight = self.resources.flight(self.flight_id);
        flight.wait(None).unwrap();

        let rcx = self.render_context.as_ref().unwrap();
        let executable = rcx.executable.as_ref().unwrap();

        // Prepare imgui frame
        let window = self.window.as_ref().unwrap();
        {
            let mut imgui_ctx = rcx.imgui_ctx.borrow_mut();
            let platform = rcx.imgui_platform.borrow_mut();
            platform
                .prepare_frame(imgui_ctx.io_mut(), window)
                .expect("Failed to prepare frame");
        }

        // Map virtual resources to physical resources
        let swapchain_id = self.swapchain_id.unwrap();
        let v_swapchain_id = rcx.v_swapchain_id.unwrap();
        let v_buffers = rcx.v_imgui_buffers.as_ref().unwrap();

        // Rotate between buffer sets
        let current_buffers = &self.imgui_buffers[self.frame_count % self.imgui_buffers.len()];

        let mut resource_map = resource_map!(
            &executable,
            v_swapchain_id => swapchain_id,
        )
        .expect("Failed to create resource map");

        v_buffers
            .map_buffers(&mut resource_map, current_buffers)
            .expect("Failed to map imgui buffers");

        self.frame_count += 1;

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
                        .with_title("imgui-vulkano-task-renderer: custom_textures"),
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
            rcx.imgui_platform.borrow_mut().handle_event::<()>(
                rcx.imgui_ctx.borrow_mut().io_mut(),
                &rcx.window,
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
