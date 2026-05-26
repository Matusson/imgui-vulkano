// Common utilities for examples

use imgui::{Context, FontConfig, FontGlyphRanges, FontSource};
use std::sync::Arc;
use vulkano::{
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
        Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    instance::{Instance, InstanceCreateInfo},
    swapchain::{ColorSpace, Surface, SurfaceInfo},
    VulkanLibrary,
};
use vulkano_taskgraph::{
    descriptor_set::{BindlessContext, BindlessContextCreateInfo, GlobalDescriptorSetCreateInfo},
    resource::{Flight, Resources, ResourcesCreateInfo},
    Id,
};
use winit::event_loop::EventLoop;

/// Holds the core Vulkan objects needed for rendering
pub struct VulkanContext {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub resources: Arc<Resources>,
    pub flight_id: Id<Flight>,
}

impl VulkanContext {
    /// Creates a new VulkanContext with default settings for examples
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let library = unsafe { VulkanLibrary::new().expect("Failed to load Vulkan library") };
        let required_extensions = Surface::required_extensions(event_loop);

        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                enabled_extensions: &required_extensions,
                ..Default::default()
            },
        )
        .expect("Failed to create Vulkan instance");

        // As of time of writing, vulkano-taskgraph doesn't support bindful descriptors.
        // As such, bindless extensions are required.
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..BindlessContext::required_extensions(&instance)
        };

        let device_features = DeviceFeatures {
            ..BindlessContext::required_features(&instance)
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter(|p| p.supported_features().contains(&device_features))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(_i, q)| q.queue_flags.intersects(QueueFlags::GRAPHICS))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("No suitable physical device found");

        let (device, mut queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                enabled_extensions: &device_extensions,
                enabled_features: &device_features,
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("Failed to create device");

        let queue = queues.next().unwrap();

        let resources = Resources::new(
            &device,
            &ResourcesCreateInfo {
                bindless_context: Some(&BindlessContextCreateInfo {
                    global_set: &GlobalDescriptorSetCreateInfo::new(),
                    local_set: None,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .expect("Failed to create resources");

        let flight_id = resources.create_flight(3).expect("Failed to create flight");

        VulkanContext {
            instance,
            device,
            queue,
            resources,
            flight_id,
        }
    }
}

/// Sets up imgui fonts with defaults
pub fn setup_fonts(imgui: &mut Context, hidpi_factor: f64) {
    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.fonts().add_font(&[
        FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        },
        FontSource::TtfData {
            data: include_bytes!("../resources/mplus-1p-regular.ttf"),
            size_pixels: font_size,
            config: Some(FontConfig {
                rasterizer_multiply: 1.75,
                glyph_ranges: FontGlyphRanges::japanese(),
                ..FontConfig::default()
            }),
        },
    ]);

    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
}

/// Selects an appropriate swapchain format from the surface's supported formats.
///
/// For the sake of examples, we use an SRGB surface here and specify gamma 2.2 when constructing
/// the renderer. You could also pick an UNORM format and specify gamma 1.0 in the renderer.
pub fn select_surface_format(device: &Device, surface: &Arc<Surface>) -> (Format, ColorSpace) {
    let surface_info = SurfaceInfo::default();
    let surface_formats = device
        .physical_device()
        .surface_formats(surface, &surface_info)
        .expect("Failed to query surface formats");

    // Prefer sRGB
    surface_formats
        .iter()
        .find(|(format, color_space)| {
            matches!(color_space, ColorSpace::SrgbNonLinear)
                && matches!(format, Format::R8G8B8A8_SRGB | Format::B8G8R8A8_SRGB)
        })
        .copied()
        .unwrap_or_else(|| {
            eprintln!(
                "Preferred SRGB format not found, using first available format: {:?}",
                surface_formats[0]
            );
            surface_formats[0]
        })
}
