use crate::shader;
use imgui::{TextureId, Textures};
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use vulkano::device::{Device, DeviceOwned, DeviceOwnedVulkanObject, Queue};
use vulkano::format::Format;
use vulkano::image::sampler::SamplerCreateInfo;
use vulkano::image::view::ImageViewCreateInfo;
use vulkano::image::{ImageCreateInfo, ImageType, ImageUsage as ImageUsageFlags};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::vertex_input::{Vertex as VertexTrait, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineShaderStageCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano_taskgraph::descriptor_set::{BindlessContext, SampledImageId, SamplerId};
use vulkano_taskgraph::resource::{
    AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources,
};
use vulkano_taskgraph::{command_buffer::CopyBufferToImageInfo, Id};

/// Error type for renderer operations.
#[derive(Debug)]
pub enum RendererError {
    /// Texture ID not found in the texture library.
    BadTexture(TextureId),
    /// Unsupported image dimensions (must be 2D).
    BadImageDimensions([u32; 3]),
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadTexture(t) => write!(f, "Texture ID not found: {:?}", t),
            Self::BadImageDimensions(d) => {
                write!(f, "Unsupported image dimensions (must be 2D): {:?}", d)
            }
        }
    }
}

impl Error for RendererError {}

/// Texture information for bindless rendering.
#[derive(Clone)]
pub struct Texture {
    pub sampled_image_id: SampledImageId,
    pub sampler_id: SamplerId,
}

/// The main renderer managing textures and graphics pipeline with bindless resources.
///
/// The renderer is responsible for:
/// - Managing the font texture atlas
/// - Managing user textures with bindless IDs
/// - Creating the graphics pipeline
pub struct VulkanoRenderer {
    device: Arc<Device>,
    gamma: f32,

    /// Graphics pipeline created after task graph compilation.
    pipeline: Option<Arc<GraphicsPipeline>>,

    /// Font atlas texture.
    font_texture: Texture,

    /// User texture library.
    textures: Textures<Texture>,
}

impl VulkanoRenderer {
    /// Create a new renderer.
    ///
    /// # Arguments
    ///
    /// * `ctx` - ImGui context (will upload font atlas)
    /// * `device` - Vulkan device
    /// * `queue` - Vulkan queue for font texture upload
    /// * `resources` - Resources for creating GPU buffers and images
    /// * `flight_id` - Flight ID for synchronization during font upload
    /// * `bindless_context` - Bindless context for registering font texture
    /// * `gamma` - Optional gamma correction (default: 1.0)
    ///
    /// # Safety
    /// This function executes a texture upload. It must meet the same safety requirements
    /// as [vulkano_taskgraph::graph::TaskGraph::new].
    pub unsafe fn new(
        ctx: &mut imgui::Context,
        device: Arc<Device>,
        queue: Arc<Queue>,
        resources: &Arc<Resources>,
        flight_id: Id<Flight>,
        bindless_context: &BindlessContext,
        gamma: Option<f32>,
    ) -> Result<Self, Box<dyn Error>> {
        let textures = Textures::new();
        let font_texture =
            Self::upload_font_texture(ctx.fonts(), queue, resources, flight_id, bindless_context)?;

        // Set the font texture ID to the special value
        ctx.fonts().tex_id = TextureId::from(usize::MAX);

        ctx.set_renderer_name(Some(format!(
            "imgui-vulkano-task-renderer {}",
            env!("CARGO_PKG_VERSION")
        )));

        Ok(Self {
            device,
            gamma: gamma.unwrap_or(1.0),
            pipeline: None,
            font_texture,
            textures,
        })
    }

    /// Create the graphics pipeline with a subpass.
    ///
    /// This is typically called from `ImguiTaskNodes::setup_after_compile()` after task graph
    /// compilation. This method is idempotent and safe to call multiple times.
    ///
    /// # Safety
    /// Requirements are the same as [vulkano::shader::ShaderModule::new].
    pub unsafe fn create_pipeline(
        &mut self,
        subpass: Arc<Subpass>,
        bindless_context: &BindlessContext,
    ) -> Result<(), Box<dyn Error>> {
        let vs = shader::vs::load(&self.device)?
            .entry_point("main")
            .ok_or("Failed to load vertex shader")?;

        let fs = shader::fs::load(&self.device)?
            .specialize(&[(0, self.gamma.into())])
            .entry_point("main")
            .ok_or("Failed to load fragment shader")?;

        let vertex_input_state = crate::Vertex::per_vertex().definition(&vs).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(&vs),
            PipelineShaderStageCreateInfo::new(&fs),
        ];

        // Create pipeline layout using bindless context
        let layout = BindlessContext::pipeline_layout_from_stages(bindless_context, &stages)?;

        let pipeline = GraphicsPipeline::new(
            &self.device,
            None,
            &GraphicsPipelineCreateInfo {
                stages: &stages,
                vertex_input_state: Some(&vertex_input_state),
                input_assembly_state: Some(&InputAssemblyState::default()),
                viewport_state: Some(&ViewportState {
                    scissors: &[Scissor::default()],
                    ..Default::default()
                }),
                rasterization_state: Some(&RasterizationState::default()),
                multisample_state: Some(&MultisampleState::default()),
                color_blend_state: Some(&ColorBlendState {
                    attachments: &[ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::alpha()),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                dynamic_state: &[DynamicState::Viewport, DynamicState::Scissor],
                subpass: Some(PipelineSubpassType::BeginRenderPass(&subpass)),
                ..GraphicsPipelineCreateInfo::new(&layout)
            },
        )?;

        // Set debug name if debug utils is enabled
        if self.device.instance().enabled_extensions().ext_debug_utils {
            unsafe {
                pipeline
                    .set_debug_utils_object_name(Some("ImGui Pipeline"))
                    .unwrap();
            }
        }

        self.pipeline = Some(pipeline);

        Ok(())
    }

    /// Get the graphics pipeline.
    ///
    /// Panics if called before a pipeline has been created.
    pub fn pipeline(&self) -> &Arc<GraphicsPipeline> {
        self.pipeline.as_ref().expect("pipeline not yet created")
    }

    /// Check if the pipeline has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.pipeline.is_some()
    }

    /// Reset the pipeline to uninitialized state.
    ///
    /// After calling this, you must call `create_pipeline()` again before rendering.
    pub fn reset_pipeline(&mut self) {
        self.pipeline = None;
    }

    /// Reload the font texture after font changes.
    ///
    /// # Safety
    /// Same requirements as [Self::new]
    pub unsafe fn reload_font_texture(
        &mut self,
        ctx: &mut imgui::Context,
        queue: Arc<Queue>,
        resources: &Arc<Resources>,
        flight_id: Id<Flight>,
        bindless_context: &BindlessContext,
    ) -> Result<(), Box<dyn Error>> {
        self.font_texture =
            Self::upload_font_texture(ctx.fonts(), queue, resources, flight_id, bindless_context)?;

        Ok(())
    }

    /// Get immutable access to the texture library.
    pub fn textures(&self) -> &Textures<Texture> {
        &self.textures
    }

    /// Get mutable access to the texture library.
    pub fn textures_mut(&mut self) -> &mut Textures<Texture> {
        &mut self.textures
    }

    /// Register a texture with the renderer using bindless descriptor IDs.
    pub fn register_texture(
        &mut self,
        sampled_image_id: SampledImageId,
        sampler_id: SamplerId,
    ) -> TextureId {
        let texture = Texture {
            sampled_image_id,
            sampler_id,
        };

        self.textures.insert(texture)
    }

    /// Upload the font atlas texture to the GPU and register with bindless context.
    unsafe fn upload_font_texture(
        fonts: &mut imgui::FontAtlas,
        queue: Arc<Queue>,
        resources: &Arc<Resources>,
        flight_id: Id<Flight>,
        bindless_context: &BindlessContext,
    ) -> Result<Texture, Box<dyn Error>> {
        use vulkano::buffer::{BufferCreateInfo, BufferUsage};
        use vulkano::memory::allocator::DeviceLayout;

        let font_atlas = fonts.build_rgba32_texture();

        // Create image
        let image_id = resources.create_image(
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [font_atlas.width, font_atlas.height, 1],
                usage: ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )?;

        // Create staging buffer
        let data_size = font_atlas.data.len() as u64;
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

        // Set debug names if debug utils is enabled
        if resources
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            unsafe {
                resources
                    .buffer(staging_buffer_id)
                    .buffer()
                    .set_debug_utils_object_name(Some("ImGui Font Staging Buffer"))
                    .unwrap();
                resources
                    .image(image_id)
                    .image()
                    .set_debug_utils_object_name(Some("ImGui Font Texture"))
                    .unwrap();
            }
        }

        // Wait for the flight before using it
        resources.flight(flight_id).wait(None).unwrap();

        // Upload texture data
        unsafe {
            vulkano_taskgraph::execute(
                &queue,
                resources,
                flight_id,
                |cbf, tcx| {
                    // Write data to staging buffer
                    tcx.write_buffer::<[u8]>(staging_buffer_id, ..)
                        .copy_from_slice(font_atlas.data);

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
                            image_extent: [font_atlas.width, font_atlas.height, 1],
                            ..Default::default()
                        }],
                        ..CopyBufferToImageInfo::new()
                    });

                    Ok(())
                },
                [(staging_buffer_id, HostAccessType::Write)],
                [(staging_buffer_id, AccessTypes::COPY_TRANSFER_READ)],
                [(
                    image_id,
                    AccessTypes::COPY_TRANSFER_WRITE,
                    ImageLayoutType::Optimal,
                )],
            )
            .unwrap();
        }

        // Register with bindless context
        let global_set = bindless_context.global_set();
        let sampler_id = global_set
            .create_sampler(&SamplerCreateInfo::simple_repeat_linear())
            .expect("Failed to create sampler");
        let sampled_image_id = global_set
            .create_sampled_image(
                image_id,
                &ImageViewCreateInfo::from_image(resources.image(image_id).image()),
                vulkano::image::ImageLayout::ShaderReadOnlyOptimal,
            )
            .expect("failed to create sampled image");

        Ok(Texture {
            sampled_image_id,
            sampler_id,
        })
    }

    /// Look up a texture by ID.
    ///
    /// Returns the font texture for the special font texture ID.
    pub fn lookup_texture(&self, texture_id: TextureId) -> Result<&Texture, RendererError> {
        if texture_id.id() == usize::MAX {
            Ok(&self.font_texture)
        } else {
            self.textures
                .get(texture_id)
                .ok_or(RendererError::BadTexture(texture_id))
        }
    }
}
