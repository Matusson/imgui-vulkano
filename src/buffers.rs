//! Buffer management for imgui rendering.
//!
//! This module provides utilities for creating and managing the vertex and index buffers
//! needed for imgui.

use std::sync::Arc;
use vulkano::{
    buffer::{AllocateBufferError, BufferCreateInfo, BufferUsage},
    memory::allocator::{AllocationCreateInfo, DeviceLayout},
    Validated,
};
use vulkano::buffer::Buffer;
use vulkano::device::{DeviceOwned, DeviceOwnedVulkanObject};
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano_taskgraph::{
    graph::{NodeId, TaskGraph},
    resource::{AccessTypes, HostAccessType, ImageLayoutType, Resources},
    Id, QueueFamilyType,
};
use crate::task::{HasImguiContext, ImguiDrawTask, ImguiUploadTask};

/// Default vertex buffer size (1MB)
pub const DEFAULT_VERTEX_BUFFER_SIZE: u64 = 1024 * 1024;

/// Default index buffer size (512KB)
pub const DEFAULT_INDEX_BUFFER_SIZE: u64 = 512 * 1024;

/// Configuration for ImGui buffers.
#[derive(Debug, Clone)]
pub struct ImguiBufferConfig {
    /// Size of the vertex buffer in bytes
    pub vertex_buffer_size: u64,
    /// Size of the index buffer in bytes
    pub index_buffer_size: u64,
}

impl Default for ImguiBufferConfig {
    fn default() -> Self {
        Self {
            vertex_buffer_size: DEFAULT_VERTEX_BUFFER_SIZE,
            index_buffer_size: DEFAULT_INDEX_BUFFER_SIZE,
        }
    }
}

/// Stores the physical buffers needed for imgui rendering.
///
/// # Multiple Frames in Flight
///
/// For applications using multiple frames in flight,
/// you have two options:
///
/// 1. Use a single `ImguiBuffers` instance for all frames.
///    This is safe if you wait for the GPU to complete its work before starting work on the next
///    frame. Though, this removes your ability to do pipelining.
///
/// 2. Create one `ImguiBuffers` instance per frame in flight for better CPU/GPU overlap.
///    This allows the CPU to prepare frame N+1 while the GPU is still processing frame N.
pub struct ImguiBuffers {
    /// device-local vertex buffer
    pub vertex_buffer_id: Id<Buffer>,
    /// device-local index buffer
    pub index_buffer_id: Id<Buffer>,
    /// host-visible vertex staging buffer
    pub vertex_staging_id: Id<Buffer>,
    /// host-visible index staging buffer
    pub index_staging_id: Id<Buffer>,
}

impl ImguiBuffers {
    /// Create imgui buffers with default configuration.
    pub fn new(resources: &Arc<Resources>) -> Result<Self, Validated<AllocateBufferError>> {
        Self::new_with_config(resources, &ImguiBufferConfig::default())
    }

    /// Create imgui buffers with custom configuration, for example if custom size is required.
    pub fn new_with_config(
        resources: &Arc<Resources>,
        config: &ImguiBufferConfig,
    ) -> Result<Self, Validated<AllocateBufferError>> {
        let vertex_buffer_id = resources.create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[u8]>(config.vertex_buffer_size).unwrap(),
        )?;

        let index_buffer_id = resources.create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[u8]>(config.index_buffer_size).unwrap(),
        )?;

        let vertex_staging_id = resources.create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[u8]>(config.vertex_buffer_size).unwrap(),
        )?;

        let index_staging_id = resources.create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[u8]>(config.index_buffer_size).unwrap(),
        )?;

        // Set debug names if debug utils is enabled
        if resources.device().instance().enabled_extensions().ext_debug_utils {
            unsafe {
                resources
                    .buffer(vertex_buffer_id)
                    .buffer()
                    .set_debug_utils_object_name(Some("ImGui Vertex Buffer"))
                    .unwrap();
                resources
                    .buffer(index_buffer_id)
                    .buffer()
                    .set_debug_utils_object_name(Some("ImGui Index Buffer"))
                    .unwrap();
                resources
                    .buffer(vertex_staging_id)
                    .buffer()
                    .set_debug_utils_object_name(Some("ImGui Vertex Staging Buffer"))
                    .unwrap();
                resources
                    .buffer(index_staging_id)
                    .buffer()
                    .set_debug_utils_object_name(Some("ImGui Index Staging Buffer"))
                    .unwrap();
            }
        }

        Ok(Self {
            vertex_buffer_id,
            index_buffer_id,
            vertex_staging_id,
            index_staging_id,
        })
    }

    /// Setup imgui resources in a task graph (virtual buffers + host access).
    ///
    /// This is a convenience method that creates virtual buffers and declares
    /// host access for staging buffers. Equivalent to calling
    /// [`ImguiVirtualBuffers::new()`] and [`ImguiVirtualBuffers::add_host_access()`].
    pub fn setup_resources<T>(&self, task_graph: &mut TaskGraph<T>) -> ImguiVirtualBuffers {
        let v_buffers = ImguiVirtualBuffers::new(task_graph);
        v_buffers.add_host_access(task_graph);
        v_buffers
    }
}

/// Virtual buffer IDs for imgui rendering in a task graph.
///
/// These are the virtual buffer resources created in a task graph that
/// will be mapped to physical buffers at execution time.
pub struct ImguiVirtualBuffers {
    pub v_vertex_buffer: Id<Buffer>,
    pub v_index_buffer: Id<Buffer>,
    pub v_vertex_staging: Id<Buffer>,
    pub v_index_staging: Id<Buffer>,
}

impl ImguiVirtualBuffers {
    /// Create virtual buffers for ImGui rendering in a task graph.
    ///
    /// After creating virtual buffers, call [`add_host_access()`](Self::add_host_access)
    /// to declare host access for the staging buffers.
    pub fn new<T>(task_graph: &mut TaskGraph<T>) -> Self {
        let v_vertex_buffer = task_graph.add_buffer(&BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        });

        let v_index_buffer = task_graph.add_buffer(&BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        });

        let v_vertex_staging = task_graph.add_buffer(&BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        });

        let v_index_staging = task_graph.add_buffer(&BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        });

        Self {
            v_vertex_buffer,
            v_index_buffer,
            v_vertex_staging,
            v_index_staging,
        }
    }

    /// Add host access declarations for the staging buffers.
    ///
    /// This declares that the CPU will write to the staging buffers.
    /// Must be called after creating the virtual buffers with [`new()`](Self::new).
    pub fn add_host_access<T>(&self, task_graph: &mut TaskGraph<T>) {
        task_graph.add_host_buffer_access(self.v_vertex_staging, HostAccessType::Write);
        task_graph.add_host_buffer_access(self.v_index_staging, HostAccessType::Write);
    }
}

/// Node IDs for the ImGui upload and draw tasks.
///
/// These are returned by `setup_imgui_tasks()` and can be used to
/// add dependencies to/from other tasks in your graph.
pub struct ImguiTaskNodes {
    /// The upload task node (transfers vertex/index data to GPU)
    pub upload: NodeId,
    /// The draw task node (renders ImGui to target image)
    pub draw: NodeId,
}

impl ImguiTaskNodes {
    /// Returns the first imgui task node.
    ///
    /// Use this when adding dependencies before imgui rendering.
    pub fn first(&self) -> NodeId {
        self.upload
    }

    /// Returns the last imgui task node.
    ///
    /// Use this when adding dependencies after imgui rendering.
    pub fn last(&self) -> NodeId {
        self.draw
    }

    /// Setup the imgui draw task pipeline after task graph compilation.
    ///
    /// This must be called after the task graph is compiled to initialize
    /// the rendering pipeline and register the font texture with the bindless context.
    ///
    /// This function is safe to call multiple times (e.g., during swapchain
    /// recreation). If the pipeline is already initialized, the method returns immediately
    /// without modifying state or re-registering textures.
    pub fn setup_after_compile<T>(
        &self,
        executable: &mut vulkano_taskgraph::graph::ExecutableTaskGraph<T>,
        resources: &Resources,
        context: &T,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        T: HasImguiContext + Send + Sync + 'static,
    {
        use crate::task::ImguiDrawTask;

        // Check if already initialized
        let (_, renderer_ref) = context.imgui_components();
        let is_initialized = renderer_ref.borrow().is_initialized();

        if is_initialized {
            // Already initialized, skip to avoid re-registering textures
            return Ok(());
        }

        // Not initialized, proceed with setup
        let draw_task_node = executable.task_node_mut(self.draw)
            .map_err(|e| format!("Draw task node not found in executable: {e:?}"))?;

        let subpass = draw_task_node.subpass()
            .map(|sp| Arc::new(sp.clone()))
            .ok_or("Task node should have a subpass")?;

        let bindless_context = resources.bindless_context()
            .ok_or("Resources should have bindless context")?;

        unsafe {
            draw_task_node
                .task_mut()
                .downcast_mut::<ImguiDrawTask<T>>()
                .ok_or("Failed to downcast to ImguiDrawTask")?
                .setup_after_compile(subpass, bindless_context, context)?;
        }

        Ok(())
    }
}

impl ImguiVirtualBuffers {
    /// Map ImGui virtual buffers to physical buffers in a resource map.
    ///
    /// This adds all four buffer mappings to the provided resource map. Call this after
    /// creating your resource map.
    pub fn map_buffers(
        &self,
        resource_map: &mut vulkano_taskgraph::graph::ResourceMap<'_>,
        physical_buffers: &ImguiBuffers,
    ) -> Result<(), vulkano_taskgraph::InvalidSlotError> {
        resource_map.insert_buffer(self.v_vertex_buffer, physical_buffers.vertex_buffer_id)?;
        resource_map.insert_buffer(self.v_index_buffer, physical_buffers.index_buffer_id)?;
        resource_map.insert_buffer(self.v_vertex_staging, physical_buffers.vertex_staging_id)?;
        resource_map.insert_buffer(self.v_index_staging, physical_buffers.index_staging_id)?;
        Ok(())
    }

    /// Setup ImGui tasks in the task graph.
    ///
    /// This creates the ImGui upload and draw tasks, sets up their buffer accesses,
    /// and creates the internal taskgraph edges.
    ///
    /// # Arguments
    ///
    /// * `task_graph` - The task graph to add tasks to
    /// * `target_image` - The image to render ImGui to
    /// * `last_pre_imgui_node` - Optional node that the internal tasks connect to. This task
    ///   must complete before imgui tasks start executing.
    ///  * `first_post_imgui_node` - Optional node that the internal tasks connect to. This task
    ///    can begin executing after imgui tasks have completed.
    ///
    /// # Returns
    ///
    /// Returns `ImguiTaskNodes` containing the imgui node IDs, which you
    /// can use to create additional dependencies.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use vulkano_taskgraph::{graph::TaskGraph, resource::Resources, Id};
    /// # use imgui_vulkano_renderer::{ImguiBuffers, HasImguiContext};
    /// # let resources: Arc<Resources> = unimplemented!();
    /// # let mut task_graph = TaskGraph::<MyContext>::new(&resources);
    /// # let target_image: Id<vulkano::image::Image> = unimplemented!();
    /// # struct MyContext;
    /// # impl HasImguiContext for MyContext {
    /// #     fn imgui_components(&self) -> (&std::cell::RefCell<imgui::Context>, &std::cell::RefCell<imgui_winit_support::WinitPlatform>, &std::cell::RefCell<imgui_vulkano_renderer::VulkanoRenderer>) { unimplemented!() }
    /// #     fn window(&self) -> &winit::window::Window { unimplemented!() }
    /// #     fn build_ui(&self, ui: &imgui::Ui) { }
    /// # }
    /// let buffers = ImguiBuffers::new(&resources).unwrap();
    /// let virtual_buffers = buffers.setup_resources(&mut task_graph);
    /// let imgui_nodes = virtual_buffers.setup_imgui_tasks(&mut task_graph, target_image, None).unwrap();
    ///
    /// // You can now add dependencies:
    /// // task_graph.add_edge(some_other_node, imgui_nodes.first()).unwrap();
    /// // task_graph.add_edge(imgui_nodes.last(), yet_another_node).unwrap();
    /// ```
    pub fn setup_imgui_tasks<T>(
        &self,
        task_graph: &mut TaskGraph<T>,
        target_image: Id<vulkano::image::Image>,
        last_pre_imgui_node: Option<NodeId>,
        first_post_imgui_node: Option<NodeId>,
    ) -> Result<ImguiTaskNodes, vulkano_taskgraph::graph::TaskGraphError>
    where
        T: HasImguiContext + Send + Sync + 'static,
    {
        use vulkano_taskgraph::graph::AttachmentInfo;

        // Create framebuffer for imgui rendering
        let framebuffer_id = task_graph.add_framebuffer();

        // Create imgui upload task (runs outside render pass)
        let imgui_upload_task = ImguiUploadTask::<T> {
            v_vertex_buffer: self.v_vertex_buffer,
            v_index_buffer: self.v_index_buffer,
            v_vertex_staging: self.v_vertex_staging,
            v_index_staging: self.v_index_staging,
            _phantom: std::marker::PhantomData,
        };

        let upload_node = task_graph
            .create_task_node("ImGui Upload", QueueFamilyType::Graphics, imgui_upload_task)
            .buffer_access(self.v_vertex_buffer, AccessTypes::COPY_TRANSFER_WRITE)
            .buffer_access(self.v_index_buffer, AccessTypes::COPY_TRANSFER_WRITE)
            .build();


        // Create imgui draw task (runs inside render pass)
        let imgui_draw_task = ImguiDrawTask::<T> {
            v_vertex_buffer: self.v_vertex_buffer,
            v_index_buffer: self.v_index_buffer,
            target: target_image,
            subpass: None,
            _phantom: std::marker::PhantomData,
        };

        let draw_node = task_graph
            .create_task_node("ImGui Draw", QueueFamilyType::Graphics, imgui_draw_task)
            .framebuffer(framebuffer_id)
            .buffer_access(self.v_vertex_buffer, AccessTypes::VERTEX_ATTRIBUTE_READ)
            .buffer_access(self.v_index_buffer, AccessTypes::INDEX_READ)
            .color_attachment(
                target_image,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    index: 0,
                    clear: true,
                    ..Default::default()
                },
            )
            .build();

        task_graph.add_edge(upload_node, draw_node)?;

        // Add dependencies if provided
        if let Some(dep) = last_pre_imgui_node {
            task_graph.add_edge(dep, upload_node)?;
        }

        if let Some(dep) = first_post_imgui_node {
            task_graph.add_edge(draw_node, dep)?;
        }

        Ok(ImguiTaskNodes {
            upload: upload_node,
            draw: draw_node,
        })
    }
}
