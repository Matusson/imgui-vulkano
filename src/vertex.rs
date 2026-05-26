#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    bytemuck::Pod,
    bytemuck::Zeroable,
    vulkano::pipeline::graphics::vertex_input::Vertex,
)]
#[repr(C)]
/// Vertex format for imgui rendering.
pub struct Vertex {
    /// 2D position
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    /// Texture coordinates
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    /// RGBA color (normalized u8 values)
    #[format(R8G8B8A8_UNORM)]
    pub col: [u8; 4],
}

impl From<imgui::DrawVert> for Vertex {
    fn from(v: imgui::DrawVert) -> Self {
        Self {
            pos: v.pos,
            uv: v.uv,
            col: v.col,
        }
    }
}
