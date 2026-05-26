//! Convenience struct for managing ImGui context components.
//!
//! This module provides [`ImguiContext`], a helper struct that implements [`HasImguiContext`]
//! and handles the boilerplate of storing and managing imgui components.
//!
//! # Usage
//!
//! `ImguiContext` is designed to be wrapped by your application struct. You delegate
//! the boilerplate methods to it while providing your own custom UI:
//!
//! ```ignore
//! struct MyApp {
//!     imgui: ImguiContext,
//!     // Your application state
//!     game_state: GameState,
//! }
//!
//! impl HasImguiContext for MyApp {
//!     fn imgui_components(&self) -> (...) {
//!         self.imgui.imgui_components()  // Delegate to ImguiContext
//!     }
//!
//!     fn imgui_frame_data(&self) -> &ImguiFrameData {
//!         self.imgui.imgui_frame_data()  // Delegate to ImguiContext
//!     }
//!
//!     fn build_ui(&self, ui: &imgui::Ui) {
//!         // Your actual UI goes here
//!         ui.window("Game Stats").build(|| {
//!             ui.text(format!("Score: {}", self.game_state.score));
//!             ui.text(format!("Health: {}", self.game_state.health));
//!         });
//!     }
//! }
//! ```

use crate::{HasImguiContext, ImguiFrameData, VulkanoRenderer};
use imgui_winit_support::WinitPlatform;
use std::cell::RefCell;
use std::sync::Arc;
use winit::window::Window;

/// Helper struct for implementing [`HasImguiContext`] with less boilerplate.
///
/// This struct manages the standard imgui components (context, platform, renderer, frame data)
/// and provides default implementations of [`HasImguiContext`] methods that you can delegate to.
///
/// It is meant to be wrapped by your own application struct, not used directly.
/// The default `build_ui()` implementation is a no-op - you should provide your own UI in your
/// wrapper struct.
///
/// # Thread Safety
///
/// While this struct implements `Send` and `Sync`, it must only be accessed from
/// the main thread during rendering. The `RefCell` interior mutability is safe
/// as long as concurrent access doesn't occur.
///
/// # Examples
///
/// ```ignore
///
/// struct MyApp {
///     imgui: ImguiContext,
///     game_state: GameState,
/// }
///
/// impl HasImguiContext for MyApp {
///     fn imgui_components(&self) -> (...) {
///         self.imgui.imgui_components()  // Delegate
///     }
///
///     fn imgui_frame_data(&self) -> &ImguiFrameData {
///         self.imgui.imgui_frame_data()  // Delegate
///     }
///
///     fn build_ui(&self, ui: &imgui::Ui) {
///         // Your UI goes here
///         ui.window("Stats").build(|| {
///             ui.text(format!("Score: {}", self.game_state.score));
///         });
///     }
/// }
/// ```
pub struct ImguiContext {
    imgui_ctx: RefCell<imgui::Context>,
    imgui_platform: RefCell<WinitPlatform>,
    renderer: RefCell<VulkanoRenderer>,
    window: Arc<Window>,
    imgui_frame_data: ImguiFrameData,
}

impl ImguiContext {
    /// Creates a new `ImguiContext` with the given components.
    pub fn new(
        imgui_ctx: imgui::Context,
        imgui_platform: WinitPlatform,
        renderer: VulkanoRenderer,
        window: Arc<Window>,
    ) -> Self {
        Self {
            imgui_ctx: RefCell::new(imgui_ctx),
            imgui_platform: RefCell::new(imgui_platform),
            renderer: RefCell::new(renderer),
            window,
            imgui_frame_data: ImguiFrameData::new(),
        }
    }

    /// Gets a reference to the imgui context.
    pub fn imgui_ctx(&self) -> &RefCell<imgui::Context> {
        &self.imgui_ctx
    }

    /// Gets a reference to the winit platform.
    pub fn imgui_platform(&self) -> &RefCell<WinitPlatform> {
        &self.imgui_platform
    }

    /// Gets a reference to the renderer.
    pub fn renderer(&self) -> &RefCell<VulkanoRenderer> {
        &self.renderer
    }
}

impl HasImguiContext for ImguiContext {
    fn imgui_components(&self) -> (&RefCell<imgui::Context>, &RefCell<VulkanoRenderer>) {
        (&self.imgui_ctx, &self.renderer)
    }

    fn imgui_frame_data(&self) -> &ImguiFrameData {
        &self.imgui_frame_data
    }

    fn build_ui(&self, _ui: &imgui::Ui) {
        // No-op: ImguiContext is meant to be wrapped.
        // Implement your UI in your wrapper struct's build_ui() method.
    }
}

impl ImguiContext {
    /// Get a reference to the winit platform
    pub fn platform(&self) -> &RefCell<WinitPlatform> {
        &self.imgui_platform
    }

    /// Get a reference to the window
    pub fn window(&self) -> &Window {
        &self.window
    }
}

// SAFETY: ImguiContext is only accessed from the main thread during rendering.
// The RefCell interior mutability is safe as long as we don't access it concurrently.
// The Arc<Window> is thread-safe, and Vulkan resources in Renderer are thread-safe.
unsafe impl Send for ImguiContext {}
unsafe impl Sync for ImguiContext {}
