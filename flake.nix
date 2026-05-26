{
  description = "imgui-vulkano-task-renderer development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" ];
        };
      in
      {
        devShells.default = pkgs.mkShell rec {
          buildInputs = with pkgs; [
            rustToolchain
            pkg-config

            # Vulkan essentials
            shaderc
            vulkan-loader
            vulkan-validation-layers
            vulkan-headers

            # Windowing system dependencies (for winit)
            libxkbcommon
            wayland
            wayland-protocols
            wayland-scanner
            libx11
            libxcursor
            libxi
            libxrandr

            # OpenGL dependencies
            libGL
          ];

          # Set up library paths for dynamic linking
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

          shellHook = ''
            export XDG_RUNTIME_DIR="''${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
            export SHADERC_LIB_DIR="${pkgs.shaderc.lib}/lib"
            export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
            export WAYLAND_DISPLAY="''${WAYLAND_DISPLAY:-wayland-0}"
          '';
        };
      }
    );
}
