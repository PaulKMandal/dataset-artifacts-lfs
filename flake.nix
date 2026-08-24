{
  description = "Reproducible shell for dataset-artifacts cartography experiments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          commonPackages = with pkgs; [
            bashInteractive
            coreutils
            direnv
            findutils
            git
            git-lfs
            gnumake
            jq
            openssh
            pkg-config
            rsync
            uv
            python311
            stdenv.cc.cc.lib
            tmux
            zlib
          ];
          nativeLibraryPath = pkgs.lib.makeLibraryPath (with pkgs; [
            stdenv.cc.cc.lib
            zlib
          ]);
          commonHook = ''
            export UV_PROJECT_ENVIRONMENT="''${UV_PROJECT_ENVIRONMENT:-.venv}"
            export UV_PYTHON="${pkgs.python311}/bin/python"
            export PYTHONNOUSERSITE=1
            export UV_NO_SYNC=1

            # Host NVIDIA driver libraries on non-NixOS GPU servers.
            # Do not add /usr/lib64 directly to LD_LIBRARY_PATH, because it contains
            # system glibc and can break Nix-built tools.
            export DATASET_ARTIFACTS_DRIVER_LIB_DIR="$PWD/.nix-driver-libs"
            mkdir -p "$DATASET_ARTIFACTS_DRIVER_LIB_DIR"

            for root in /run/opengl-driver/lib /usr/lib64 /usr/lib64/nvidia /run/nvidia/driver/usr/lib64; do
                if [ -d "$root" ]; then
                for lib in libcuda.so libcuda.so.1 libnvidia-ml.so.1; do
                    if [ -e "$root/$lib" ]; then
                    ln -sfn "$root/$lib" "$DATASET_ARTIFACTS_DRIVER_LIB_DIR/$lib"
                    fi
                done

                for lib in "$root"/libcuda.so.* "$root"/libnvidia-ml.so.*; do
                    if [ -e "$lib" ]; then
                    ln -sfn "$lib" "$DATASET_ARTIFACTS_DRIVER_LIB_DIR/$(basename "$lib")"
                    fi
                done
                fi
            done

            export LD_LIBRARY_PATH="$DATASET_ARTIFACTS_DRIVER_LIB_DIR:${nativeLibraryPath}:''${LD_LIBRARY_PATH:-}"

            # PyTorch CUDA wheel libraries. These exist after uv sync.
            for p in "$PWD"/.venv/lib/python3.11/site-packages/nvidia/*/lib; do
                if [ -d "$p" ]; then
                export LD_LIBRARY_PATH="$p:''${LD_LIBRARY_PATH:-}"
                fi
            done
          '';
        in
        {
          default = pkgs.mkShell {
            packages = commonPackages;
            shellHook = commonHook + ''
              echo "dataset-artifacts local shell"
              echo "Run: uv sync --extra cpu --group dev"
            '';
          };

          server = pkgs.mkShell {
            packages = commonPackages;
            shellHook = commonHook + ''
              echo "dataset-artifacts GPU server shell"
              echo "Run: uv sync --extra cuda --group dev"
            '';
          };
        });
    };
}
