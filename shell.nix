{ pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
    config.cudaSupport = true;
    overlays = [ (import ./opencv.nix) ];
  }
}:
let
  pythonPackages = pkgs.python3Packages;
  libs = with pkgs;[
    git gitRepo gnupg autoconf curl
    procps gnumake util-linux m4 gperf unzip
    linuxPackages.nvidia_x11
    opencv
    cudatoolkit
    cudaPackages.cuda_cudart
    cudaPackages.cudnn
    fontconfig
    freetype
    xorg.libX11
    xorg.libXi xorg.libXmu freeglut
    xorg.libXext xorg.libXv xorg.libXrandr
    xorg.libxcb
    xorg.libSM
    xorg.libICE
    xorg.xcbutilwm
    xorg.xcbutilimage
    xorg.xcbutil
    xorg.xcbutilkeysyms
    xorg.xcbutilrenderutil
    libxkbcommon
    stdenv.cc.cc
    dbus
    libGL
    libGLU
    glib
    zlib
    ncurses5 stdenv.cc binutils
  ];
  # python = pkgs.python3.override {
  #   self = python;
  #   packageOverrides = pyfinal: pyprev: {
  #     mediapipe = pyfinal.callPackage ./mediapipe.nix { };
  #   };
  # };
in
with pkgs.libsForQt5;
pkgs.mkShell rec {
  name = "impurePythonEnv";
  venvDir = "./.venv";
  buildInputs = [
    pkgs.opencv
    # A Python interpreter including the 'venv' module is required to bootstrap
    # the environment.
    pythonPackages.python
    pythonPackages.opencv-python
    pythonPackages.numpy
    pythonPackages.cycler
    pythonPackages.fonttools
    pythonPackages.kiwisolver
    pythonPackages.matplotlib
    pythonPackages.pillow
    pythonPackages.protobuf
    pythonPackages.absl-py

    # This executes some shell code to initialize a venv in $venvDir before
    # dropping into the shell
    pythonPackages.venvShellHook
  ];

   # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt --no-deps
  '';

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    export NVIDIA_VISIBLE_DEVICES=all
    export OPENCV_DIR=${pkgs.opencv}

    echo "CUDA C/C++ development environment ready"
    echo "GCC version in use:"
    gcc --version
    echo "NVCC version:"
    nvcc --version
    echo "SHELL NVIDIA driver version:"
    cat ${pkgs.linuxPackages.nvidia_x11}/lib/nvidia/version

    # allow pip to install wheels
    SOURCE_DATE_EPOCH=$(date +%s)
    export QT_DEBUG_PLUGINS=1
    export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath libs}:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.glib.out}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib"
  '';
}
