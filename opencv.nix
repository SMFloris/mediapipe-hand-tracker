final: prev: {
  opencv4 = prev.opencv4.override {enableGtk2 = true; enableGtk3 = true; enableFfmpeg = true; enableCuda = true; enableUnfree = true;};
}
