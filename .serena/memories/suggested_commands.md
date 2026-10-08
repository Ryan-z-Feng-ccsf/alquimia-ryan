# Suggested Commands

## Build & Compile (ONNX Superbuild)
```bash
# Configure Superbuild
mkdir -p build && cd build
cmake .. \
  -DALQUIMIA_SUPERBUILD=ON \
  -DXSDK_WITH_PFLOTRAN=OFF \
  -DXSDK_WITH_CRUNCHFLOW=OFF \
  -DXSDK_WITH_ONNX=ON

# Compile Project
make -j$(nproc)
```

## Testing & Verification
```bash
# Run All Tests
cd build/alquimia-build
ctest --output-on-failure

# Run Specific Tests (using Regex)
ctest -R onnx
```