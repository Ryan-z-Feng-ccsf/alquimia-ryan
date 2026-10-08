# Conventions

- **Code Style:** Google C++ Style Guide adapted for standard C.
  - Indentation: 2 spaces. No tabs.
  - Encoding: UTF-8.
- **Naming Conventions:**
  - Functions: `CamelCase`
  - Variables: `snake_case`
  - Types/Structs: `CamelCase`
  - Constants/Enums: `kCamelCase`
  - Macros: `UPPER_SNAKE_CASE`
- **Memory Safety:**
  - Allocate and free using `AllocateAlquimia*` / `FreeAlquimia*` helper routines in `alquimia_memory.h`.
  - Always clean up the ONNX state via `onnx_alquimia_shutdown` with zero leaks.
- **Commenting Standards:**
  - Avoid redundant noise comments (e.g. `i++; // increment i`).
  - Use Doxygen/Google-style for all public APIs.
  - Maintain/preserve existing comments and TODOs during refactoring.
  - Details: `mem:code-commenting-standards` (persistent workspace memory).