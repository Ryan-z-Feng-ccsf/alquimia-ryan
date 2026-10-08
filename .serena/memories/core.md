# Core Map & Invariants

## Source Map
- `/alquimia`: Core implementation, engine interfaces (PFlotran, CrunchFlow, ONNX).
  - `alquimia_interface.c/h`: Main entry points.
  - `onnx_alquimia_interface.c/h`: Implementation of ONNX engine.
- `/drivers`: Simulator drivers (`BatchChemDriver.c`, `TransportDriver.c`).
- `/benchmarks/batch_chem`: Integration inputs (`.cfg` files).
- `/unit_tests`: Test suites (`test_alquimia_onnx_*.c`).
- `/models`: Pre-trained models (`models/ex8_nn/*`, `models/ex8_rf/*`).

## Project Invariants
- **Pointer-to-Pointer Dereferencing:** Alquimia drivers pass the address of the engine pointer (`void**` / `OnnxEngineState**`) to all interface function pointers. Always use:
  `onnx_state = *(OnnxEngineState **)onnx_engine_state;`
  to avoid Segmentation Faults (SIGSEGV) across the API boundary.
- **No Direct malloc/free:** Use library allocators for Alquimia structs.
- **Onboarding/Related Memories:**
  - Build/dependencies: `mem:tech_stack`
  - Development workflow commands: `mem:suggested_commands`
  - Formatting/style: `mem:conventions`
  - Definition of Done: `mem:task_completion`