# Tech Stack

- **Languages:** C (C99/C11) for interfaces/drivers, C++ for wrapper compatibility, modern free-form Fortran.
- **Libraries/Engines:**
  - **ONNX Runtime (ORT):** C API used for inference routing.
  - **cJSON:** Utilized for parsing configuration/model routing JSONs.
- **Build System:** CMake (Superbuild configuration downloading and building standard dependencies).
- **Core Dependencies:**
  - PETSc (Required)
  - MPI (Required)
  - PFlotran (Optional)
  - CrunchFlow (Optional)
  - ONNX Runtime (Embedded/Superbuild dependency)
- **Environments:** Conda (`conda activate geochem_env` for python workflows).