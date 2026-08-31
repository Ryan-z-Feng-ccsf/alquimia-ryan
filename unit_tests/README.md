# Unit Tests

This directory contains CTest unit tests for Alquimia core utilities and ONNX engine behavior.

## Files

| File | Purpose |
|---|---|
| `test_alquimia_c_utils.c` | Tests generic C utility behavior. |
| `test_alquimia_onnx_mapping.c` | Tests ONNX JSON config (`onnx_alquimia_config.c`) parsing, mapping validation, metadata (`GetProblemMetadata`), and relative model paths. |
| `test_alquimia_onnx_failures.c` | Tests ONNX (`onnx_alquimia_interface.c`) setup failures (`Setup`), condition-processing guards (`ProcessCondition`), and lifecycle cleanup (`Shutdown`). |
| `test_alquimia_onnx_inference_routing.c` | Tests ONNX (`ReactionStepOperatorSplit`) input/output routing, model-family compatibility, runtime errors, and ALSURF integration. |
| `onnx_test_utils.c` / `onnx_test_utils.h` | Shared ONNX test helpers for setup, shutdown, paths, state allocation, condition setup, inference, diagnostics, and temporary config files. |
| `onnx_test_cases/` | Test JSON configs and small ONNX artifacts used by the ONNX unit tests. |

ONNX test IDs are file-local. For example, `E01` in the mapping test is a different issue from `E01` in the routing test.

## `onnx_test_cases/`

This folder contains the ONNX unit-test inputs. Tests load these files through `onnx_test_utils.c`, so each test can focus on the behavior being checked instead of path setup. The ONNX model inside are mock models used for testing purpose.

| Folder | Purpose |
|---|---|
| `configs/` | Configs for path-resolution and shared fixture cases (`ALSURF NN JSON config`). |
| `deterministic/` | Small deterministic models and configs for routing, condition, and runtime behavior. |
| `invalid_models/` | Broken configs or model artifacts used to verify setup failure diagnostics and cleanup. |
| `model_families/` | Representative model families used to verify ONNX Runtime compatibility. |

## ID Prefixes

| Prefix | Meaning |
|---|---|
| `Exx` | Deliberate error-handling cases. |
| `Mxx` | Mapping/config success cases. |
| `Cxx` | Standard process condition cases. |
| `Lxx` | Lifecycle and cleanup cases. |
| `Rxx` | Inference/routing cases. |
| `Fxx` | ONNX model-family compatibility cases. |
| `Ixx` | Integration cases. |

## `test_alquimia_onnx_mapping.c`

| ID | Issue Covered |
|---|---|
| `M01` | Named JSON conditions parse successfully, cover required input features, and may include unused extra features. |
| `M02` | Input/output feature names populate the correct `AlquimiaProblemMetaData` vectors by state category. |
| `M03` | Relative ONNX model paths resolve (`ResolveModelPath` at `onnx_alquimia_config.c`) from the config file directory. |
| `E01` | Invalid `conditions` JSON schemas are rejected. |
| `E02` | Missing, malformed, or unsupported `schema_version` is rejected. |
| `E03` | Missing required top-level `model`, `inputs`, or `outputs` is rejected. |
| `E04` | Missing input mapping properties are rejected. |
| `E05` | Missing output mapping properties are rejected. |
| `E06` | Unknown or duplicate JSON properties are rejected. |
| `E07` | Unsupported `alquimia_state` names are rejected. |
| `E08` | Invalid mapping indices are rejected, including negative, fractional, and out-of-range integer values. |
| `E09` | Scalar state mappings reject nonzero `alquimia_state_index`. |
| `E10` | Unknown ONNX tensor names are rejected. |
| `E11` | Tensor element indices outside the flattened tensor extent are rejected. |
| `E12` | Duplicate input/output tensor element mappings are rejected. |
| `E13` | Required model tensor elements that are not mapped are rejected. |
| `E14` | Mapping indices that would derive unsafe or overflowing Alquimia sizes are rejected. |
| `E15` | Two/more conflicting feature names for the same metadata destination are rejected. |
| `E16` | Similar but invalid property names with trailing text are rejected. |
| `E17` | Duplicate input feature names are rejected. |

## `test_alquimia_onnx_failures.c`

| ID | Issue Covered |
|---|---|
| `C01` | `ProcessCondition` accepts a `NULL` condition as a no-op. |
| `C02` | `ProcessCondition` accepts a condition with no aqueous constraints as a no-op. |
| `C03` | Mixed scalar/vector condition constraints route to the correct state fields. |
| `C04` | Named JSON conditions initialize independent cells without sharing state. |
| `C05` | Driver-supplied constraints are used in normal mode. |
| `E01` | Missing ONNX config file fails setup without publishing an engine state. |
| `E02-malformed` | Malformed JSON config fails setup as strict JSON. |
| `E02-trailing` | JSON with trailing content fails setup as strict JSON. |
| `E03` | A raw `.onnx` model path is rejected where a JSON config is required. |
| `E04` | Missing referenced ONNX model file fails setup. |
| `E05` | Non-ONNX model content fails setup through ONNX Runtime. |
| `E06` | Invalid ONNX graph content fails setup through ONNX Runtime. |
| `E07-zero-inputs` | Model with zero inputs is rejected. |
| `E07-zero-outputs` | Model with zero outputs is rejected. |
| `E08` | Unsupported ONNX input tensor element type is rejected. |
| `E09` | Unsupported ONNX output tensor element type is rejected. |
| `E10` | Unsupported dynamic tensor extent is rejected. |
| `E11` | Runtime setup error is reported as an ONNX Runtime setup failure. |
| `E12` | Shutdown after failed setup rejects the `NULL` engine state without changing it. |
| `E13` | `ProcessCondition` rejects a `NULL` state pointer. |
| `E14` | `ProcessCondition` rejects `NULL` mapped state-vector storage. |
| `E15` | `ProcessCondition` rejects undersized mapped state vectors. |
| `E16` | Unknown named JSON condition is rejected without mutating state. |
| `E17` | Hands-off setup requires at least one named JSON condition. |
| `E18` | Shutdown rejects `NULL` shutdown inputs without crashing. |
| `L01` | Repeated setup, inference, and shutdown clean up engine state each time. |

## `test_alquimia_onnx_inference_routing.c`

| ID | Issue Covered |
|---|---|
| `R01` | Single input/output route through one state value. |
| `R02` | Multiple mapped inputs can feed one output tensor value. |
| `R03` | One input tensor value can populate multiple output mappings. |
| `R04` | Multiple input/output tensors keep tensor boundaries distinct. |
| `R05` | Mixed scalar and vector `AlquimiaState` mappings route correctly. |
| `R06` | Outputs across all supported state categories avoid cross-overwrites. |
| `R07` | Rank-0 scalar tensor inputs/outputs are preserved. |
| `R08` | Mobile/immobile paired outputs conserve totals only when one phase is output. |
| `R09` | Repeated inference calls do not reuse stale buffer values. |
| `R10` | Independent ONNX engine instances do not share runtime state or buffers. |
| `E01` | Undersized output vectors fail before out-of-bounds writes. |
| `E02` | ONNX Runtime inference failures become Alquimia engine integrity errors. |
| `F01` | Linear or identity ONNX graphs run through the adapter. |
| `F02` | `ai.onnx.ml` SVR regression models are supported. |
| `F03` | Small neural-network graphs with dense activations are supported. |
| `F04` | `ai.onnx.ml` tree ensemble models are supported. |
| `F05` | Multi-target model outputs route to multiple Alquimia destinations. |
| `F06` | Shared ALSURF neural-network and random-forest model shapes preserve H/Zn routing. |
| `I01` | Full ALSURF lifecycle succeeds: setup, metadata, condition, inference, and shutdown. |