# Batch Chemistry ONNX Benchmarks

This directory contains batch chemistry input files that exercise the ONNX chemistry engine with LSURF isotherm models and ALSURF neural-network and random-forest models. Each case uses `hands_off = true`, the ONNX engine, the `initial` condition from the referenced model config JSON file, verbose gnuplot output, and these shared state/material settings:

- `timestep = 864000`
- `density = 997.16`
- `porosity = 0.25`
- `temperature = 25.0`
- `pressure = 101325.0`
- `volume = 1.0`
- `saturation = 1.0`

Run a case from this directory with the installed batch chemistry driver, for example:

```bash
cd [project_root]/benchmarks/batch_chem
../../build/install/bin/batch_chem alsurf-rf-9-dynamic-batch.cfg
```

## Baseline Input Conditions

The baseline values below come from the `conditions` block in each referenced model config JSON file:

```json
"conditions": {
  "initial": {
    ...
  }
}
```

The model config maps those named values into specific `AlquimiaState` fields and ONNX tensor elements.

### ALSURF Integrated Neural-Network Baseline

Used by:

- `alsurf-nn-integrated-1d.cfg`
- `alsurf-nn-integrated-batch1.cfg`
- `alsurf-nn-integrated-dyn-batch.cfg`

| Feature | Baseline value |
| --- | ---: |
| `H` | `9.999999999999999e-06` |
| `Zn` | `1e-07` |

### ALSURF Six-Feature Random-Forest Baseline

Used by `alsurf-rf-6.cfg`.

| Feature | Baseline value |
| --- | ---: |
| `Zn(OH)2(aq)` | `1.813505366090861e-15` |
| `Zn(OH)3-` | `7.349990948801437e-22` |
| `Zn(OH)4--` | `2.7351872885715046e-29` |
| `Zn++` | `9.960327396019113e-08` |
| `NO3-` | `0.1000000031975435` |
| `Fe++` | `2.9324029589485202e-12` |

### ALSURF Nine-Feature Random-Forest Baseline

Used by:

- `alsurf-rf-9-batch1.cfg`
- `alsurf-rf-9-dynamic-batch.cfg`
- `alsurf-rf-9-feature-vector.cfg`
- `alsurf-rf-9-scalar.cfg`

| Feature | Baseline value |
| --- | ---: |
| `Zn(OH)2(aq)` | `1.813505366090861e-15` |
| `Zn(OH)3-` | `7.349990948801437e-22` |
| `Zn(OH)4--` | `2.7351872885715046e-29` |
| `ZnOH+` | `5.6323585057657926e-12` |
| `Zn++` | `9.960327396019113e-08` |
| `Na+` | `0.0999877981223431` |
| `NO3-` | `0.1000000031975435` |
| `Fe++` | `2.9324029589485202e-12` |
| `O2(aq)` | `0.0002464700291695` |

### LSURF Isotherm Baseline

Used by:

- `isotherms-onnx-1.cfg`
- `isotherms-onnx-2.cfg`
- `isotherms-onnx-5.cfg`
- `isotherms-onnx-6.cfg`

These values are pre-scaled values expected by the LSURF ONNX models. In the inspected configs, the adapter copies the values into ONNX input tensors without applying an additional log or inverse-log transform.

`uranium_total`, `site_density`, `U_aqueous(output)` are `log10` value

| Feature | Baseline value |
| --- | ---: |
| `Mineral_source` | `7` |
| `uranium_total` | `-6.677780705266080` |
| `Site_Density` | `-4.54327863489071` |
| `U_species1` | `-25.419000000000000` |
| `U_species8` | `-22.514000000000000` |
| `U_species14` | `-37.488999999999997` |
| `U_species20` | `-20.510000000000002` |

## Benchmark Cases

This section lists the batch chemistry execution scripts (`.cfg`). For detailed information regarding the internal tensor mappings (Inputs/Outputs) of each model, please refer to the ALSURF/LSURF [model documentation](../../models/READMD.md).

### ALSURF Neural-Network Benchmarks
| Benchmark Config | Description | Model Config | Steps | Output Log |
|---|---|---|---|---|
| `alsurf-nn-integrated-1d.cfg` | NN feature-vector inference | `../../models/alsurf_nn/zn_h_regressor_integrated_1D.json` | 100 | `*.out` |
| `alsurf-nn-integrated-batch1.cfg` | NN fixed-batch inference | `../../models/alsurf_nn/zn_h_regressor_integrated_batch1.json` | 100 | `*.out` |
| `alsurf-nn-integrated-dyn-batch.cfg`| NN dynamic-batch inference | `../../models/alsurf_nn/zn_h_regressor_integrated_dyn_batch.json` | 1 | `*.out` |

### ALSURF Random-Forest Benchmarks
| Benchmark Config | Description | Model Config | Steps | Output Log |
|---|---|---|---|---|
| `alsurf-rf-6.cfg` | 6-feature RF inference | `../../models/alsurf_rf/alsurf_6.json` | 1 | `*.out` |
| `alsurf-rf-9-batch1.cfg` | 9-feature RF fixed-batch | `../../models/alsurf_rfalsurf_9_batch1.json` | 1 | `*.out` |
| `alsurf-rf-9-dynamic-batch.cfg` | 9-feature RF dynamic-batch | `../../models/alsurf_rf/alsurf_9_dynamic_batch.json` | 1 | `*.out` |
| `alsurf-rf-9-feature-vector.cfg` | 9-feature RF feature-vector | `../../models/alsurf_rf/alsurf_9_feature_vector.json` | 1 | `*.out` |
| `alsurf-rf-9-scalar.cfg` | 9-feature RF scalar inference | `../../models/alsurf_rf/alsurf_9_scalar.json` | 1 | `*.out` |

### LSURF Isotherm Benchmarks
| Benchmark Config | Description | Model Config | Steps | Output Log |
|---|---|---|---|---|
| `isotherms-onnx-1.cfg` | 1-feature RF inference | `../../models/lsurf_model_1_float_64.json` | 50 | `*.out` |
| `isotherms-onnx-2.cfg` | 2-feature RF inference | `../../models/lsurf_model_2_float_64.json` | 50 | `*.out` |
| `isotherms-onnx-5.cfg` | 5-feature RF inference | `../../models/lsurf_model_5_float_64.json` | 50 | `*.out` |
| `isotherms-onnx-6.cfg` | 6-feature RF inference | `../../models/lsurf_model_6_float_64.json` | 50 | `*.out` |