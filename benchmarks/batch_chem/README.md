# Batch Chemistry ONNX Benchmarks

This directory contains batch chemistry input files that exercise the ONNX chemistry engine with LSURF models and EX8 neural-network and random-forest models. Each case uses `hands_off = true`, the ONNX engine, the `initial` condition from the referenced model config JSON file, verbose gnuplot output, and these shared state/material settings:

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
../../build/install/bin/batch_chem ex8-rf-9-dynamic-batch.cfg
```

## Baseline Input Conditions

> **Note on Units:** The values defined in the JSON `conditions` block follow **Alquimia's standard unit conventions**. However, during mobile and immobile model inference, the actual input and output tensors use **[molarity]**. The Alquimia interface handles the necessary conversions between the Alquimia state and the model tensors.

The baseline values below come from the `conditions` block in each referenced model config JSON file:

```json
"conditions": {
  "initial": {
    ...
  }
}
```

The model config maps those named values into specific `AlquimiaState` fields and ONNX tensor elements.

### EX8 Integrated Neural-Network Baseline

Used by:

- `ex8-nn-1d.cfg`
- `ex8-nn-batch1.cfg`
- `ex8-nn-dynamic-batch.cfg`

| Feature | Baseline value |
| --- | ---: |
| `H` | `9.999999999999999e-06` |
| `Zn` | `1e-07` |

### EX8 Six-Feature Random-Forest Baseline

Used by `ex8-rf-6-dynamic-batch.cfg`.

| Feature | Baseline value |
| --- | ---: |
| `Zn(OH)2(aq)` | `1.813505366090861e-15` |
| `Zn(OH)3-` | `7.349990948801437e-22` |
| `Zn(OH)4--` | `2.7351872885715046e-29` |
| `Zn++` | `9.960327396019113e-08` |
| `NO3-` | `0.1000000031975435` |
| `Fe++` | `2.9324029589485202e-12` |

### EX8 Nine-Feature Random-Forest Baseline

Used by:

- `ex8-rf-9-1d.cfg`
- `ex8-rf-9-batch1.cfg`
- `ex8-rf-9-dynamic-batch.cfg`
- `ex8-rf-9-scalar.cfg`

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

### LSURF Baseline

Used by:

- `lsurf-1-dynamic-batch.cfg`
- `lsurf-2-dynamic-batch.cfg`
- `lsurf-5-dynamic-batch.cfg`
- `lsurf-6-dynamic-batch.cfg`

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

All benchmarks output logs to `*.out`. For internal tensor mappings, refer to the [model documentation](../../models/READMD.md).

### EX8 Neural-Network Benchmarks
| Config | Description | Model | Steps | 
|---|---|---|---|
| `ex8-nn-1d.cfg` | NN feature-vector inference | `../../models/ex8_nn/ex8_nn_1d.json` | 100 |
| `ex8-nn-batch1.cfg` | NN fixed-batch inference | `../../models/ex8_nn/ex8_nn_batch1.json` | 100 |
| `ex8-nn-dynamic-batch.cfg`| NN dynamic-batch inference | `../../models/ex8_nn/ex8_nn_dynamic_batch.json` | 1000 |

### EX8 Random-Forest Benchmarks
| Config | Description | Model | Steps |
|---|---|---|---|
| `ex8-rf-6-dynamic-batch.cfg` | 6-feature RF inference | `../../models/ex8_rf/ex8_rf_6_dynamic_batch.json` | 2 |
| `ex8-rf-9-1d.cfg` | 9-feature RF feature-vector | `../../models/ex8_rf/ex8_rf_9_1d.json` | 4 |
| `ex8-rf-9-batch1.cfg` | 9-feature RF fixed-batch | `../../models/ex8_rf/ex8_rf_9_batch1.json` | 6 |
| `ex8-rf-9-dynamic-batch.cfg` | 9-feature RF dynamic-batch | `../../models/ex8_rf/ex8_rf_9_dynamic_batch.json` | 10 |
| `ex8-rf-9-scalar.cfg` | 9-feature RF scalar inference | `../../models/ex8_rf/ex8_rf_9_scalar.json` | 50 |

### LSURF Benchmarks

The LSURF config and ONNX model files are optional for CTest benchmarks. If either file for a case is missing at test time, CTest reports that case as `Skipped` and continues without failing the suite. Use `ctest -V -R lsurf` to see which file is missing. Files can be added and tests rerun without reconfiguring. A model that is present but fails to load or run still fails the test. CTest 3.16 or newer is needed to display the skipped status.

| Config | Description | Model | Steps |
|---|---|---|---|
| `lsurf-1-dynamic-batch.cfg` | 1-feature RF inference | `../../models/lsurf/lsurf_model_1_float_64.json` | 50 |
| `lsurf-2-dynamic-batch.cfg` | 2-feature RF inference | `../../models/lsurf/lsurf_model_2_float_64.json` | 50 |
| `lsurf-5-dynamic-batch.cfg` | 5-feature RF inference | `../../models/lsurf/lsurf_model_5_float_64.json` | 50 |
| `lsurf-6-dynamic-batch.cfg` | 6-feature RF inference | `../../models/lsurf/lsurf_model_6_float_64.json` | 50 |
