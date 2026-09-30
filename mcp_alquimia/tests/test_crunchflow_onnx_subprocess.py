"""Validate CrunchFlow calcite and ONNX EX8 through persistent subprocesses."""

import ast
import configparser
import json
import shutil
import subprocess

import pytest

from mcp_alquimia import run_crunchflow_calcite, run_onnx_ex8
from mcp_alquimia.subprocess_client import AlquimiaEngineProcess, EngineCommandError


pytestmark = pytest.mark.native

CASES = [
    pytest.param(run_crunchflow_calcite.prepare_case,
                 run_crunchflow_calcite.run_calcite,
                 "calcite-short-cc-crunch.cfg", 50, id="crunchflow-calcite"),
    pytest.param(run_onnx_ex8.prepare_case, run_onnx_ex8.run_ex8,
                 "ex8-nn-batch1.cfg", 100, id="onnx-ex8"),
]


@pytest.mark.parametrize("prepare, run, config_name, steps", CASES)
def test_matches_original_driver(repository_root, engine_process_executable,
                                 batch_chem_executable, tmp_path,
                                 prepare, run, config_name, steps):
    """Compare every state against batch_chem using independently staged assets."""
    result = run(repository_root, engine_process_executable, tmp_path / "subprocess")
    reference = tmp_path / "reference"
    setup = prepare(repository_root, reference)
    # Config Parse (INI/.cfg)
    config = configparser.ConfigParser()
    config.read(repository_root / "benchmarks/batch_chem" / config_name)
    config["chemistry"]["input_file"] = setup["input_file"]
    config["output"] = {"type": "python", "filename": "reference.py"}
    
    # Write the config file
    with (reference / "case.cfg").open("w") as stream:
        config.write(stream)
    # Run the original driver
    with (reference / "driver.log").open("w") as log:
        subprocess.run([str(batch_chem_executable), "case.cfg"], cwd=reference,
                       stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
      
    # Parse literal output without executing generated Python  
    baseline = ast.literal_eval(ast.parse((reference / "reference.py").read_text()).body[0].value)
    history = result["history"]
    assert len(history) == steps + 1
    assert result["final"] == history[-1]
    assert [row["time"] for row in history] == baseline["time"]
    for field, names, key in [
        ("total_mobile", result["metadata"]["primary_species"], "total_mobile"),
        ("total_immobile", range(len(history[0]["total_immobile"])), "total_immobile"),
        ("mineral_volume_fraction", result["metadata"]["minerals"], "mineral_volume_fractions"),
    ]:
        for index, name in enumerate(names):
            assert [row[field][index] for row in history] == pytest.approx(
                baseline[f"{key}[{name}]"], rel=1e-6, abs=1e-14), (field, name)

    if setup["engine"] == "ONNX":
        assert result["metadata"]["has_pH"] is False
        assert result["metadata"]["minerals"] == []
        assert all(row["pH"] is None for row in history)
        assert history[0]["total_mobile"] == pytest.approx([1e-5, 1e-7])
        assert history[0]["total_immobile"] == [0, 0]
        assert history[-1]["total_immobile"] != history[0]["total_immobile"]
        # Native immobile totals use mol/m^3 bulk, mobile totals mol/L water.
        for row in history:
            for index, initial in enumerate(history[0]["total_mobile"]):
                total = row["total_mobile"][index] + row["total_immobile"][index] / 250
                assert total == pytest.approx(initial, rel=1e-10, abs=1e-16)
    else:
        assert result["metadata"]["has_pH"] is True
        assert history[-1]["pH"] == pytest.approx(baseline["pH"][-1], rel=1e-6)
    assert json.loads((tmp_path / "subprocess/simulation_results.json").read_text()) == result


@pytest.mark.parametrize("prepare, run, config_name, steps", CASES)
def test_persistent_reactions_and_initialization_only(repository_root, engine_process_executable,
                                                     tmp_path, prepare, run, config_name, steps):
    """Keep native state across reactions and support a zero-step case run."""
    initial_only = run(repository_root, engine_process_executable, tmp_path / "initial",
                       max_steps=0)
    assert initial_only["history"] == [initial_only["final"]]
    run_dir = tmp_path / "persistent"
    setup = prepare(repository_root, run_dir)
    with AlquimiaEngineProcess(engine_process_executable, run_dir) as process:
        process.request("setup", **setup)
        assert process.request("initialize") == initial_only["final"]
        first = process.request("react", timestep=2)
        final = process.request("react", timestep=3)
        assert (first["time"], final["time"], final["max_steps"]) == (2, 5, 2)
        assert final == process.request("get_results")
    assert process.process.returncode == 0


@pytest.mark.parametrize("prepare, run, config_name, steps", CASES)
def test_unknown_condition_shuts_down(repository_root, engine_process_executable, tmp_path,
                                     prepare, run, config_name, steps):
    """Preserve native initialization errors and cleanly shut down both engines."""
    run_dir = tmp_path / "bad_condition"
    setup = prepare(repository_root, run_dir)
    setup["initial_condition"] = "missing_condition"
    with AlquimiaEngineProcess(engine_process_executable, run_dir) as process:
        process.request("setup", **setup)
        with pytest.raises(EngineCommandError, match="missing_condition"):
            process.request("initialize")
        assert process.process.poll() == 0


def test_onnx_inference_failure_is_reported(repository_root, engine_process_executable, tmp_path):
    """ONNX inference errors remain fatal despite having no convergence flag."""
    run_dir = tmp_path / "runtime_failure"
    setup = run_onnx_ex8.prepare_case(repository_root, run_dir)
    source = repository_root / "unit_tests/onnx_test_cases/deterministic"
    config = json.loads((source / "runtime_failure.json").read_text())
    config["conditions"] = {"initial": {"runtime_input_0": 1, "runtime_input_1": 2}}
    (run_dir / setup["input_file"]).write_text(json.dumps(config))
    shutil.copy2(source / "runtime_failure.onnx", run_dir / "runtime_failure.onnx")
    with AlquimiaEngineProcess(engine_process_executable, run_dir) as process:
        process.request("setup", **setup)
        process.request("initialize")
        with pytest.raises(EngineCommandError, match="react failed") as caught:
            process.request("react", timestep=1)
        assert "did not converge" not in str(caught.value)
        assert process.process.poll() == 0
