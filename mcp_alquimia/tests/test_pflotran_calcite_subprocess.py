"""Verify PFLOTRAN results and native engine process lifecycle against installed engines."""

import ast
import json
import subprocess

import pytest

from mcp_alquimia.run_pflotran_calcite import prepare_case, run_calcite
from mcp_alquimia.subprocess_client import AlquimiaEngineProcess, EngineProcessError, EngineCommandError


pytestmark = pytest.mark.native


def test_calcite_matches_original_driver(repository_root, engine_process_executable,
                                        batch_chem_executable, tmp_path):
    """Compare state history and final pH against the original driver."""
    # Run the engine process
    result = run_calcite(repository_root, engine_process_executable, tmp_path / "engine_process")
    reference = tmp_path / "reference"
    prepare_case(repository_root, reference)
    config = (repository_root /
              "benchmarks/batch_chem/calcite-short-pc-pflotran.cfg").read_text()
    config = config.replace("type = gnuplot", "type = python\nfilename = reference.py")
    (reference / "case.cfg").write_text(config)
    
    # Run the original driver
    with (reference / "driver.log").open("w") as log:
        subprocess.run([str(batch_chem_executable), "case.cfg"], cwd=reference,
                       stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
        
    # Parse literal output without executing generated Python.
    tree = ast.parse((reference / "reference.py").read_text())
    baseline = ast.literal_eval(tree.body[0].value)
    
    # Engine process result
    history = result["history"]
    
    assert len(history) == 51
    assert result["final"] == history[-1]
    assert result["metadata"]["units"]["total_mobile"] == "mol/L water"
    assert result["final"]["total_immobile"] == []
    assert result["final"]["total_mobile"][0] < 0  # Valid H+ total.
    assert [row["time"] for row in history] == baseline["time"]
    
    # Original driver writes only seven significant decimal digits.
    for index, name in enumerate(result["metadata"]["primary_species"]):
        values = [row["total_mobile"][index] for row in history]
        assert values == pytest.approx(baseline[f"total_mobile[{name}]"],
                                       rel=1e-6, abs=1e-14), name
        
    for index, name in enumerate(result["metadata"]["minerals"]):
        values = [row["mineral_volume_fraction"][index] for row in history]
        assert values == pytest.approx(baseline[f"mineral_volume_fractions[{name}]"],
                                       rel=1e-6, abs=1e-14), name
    
    # Original DriverOutput repeats the final pH in every history row.
    assert result["final"]["pH"] == pytest.approx(baseline["pH"][-1], rel=1e-6, abs=1e-14)
    assert history[0]["pH"] != history[-1]["pH"]
    stored = json.loads((tmp_path / "engine_process/simulation_results.json").read_text())
    assert stored == result


def test_persistent_session_and_custom_timestep(engine_process_executable, staged_calcite_case):
    """Verify that successive calls retain state and honor timestep values."""
    run_dir, setup = staged_calcite_case
    with AlquimiaEngineProcess(engine_process_executable, run_dir) as engine_process:
        engine_process.request("setup", **setup)
        initial = engine_process.request("initialize")
        first = engine_process.request("react", timestep=2.0)
        final = engine_process.request("react", timestep=3.0)
        assert final == engine_process.request("get_results")
        assert (initial["time"], first["time"], final["time"]) == (0, 2, 5)
        assert final["max_steps"] == 2
        assert final["total_mobile"][2] > initial["total_mobile"][2]
    assert engine_process.process.returncode == 0


@pytest.mark.parametrize("mode, message", [
    pytest.param("condition", "missing_condition", id="unknown-condition"),
    pytest.param("engine", "missing_engine", id="unknown-engine"),
    pytest.param("order", "Call setup first", id="react-before-setup"),
    pytest.param("negative_dt", "dt_seconds", id="negative-timestep"),
])
def test_native_errors_retain_details_and_shutdown(engine_process_executable, staged_calcite_case,
                                                   mode, message):
    """Verify each native failure retains its diagnosis and shuts down cleanly."""
    run_dir, setup = staged_calcite_case
    if mode == "condition":
        setup["initial_condition"] = "missing_condition"
    elif mode == "engine":
        setup["engine"] = "missing_engine"
    with AlquimiaEngineProcess(engine_process_executable, run_dir) as engine_process:
        with pytest.raises(EngineCommandError, match=message) as caught:
            if mode == "order":
                engine_process.request("react", timestep=5)
            else:
                engine_process.request("setup", **setup)
                engine_process.request("initialize")
                engine_process.request("react", timestep=-5)
        assert str(run_dir / "engine_output.log") in str(caught.value)
        assert engine_process.process.poll() == 0


def test_malformed_protocol_is_reported(engine_process_executable, tmp_path):
    """Verify malformed JSON produces a structured error and permits close."""
    with AlquimiaEngineProcess(engine_process_executable, tmp_path) as engine_process:
        engine_process.process.stdin.write(b"not-json\n")
        engine_process.process.stdin.flush()
        with pytest.raises(EngineCommandError, match="Expected a JSON object"):
            engine_process._receive()
    assert engine_process.process.returncode == 0
