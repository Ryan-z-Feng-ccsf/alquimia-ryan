"""Shared benchmark cases and original-driver comparisons for subprocess tests."""

import ast
from collections.abc import Callable
import configparser
from dataclasses import dataclass
import subprocess

import pytest

from mcp_alquimia import run_crunchflow_calcite, run_onnx_ex8, run_pflotran_calcite


@dataclass(frozen=True)
class EngineCase:
    """Describe a benchmark independently of its native or Python test layer.

    Attributes:
        name: Readable pytest case ID.
        prepare: Function staging inputs and returning native setup arguments.
        run: Function executing the benchmark with optional numerical overrides.
        config_name: Original-driver config under benchmarks/batch_chem.
        max_steps: Expected default number of reaction steps.
        input_path: Repository-relative input to stage without its companion asset.
    """

    name: str
    prepare: Callable
    run: Callable
    config_name: str
    max_steps: int
    input_path: str


ENGINE_CASES = (
    EngineCase("pflotran-calcite", run_pflotran_calcite.prepare_case,
               run_pflotran_calcite.run_calcite, "calcite-short-pc-pflotran.cfg", 50,
               "benchmarks/batch_chem/calcite-short-pflotran.in"),
    EngineCase("crunchflow-calcite", run_crunchflow_calcite.prepare_case,
               run_crunchflow_calcite.run_calcite, "calcite-short-cc-crunch.cfg", 50,
               "benchmarks/batch_chem/calcite-short-crunch.in"),
    EngineCase("onnx-ex8", run_onnx_ex8.prepare_case, run_onnx_ex8.run_ex8,
               "ex8-nn-batch1.cfg", 100, "models/ex8_nn/ex8_nn_batch1.json"),
)


def run_original_driver(engine_case, repository, executable, run_dir):
    """Run an independently staged batch_chem baseline and parse its literal output.

    Args:
        engine_case: EngineCase describing the benchmark.
        repository: Checkout supplying benchmark assets and configs.
        executable: Precompiled batch_chem executable.
        run_dir: New directory for staged inputs, output, and driver.log.

    Returns:
        dict: Original-driver history parsed without executing generated Python.

    Raises:
        subprocess.SubprocessError: The driver fails or exceeds 60 seconds.
        OSError: Staging, launching, or reading output fails.
    """
    setup = engine_case.prepare(repository, run_dir)
    # Config Parse (INI/.cfg)
    config = configparser.ConfigParser()
    config.read(repository / "benchmarks/batch_chem" / engine_case.config_name)
    config["chemistry"]["input_file"] = setup["input_file"]
    config["output"] = {"type": "python", "filename": "batch_chem_output.py"}
    # Write the config file
    with (run_dir / "batch_chem.cfg").open("w") as stream:
        config.write(stream)
    # Run the original driver
    with (run_dir / "batch_chem_driver.log").open("w") as log:
        subprocess.run([str(executable), "batch_chem.cfg"],
                       cwd=run_dir,
                       stdout=log,
                       stderr=subprocess.STDOUT,
                       timeout=60,
                       check=True)
    # Parse literal output without executing generate Python
    return ast.literal_eval(ast.parse((run_dir / "batch_chem_output.py").read_text()).body[0].value)


def assert_history_matches_driver(result, baseline):
    """Compare native histories and available final pH with driver precision.

    Args:
        result: Subprocess result containing history and species metadata.
        baseline: Parsed original-driver output, rounded to seven significant digits.

    Raises:
        AssertionError: Times, species histories, or final pH differ.
    """
    history = result["history"]
    assert [row["time"] for row in history] == baseline["time"]
    for field, names, key in [
        ("total_mobile", result["metadata"]["primary_species"], "total_mobile"),
        ("total_immobile", range(len(history[0]["total_immobile"])), "total_immobile"),
        ("mineral_volume_fraction", result["metadata"]["minerals"], "mineral_volume_fractions"),
    ]:
        for index, name in enumerate(names):
            assert [row[field][index] for row in history] == pytest.approx(
                baseline[f"{key}[{name}]"], rel=1e-6, abs=1e-14), (field, name)
    if result["metadata"]["has_pH"]:
        # The original driver repeats final pH in every history row.
        assert history[-1]["pH"] == pytest.approx(baseline["pH"][-1], rel=1e-6, abs=1e-14)
