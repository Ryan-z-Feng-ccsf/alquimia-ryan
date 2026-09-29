"""Run the existing PFLOTRAN calcite case through the native engine process."""

import argparse
import json
import math
from pathlib import Path
import shutil

from .subprocess_client import AlquimiaEngineProcess


def prepare_case(repository: str | Path, run_dir: str | Path):
    """Stage the calcite input and database in a new run directory.

    Args:
        repository (str or Path): Alquimia checkout containing benchmarks/batch_chem.
        run_dir (str or Path): Destination directory, which must not exist.

    Returns:
        dict: Native setup arguments with the benchmark's physical settings.
        Density is kg/m³, temperature is degrees Celsius, pressure is Pa, and
        volume is m³; porosity and saturation are dimensionless.

    Raises:
        FileNotFoundError: A required case input or database is missing.
        FileExistsError: The destination already exists.
        OSError: Creating the directory or copying an asset fails.
    """
    source = Path(repository) / "benchmarks/batch_chem"
    run_dir = Path(run_dir).resolve()
    files = ("calcite-short-pflotran.in", "calcite.dat")
    for file in files:
        if not (source / file).is_file():
            raise FileNotFoundError(source / file)
    run_dir.mkdir(parents=True, exist_ok=False)
    for file in files:
        shutil.copy2(source / file, run_dir / file)
    # These are the physical settings of calcite-short-pc-pflotran.cfg.
    # Chemical constraints remain in the staged native input file.
    return {
        "engine": "PFloTran", 
        "input_file": files[0],
        "initial_condition": "initial", 
        "density": 997.16,
        "porosity": 0.5,
        "temperature": 25.0, 
        "pressure": 101325.0, 
        "volume": 1.0, 
        "saturation": 1.0,
    }


def run_calcite(repository: str | Path, executable: str | Path, run_dir: str | Path, max_steps: int = 50, dt: float = 5.0, timeout: float = 60.0):
    """Run the known calcite case and save its inputs, log, and state history.

    Writes simulation_request.json before launching the engine process and simulation_results.json only after
    successful shutdown. Run failures are recorded in simulation_failure.json when possible.

    Args:
        repository (str or Path): Alquimia checkout supplying the case assets.
        executable (str or Path): Precompiled native engine process executable.
        run_dir (str or Path): New directory for staged inputs and run artifacts.
        max_steps (int): Nonnegative number of reaction max_steps; zero returns only
            the initialized state.
        dt (float): Positive, finite duration of each step in seconds.
        timeout (float): Positive, finite timeout per engine process response in seconds.

    Returns:
        dict: Case identifier, metadata with species/mineral order and units,
        history including the initial state, and the final state snapshot.

    Raises:
        ValueError: Step count, dt, or timeout is invalid.
        FileNotFoundError: The engine process or required case assets are missing.
        FileExistsError: The run directory already exists.
        OSError: Staging, launching, or writing run artifacts fails.
        EngineProcessError: Native setup, stepping, communication, or shutdown fails.
        subprocess.TimeoutExpired: The engine process does not exit after acknowledging
            shutdown within the configured timeout.
    """
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Engine process not built: {executable}. Run build_tools/build_alquimia_engine_process.py first.")
    
    setup = prepare_case(repository, run_dir)
    run_dir = Path(run_dir).resolve()
    request = {
        "case": "calcite-pflotran", 
        "setup": setup,
        "max_steps": max_steps, 
        "timestep": dt,
        "engine_process": str(executable)
    }
    (run_dir / "simulation_request.json").write_text(json.dumps(request, indent=2) + "\n")
    try:
        with AlquimiaEngineProcess(executable, run_dir, timeout) as engine_process:
            metadata = engine_process.request("setup", **setup)
            history = [engine_process.request("initialize")]
            for _ in range(max_steps):
                history.append(engine_process.request("react", timestep=dt))
            result = engine_process.request("get_results")
        final_result = {
            "case": request["case"], 
            "metadata": metadata,
            "history": history, 
            "final": result
        }
        (run_dir / "simulation_results.json").write_text(
            json.dumps(final_result, indent=2, allow_nan=False) + "\n"
            )
        return final_result
    except Exception as error:
        (run_dir / "simulation_failure.json").write_text(
            json.dumps({"error": str(error)}, indent=2) + "\n")
        raise


def main():
    """Run the calcite demonstration using command-line arguments.

    Prints the final state and output directory as JSON on success.

    Raises:
        SystemExit: Help is requested, arguments are invalid, or a handled run
            failure is reported with a nonzero exit status.
    """
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=root)
    parser.add_argument("--executable", type=Path,
                        default=root / "build/mcp_alquimia/bin/alquimia_engine_process")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="New directory for inputs, logs, and structured results")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=5.0, help="Seconds per step")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds per engine process call")
    args = parser.parse_args()
    try:
        result = run_calcite(args.repository, args.executable, args.run_dir,
                             args.max_steps, args.dt, args.timeout)
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(
        json.dumps(
            {
            "run_dir": str(args.run_dir.resolve()), 
            "final_result": result["final"]
            },
            indent=2, 
            allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
