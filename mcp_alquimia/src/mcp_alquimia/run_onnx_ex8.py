"""Run the existing ONNX EX8 neural-network case through the native engine process."""

from pathlib import Path
import shutil

from .case_runner import run_case, run_case_cli


def prepare_case(repository: str | Path, run_dir: str | Path):
    """Stage the EX8 neural-network config and model in a new run directory.

    Args:
        repository (str or Path): Alquimia checkout containing models/ex8_nn.
        run_dir (str or Path): Destination directory, which must not exist.

    Returns:
        dict: Native setup arguments with the benchmark's physical settings.
        Density is kg/m³, temperature is degrees Celsius, pressure is Pa, and
        volume is m³; porosity and saturation are dimensionless.

    Raises:
        FileNotFoundError: A required case config or model is missing.
        FileExistsError: The destination already exists.
        OSError: Creating the directory or copying an asset fails.
    """
    source = Path(repository) / "models/ex8_nn"
    run_dir = Path(run_dir).resolve()
    files = ("ex8_nn_batch1.json", "ex8_nn_batch1.onnx")
    for file in files:
        if not (source / file).is_file():
            raise FileNotFoundError(source / file)
    run_dir.mkdir(parents=True, exist_ok=False)
    for file in files:
        shutil.copy2(source / file, run_dir / file)
    # These are the physical settings of ex8-nn-batch1.cfg.
    # Chemical constraints remain in the staged native input file.
    return {
        "engine": "ONNX",
        "input_file": files[0],
        "initial_condition": "initial",
        "density": 997.16,
        "porosity": 0.25,
        "temperature": 25.0,
        "pressure": 101325.0,
        "volume": 1.0,
        "saturation": 1.0,
    }


def run_ex8(repository: str | Path, 
            executable: str | Path,
            run_dir: str | Path, 
            max_steps: int = 100, 
            dt: float = 864000.0,
            timeout: float = 60.0):
    """Run ONNX EX8 neural-network with step durations and response timeout in seconds.

    Args:
        repository (str or Path): Checkout supplying the benchmark assets.
        executable (str or Path): Precompiled native engine process.
        run_dir (str or Path): New directory for inputs, logs, and results.
        max_steps (int): Nonnegative reaction count; zero only initializes.
        dt (float): Positive, finite seconds per reaction.
        timeout (float): Positive, finite seconds per response.

    Returns:
        dict: Metadata, initial and reaction history, and final state.

    Raises:
        ValueError: A numerical option is invalid.
        OSError: Assets are missing, the destination exists, or file I/O fails.
        EngineProcessError: Native execution or communication fails.
    """
    return run_case(repository, executable, run_dir, case="ex8-nn-batch1-onnx",
                    prepare_case=prepare_case, max_steps=max_steps, dt=dt,
                    timeout=timeout)


def main():
    """Run the benchmark CLI, printing the final state and artifact directory."""
    run_case_cli(run_ex8, __doc__, max_steps=100, dt=864000.0)


if __name__ == "__main__":
    main()
