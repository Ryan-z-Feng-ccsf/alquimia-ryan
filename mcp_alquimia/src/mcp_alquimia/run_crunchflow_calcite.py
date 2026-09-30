"""Run the existing CrunchFlow calcite case through the native engine process."""

from pathlib import Path
import shutil

from .case_runner import run_case, run_case_cli


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
    files = ("calcite-short-crunch.in", "calcite.dbs")
    for file in files:
        if not (source / file).is_file():
            raise FileNotFoundError(source / file)
    run_dir.mkdir(parents=True, exist_ok=False)
    for file in files:
        shutil.copy2(source / file, run_dir / file)
    # These are the physical settings of calcite-short-cc-crunch.cfg.
    # Chemical constraints remain in the staged native input file.
    return {
        "engine": "CrunchFlow",
        "input_file": files[0],
        "initial_condition": "initial",
        "density": 997.0751176664442,
        "porosity": 0.5,
        "temperature": 25.0,
        "pressure": 101325.0,
        "volume": 1.0,
        "saturation": 1.0,
    }


def run_calcite(repository: str | Path, 
                executable: str | Path,
                run_dir: str | Path, 
                max_steps: int = 50, 
                dt: float = 5.0,
                timeout: float = 60.0):
    """Run CrunchFlow calcite with step durations and response timeout in seconds.

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
    return run_case(repository, executable, run_dir, case="calcite-crunchflow",
                    prepare_case=prepare_case, max_steps=max_steps, dt=dt,
                    timeout=timeout)


def main():
    """Run the benchmark CLI, printing the final state and artifact directory."""
    run_case_cli(run_calcite, __doc__, max_steps=50, dt=5.0)


if __name__ == "__main__":
    main()
