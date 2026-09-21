"""Build the engine process executable against an existing Alquimia installation."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main():
    """Compile only the native engine process using command-line installation paths.

    Creates the output directory, invokes the matching MPI C compiler, and
    prints the resulting executable path. Engines are never rebuilt.

    Raises:
        SystemExit: Help is requested, arguments are invalid, or a required
            installed dependency is missing.
        OSError: Directory creation or starting a build command fails.
        subprocess.CalledProcessError: pkg-config or the compiler fails.
    """
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=root / "build/install")
    parser.add_argument("--output", type=Path,
                        default=root / "build/mcp_alquimia/bin/alquimia_engine_process")
    parser.add_argument("--cc", default=os.environ.get("MPICC", "mpicc"),
                        help="MPI C compiler matching the installed PETSc/MPI build")
    args = parser.parse_args()
    
    prefix = args.prefix.resolve()  # Absolute path
    for name in ("include/alquimia/alquimia.h", "lib/libalquimia.so",
                 "include/cjson/cJSON.h", "lib/pkgconfig/PETSc.pc"):
        if not (prefix / name).is_file():
            parser.error(f"Missing installed dependency: {prefix / name}")
    
    env = dict(os.environ)
    
    # PATH=/root/build/install/lib/pkgconfig:/usr/...
    # Add the directory containing `PETSc.pc'
    # to the PKG_CONFIG_PATH environment variable
    env["PKG_CONFIG_PATH"] = str(prefix / "lib/pkgconfig") + os.pathsep + env.get(
        "PKG_CONFIG_PATH", "")
    
    # ~/alquimia-ryan/build/install/lib$ pkg-config --cflags --libs PETSc
    # -I/home/zfeng3/alquimia-ryan/build/install/include -L/home/zfeng3/alquimia-ryan/build/install/lib -lpetsc
    petsc_flags = shlex.split(subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "PETSc"], env=env, text=True))
    
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    
    command = [args.cc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
               f"-I{prefix / 'include'}",
               str(root / "mcp_alquimia/engine_process/alquimia_engine_process.c"),
               str(root / "mcp_alquimia/engine_process/alquimia_engine_process_helpers.c"),
               f"-L{prefix / 'lib'}", f"-Wl,-rpath,{prefix / 'lib'}",
               "-lalquimia", "-lcjson", *petsc_flags, "-lm", "-o", str(output)]
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
