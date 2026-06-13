# Use Ubuntu 22.04 as base image (compatible with ubuntu-latest in dev.yml)
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# gcc/gfortran, openmpi, blas/lapack, cmake, python3.10 are requested
# git and make are required for the build process
# libhdf5-openmpi-dev is used in dev.yml and detected by Superbuild
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    cmake \
    python3.10 \
    libopenmpi-dev \
    openmpi-bin \
    libblas-dev \
    liblapack-dev \
    libhdf5-openmpi-dev \
    hdf5-helpers \
    git \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set compilers to MPI wrappers to ensure correct linkage
ENV CC=mpicc
ENV CXX=mpicxx
ENV FC=mpif90

# Copy the Alquimia source code into the container
WORKDIR /alquimia
COPY . .

# Create build directory if it doesn't exist
RUN mkdir -p build

# Set the working directory for the build
WORKDIR /alquimia/build

# Configure Alquimia using the Superbuild option
# This will download and build PETSc, PFlotran, and CrunchFlow
# Binaries will be installed to /alquimia/build/install/bin
RUN cmake .. \
    -DALQUIMIA_SUPERBUILD=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && make

# Add the installed executables to the PATH
# Based on Superbuild.cmake, INSTALL_DIR is ${CMAKE_BINARY_DIR}/install
ENV PATH="/alquimia/build/install/bin:${PATH}"

# Update LD_LIBRARY_PATH so the executables can find the shared libraries
ENV LD_LIBRARY_PATH="/alquimia/build/install/lib"

# Default command
CMD ["/bin/bash"]
