include(ExternalProject)

# Set the install directory for dependencies
set(INSTALL_DIR "${CMAKE_BINARY_DIR}/install")

# Pass down compiler/flags to external projects
set(COMMON_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
)

# Chemistry engines options
option(XSDK_WITH_PFLOTRAN "Enables support for the PFlotran chemistry engine [ON]." ON)
option(XSDK_WITH_CRUNCHFLOW "Enables support for the CrunchFlow chemistry engine [ON]." ON)
option(XSDK_WITH_ONNX "Enables support for the ONNX Runtime chemistry engine [ON]." ON)
option(ALQUIMIA_BUILD_STANDALONE_ENGINES "Build standalone versions of requested chemistry engines [OFF]." OFF)

if (NOT XSDK_WITH_PFLOTRAN AND NOT XSDK_WITH_CRUNCHFLOW AND NOT XSDK_WITH_ONNX)
  message(FATAL_ERROR "At least one chemistry engine must be enabled (XSDK_WITH_PFLOTRAN or XSDK_WITH_CRUNCHFLOW).")
endif()

# Only search for and compile HDF5, BLAS, LAPACK, and PETSc if traditional physical
# engines are enabled. For pure machine learning/ONNX configurations, these are bypassed.
if (XSDK_WITH_PFLOTRAN OR XSDK_WITH_CRUNCHFLOW)
  # Detect system dependencies to avoid unnecessary downloads
  find_package(HDF5 QUIET)
  if(HDF5_FOUND)
    message(STATUS "Found system HDF5: ${HDF5_INCLUDE_DIRS}")
    set(PETSC_HDF5_ARGS "--with-hdf5=1")
  else()
    message(STATUS "HDF5 not found, will be downloaded by PETSc")
    set(PETSC_HDF5_ARGS "--download-hdf5=1")
  endif()

  find_package(BLAS QUIET)
  find_package(LAPACK QUIET)
  if(BLAS_FOUND AND LAPACK_FOUND AND FALSE)
    message(STATUS "Found system BLAS/LAPACK")
    set(PETSC_BLASLAPACK_ARGS "")
  else()
    message(STATUS "BLAS/LAPACK not found, will be downloaded by PETSc")
    set(PETSC_BLASLAPACK_ARGS "--download-fblaslapack=1")
  endif()

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(PETSC_DEBUG_ARG "--with-debugging=1")
  else()
    set(PETSC_DEBUG_ARG "--with-debugging=0")
  endif()

  # PETSc
  ExternalProject_Add(petsc
      GIT_REPOSITORY https://gitlab.com/petsc/petsc.git
      GIT_TAG v3.20.0
      PREFIX ${CMAKE_BINARY_DIR}/external/petsc
      BUILD_IN_SOURCE 1
      UPDATE_DISCONNECTED 1
      CONFIGURE_COMMAND /usr/bin/python3.10 ./configure --prefix=${INSTALL_DIR} --with-mpi=1 ${PETSC_DEBUG_ARG} --with-shared-libraries=1 ${PETSC_HDF5_ARGS} ${PETSC_BLASLAPACK_ARGS}
      BUILD_COMMAND make
      INSTALL_COMMAND make install
  )
endif()

set(ALQUIMIA_DEPS)
set(ALQUIMIA_EXTRA_ARGS)

if (XSDK_WITH_PFLOTRAN OR XSDK_WITH_CRUNCHFLOW)
  list(APPEND ALQUIMIA_DEPS petsc)
endif()

# PFLOTRAN
if (XSDK_WITH_PFLOTRAN)
  ExternalProject_Add(pflotran
      DEPENDS petsc
      GIT_REPOSITORY https://bitbucket.org/pflotran/pflotran
      GIT_TAG v5.0.0
      PREFIX ${CMAKE_BINARY_DIR}/external/pflotran
      CONFIGURE_COMMAND ""
      UPDATE_DISCONNECTED 1
      BUILD_COMMAND make -C src/pflotran libpflotranchem.a PETSC_DIR=${INSTALL_DIR} PETSC_ARCH=""
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/src/pflotran/libpflotranchem.a ${INSTALL_DIR}/lib/libpflotranchem.a
              COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_DIR}/include/pflotran
              COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/src/pflotran ${INSTALL_DIR}/include/pflotran
  )
  list(APPEND ALQUIMIA_DEPS pflotran)
  list(APPEND ALQUIMIA_EXTRA_ARGS 
       -DXSDK_WITH_PFLOTRAN=ON
       -DTPL_PFLOTRAN_LIBRARIES=${INSTALL_DIR}/lib/libpflotranchem.a
       -DTPL_PFLOTRAN_INCLUDE_DIRS=${INSTALL_DIR}/include/pflotran)

  if (ALQUIMIA_BUILD_STANDALONE_ENGINES)
    set(PFLOTRAN_STANDALONE_EXTRA_MAKE_ARGS "")
    if(HDF5_FOUND)
      # Extract library directories from HDF5_LIBRARIES to form HDF5_LIB
      set(HDF5_LDFLAGS_DIR "")
      foreach(lib ${HDF5_LIBRARIES})
        get_filename_component(lib_dir ${lib} DIRECTORY)
        if (HDF5_LDFLAGS_DIR STREQUAL "")
          set(HDF5_LDFLAGS_DIR "${lib_dir}")
        endif()
      endforeach()
      
      # Format includes into HDF5_INCLUDE flags
      # PFLOTRAN makefile does: MYFLAGS += -I$(HDF5_INCLUDE) -I$(HDF5_LIB) ${FC_DEFINE_FLAG}PETSC_HAVE_HDF5
      # We need HDF5_INCLUDE to be space separated with -I prefixes if there are multiple, or just the first one if the makefile only expects one.
      # But wait, looking at the makefile: MYFLAGS += -I$(HDF5_INCLUDE) -I$(HDF5_LIB) 
      # This means it prepends -I. If we have multiple includes, we should pass them through INC or something? No, let's just grab the first include dir or format it.
      # Wait, if we set HDF5_INCLUDE="/dir1 -I/dir2" then it expands to -I/dir1 -I/dir2
      set(HDF5_INCFLAGS "")
      foreach(inc ${HDF5_INCLUDE_DIRS})
        if (HDF5_INCFLAGS STREQUAL "")
          set(HDF5_INCFLAGS "${inc}")
        else()
          set(HDF5_INCFLAGS "${HDF5_INCFLAGS} -I${inc}")
        endif()
      endforeach()

      # Pass these flags to the make call
      # We override LIBS to include hdf5hl_fortran because system HDF5 might separate HL Fortran bindings
      # which the PFLOTRAN makefile doesn't link by default.
      set(PFLOTRAN_STANDALONE_EXTRA_MAKE_ARGS 
          "have_hdf5=1"
          "HDF5_LIB=${HDF5_LDFLAGS_DIR}"
          "HDF5_INCLUDE=${HDF5_INCFLAGS}"
          "LIBS=-L${HDF5_LDFLAGS_DIR} -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz")
    endif()

    ExternalProject_Add(pflotran_standalone
        DEPENDS petsc
        GIT_REPOSITORY https://bitbucket.org/pflotran/pflotran
        GIT_TAG v5.0.0
        PREFIX ${CMAKE_BINARY_DIR}/external/pflotran_standalone
        CONFIGURE_COMMAND ""
        UPDATE_DISCONNECTED 1
        BUILD_COMMAND make -j4 -C src/pflotran pflotran PETSC_DIR=${INSTALL_DIR} PETSC_ARCH="" ${PFLOTRAN_STANDALONE_EXTRA_MAKE_ARGS}
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_DIR}/bin
                COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/src/pflotran/pflotran ${INSTALL_DIR}/bin/pflotran
    )
  endif()
else()
  list(APPEND ALQUIMIA_EXTRA_ARGS -DXSDK_WITH_PFLOTRAN=OFF)
endif()

# CrunchFlow
if (XSDK_WITH_CRUNCHFLOW)
  ExternalProject_Add(crunchflow
      DEPENDS petsc
      GIT_REPOSITORY https://bitbucket.org/crunchflow/crunchtope-dev
      GIT_TAG master
      PREFIX ${CMAKE_BINARY_DIR}/external/crunchflow
      CONFIGURE_COMMAND ""
      UPDATE_DISCONNECTED 1
      PATCH_COMMAND git apply --check source/MakefileForAlquimia.patch && git apply source/MakefileForAlquimia.patch || echo "Patch already applied or failed"
      BUILD_COMMAND make -C source libcrunchchem.a PETSC_DIR=${INSTALL_DIR} PETSC_ARCH=""
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/source/libcrunchchem.a ${INSTALL_DIR}/lib/libcrunchchem.a
              COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_DIR}/include/crunchflow
              COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/source ${INSTALL_DIR}/include/crunchflow
  )
  list(APPEND ALQUIMIA_DEPS crunchflow)
  list(APPEND ALQUIMIA_EXTRA_ARGS 
       -DXSDK_WITH_CRUNCHFLOW=ON
       -DTPL_CRUNCHFLOW_LIBRARIES=${INSTALL_DIR}/lib/libcrunchchem.a
       -DTPL_CRUNCHFLOW_INCLUDE_DIRS=${INSTALL_DIR}/include/crunchflow)

  if (ALQUIMIA_BUILD_STANDALONE_ENGINES)
    ExternalProject_Add(crunchflow_standalone
        DEPENDS petsc
        GIT_REPOSITORY https://bitbucket.org/crunchflow/crunchtope-dev
        GIT_TAG master
        PREFIX ${CMAKE_BINARY_DIR}/external/crunchflow_standalone
        CONFIGURE_COMMAND ""
        UPDATE_DISCONNECTED 1
        PATCH_COMMAND sed -i "s/chkopts//g" source/Makefile
        BUILD_COMMAND make -C source CrunchMain PETSC_DIR=${INSTALL_DIR} PETSC_ARCH=""
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_DIR}/bin
                COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/source/CrunchTope ${INSTALL_DIR}/bin/crunchflow
    )
  endif()
else()
  list(APPEND ALQUIMIA_EXTRA_ARGS -DXSDK_WITH_CRUNCHFLOW=OFF)
endif()

# ==============================================================================
# ONNX Runtime Chemistry Engine Integration
# ==============================================================================
# This enables high-performance machine learning-based chemistry engines
# within the Alquimia framework. This block manages the automated retrieval
# and isolated filesystem packaging of ONNX Runtime's pre-built binaries.
# ==============================================================================
if (XSDK_WITH_ONNX)
  # cJSON publishes source archives rather than pre-built binaries. Build and
  # install its shared library into the same private prefix used for the ONNX
  # Runtime package so the inner Alquimia build sees one dependency boundary.
  ExternalProject_Add(cjson
      URL "https://github.com/DaveGamble/cJSON/archive/refs/tags/v1.7.19.tar.gz"
      URL_HASH SHA256=7fa616e3046edfa7a28a32d5f9eacfd23f92900fe1f8ccd988c1662f30454562
      PREFIX ${CMAKE_BINARY_DIR}/external/cjson
      CMAKE_ARGS
          ${COMMON_CMAKE_ARGS}
          -DBUILD_SHARED_LIBS=ON
          -DBUILD_SHARED_AND_STATIC_LIBS=OFF
          -DENABLE_CJSON_UTILS=OFF
          -DENABLE_CJSON_TEST=OFF
          -DENABLE_TARGET_EXPORT=OFF
          -DENABLE_CUSTOM_COMPILER_FLAGS=OFF
          -DCMAKE_INSTALL_LIBDIR=lib
          -DCMAKE_INSTALL_INCLUDEDIR=include
  )
  ExternalProject_Add_Step(cjson stage_license
      COMMAND ${CMAKE_COMMAND} -E make_directory
              ${INSTALL_DIR}/share/licenses/cjson
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              <SOURCE_DIR>/LICENSE
              ${INSTALL_DIR}/share/licenses/cjson/LICENSE
      DEPENDEES build
      DEPENDERS install
  )

  # Declare ONNX pre-built binary package as an ExternalProject target.
  # Since this uses official pre-compiled assets from Microsoft, the configure
  # and build commands are left empty to bypass compilation overhead.
  # Filesystem encapsulation is strictly enforced at the installation stage.
  ExternalProject_Add(onnx
      URL "https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-x64-1.27.0.tgz"
      PREFIX ${CMAKE_BINARY_DIR}/external/onnx
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_DIR}/include/onnxruntime
              COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${INSTALL_DIR}/include/onnxruntime
              COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_DIR}/lib
              COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/lib/libonnxruntime.so ${INSTALL_DIR}/lib/libonnxruntime.so
              COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/lib/libonnxruntime.so.1.27.0 ${INSTALL_DIR}/lib/libonnxruntime.so.1.27.0
              COMMAND ${CMAKE_COMMAND} -E create_symlink libonnxruntime.so ${INSTALL_DIR}/lib/libonnxruntime.so.1
  )

  # Register 'onnx' as a hard dependency of Alquimia Core.
  # This ensures the ONNX Runtime assets are fully downloaded and organized
  # in the installation directory BEFORE the inner Alquimia build is configured.
  list(APPEND ALQUIMIA_DEPS cjson onnx)

  # Route the build arguments down to the inner-core Alquimia project.
  # This cleanly exposes ONNX integration flags and targets.
  list(APPEND ALQUIMIA_EXTRA_ARGS
       -DXSDK_WITH_ONNX=ON
       -DTPL_ONNX_LIBRARIES=${INSTALL_DIR}/lib/libonnxruntime.so
       -DTPL_ONNX_INCLUDE_DIRS=${INSTALL_DIR}/include/onnxruntime
       -DTPL_CJSON_LIBRARIES=${INSTALL_DIR}/lib/libcjson${CMAKE_SHARED_LIBRARY_SUFFIX}
       -DTPL_CJSON_INCLUDE_DIRS=${INSTALL_DIR}/include/cjson)
else()
  # Gracefully report ONNX exclusion to the inner build system
  list(APPEND ALQUIMIA_EXTRA_ARGS -DXSDK_WITH_ONNX=OFF)
endif()

# Dynamic configuration arguments for PETSc forwarding
set(ALQUIMIA_CORE_PETSC_ARGS)
if (XSDK_WITH_PFLOTRAN OR XSDK_WITH_CRUNCHFLOW)
  set(ALQUIMIA_CORE_PETSC_ARGS
      -DPETSC_DIR=${INSTALL_DIR}
      -DPETSC_ARCH=.
  )
endif()

# Alquimia itself
ExternalProject_Add(alquimia_core
    DEPENDS ${ALQUIMIA_DEPS}
    SOURCE_DIR ${CMAKE_SOURCE_DIR}
    BINARY_DIR ${CMAKE_BINARY_DIR}/alquimia-build
    INSTALL_DIR ${INSTALL_DIR}
    CMAKE_ARGS
        ${COMMON_CMAKE_ARGS}
        ${ALQUIMIA_EXTRA_ARGS}
        ${ALQUIMIA_CORE_PETSC_ARGS}
        -DALQUIMIA_SUPERBUILD=OFF
)

# Forward the test target to the inner build
# We remove the dependency on alquimia_core so that 'make test' doesn't 
# trigger a re-check of all dependencies.
add_custom_target(test
    COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${INSTALL_DIR}/lib:$ENV{LD_LIBRARY_PATH} ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/alquimia-build --target test
)

# Rule to install the contents of the local install directory to the final destination
install(DIRECTORY ${INSTALL_DIR}/ DESTINATION .)
