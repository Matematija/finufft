cmake_minimum_required(VERSION 3.19)

project(finufft VERSION 2.2.0 LANGUAGES C CXX)

set(GNU_LIKE_FRONTENDS AppleClang Clang GNU)
if(CMAKE_CXX_COMPILER_ID IN_LIST GNU_LIKE_FRONTENDS)
    # Set custom compiler flags for gcc-compatible compilers
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -funroll-loops")
endif()

include(CTest)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|ppc64|powerpc|powerpc64" OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "ppc|ppc64"))
    # PowerPC arch does not have -march flag.
    set(FINUFFT_ARCH_FLAGS "-mtune=native" CACHE STRING "Compiler flags for specifying target architecture.")
else()
    set(FINUFFT_ARCH_FLAGS "-march=native" CACHE STRING "Compiler flags for specifying target architecture.")
endif()
set(FINUFFT_FFTW_SUFFIX "OpenMP" CACHE STRING "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.)")
set(FINUFFT_FFTW_LIBRARIES "DEFAULT" CACHE STRING "Specify a custom FFTW library")

# All options go here
# sphinx tag (don't remove): @cmake_opts_start
option(FINUFFT_BUILD_EXAMPLES "Whether to build the FINUFFT examples" OFF)
option(FINUFFT_BUILD_TESTS "Whether to build the FINUFFT tests" OFF)
option(FINUFFT_BUILD_FORTRAN "Whether to build the FINUFFT Fortran examples" OFF)
option(FINUFFT_BUILD_MATLAB "Whether to build the FINUFFT Matlab interface" OFF)
option(FINUFFT_ENABLE_SANITIZERS "Whether to enable sanitizers, only effective for Debug configuration." ON)
option(FINUFFT_USE_OPENMP "Whether to use OpenMP for parallelization. If disabled, the finufft library will be single threaded. This does not affect the choice of FFTW library." ON)
option(FINUFFT_USE_CUDA "Whether to build CUDA accelerated FINUFFT library (libcufinufft). This is completely independent of the main FINUFFT library" OFF)
option(FINUFFT_USE_CPU "Whether to build the ordinary FINUFFT library (libfinufft)." ON)
# sphinx tag (don't remove): @cmake_opts_end

if(FINUFFT_USE_CPU)
    set(CPM_DOWNLOAD_VERSION 0.38.0)
    include(cmake/setupCPM.cmake)

    set(FFTW_VERSION 3.3.10)
    include(cmake/setupFFTW.cmake)
endif()

if (FINUFFT_BUILD_MATLAB)
    # When building for matlab, we will fetch the OpenMP library used by matlab
    # instead of system default for compatibility.
    find_package(Matlab REQUIRED)
    find_library(matlab_iomp5_lib NAMES iomp5 HINTS ${Matlab_ROOT_DIR}/sys/os/ PATH_SUFFIXES glnxa64 maci64)
    find_library(pthreads_lib NAMES pthread CMAKE_FIND_ROOT_PATH_BOTH)

    # Create a "fake" imported library pointing to the matlab openmp implementation
    add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_LOCATION ${matlab_iomp5_lib})
    target_link_libraries(OpenMP::OpenMP_CXX INTERFACE ${pthreads_lib})
    # Set the OpenMP flag.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(OpenMP::OpenMP_CXX INTERFACE -Xclang -fopenmp)
    else ()
        target_compile_options(OpenMP::OpenMP_CXX INTERFACE -fopenmp)
    endif ()
else ()
    # For non-matlab builds, find system OpenMP
    if (FINUFFT_USE_OPENMP)
        find_package(OpenMP REQUIRED)
    endif ()
endif ()

# This set of sources is compiled twice, once in single precision and once in double precision
# The single precision compilation is done with -DSINGLE
set(FINUFFT_PRECISION_DEPENDENT_SOURCES src/finufft.cpp src/simpleinterfaces.cpp src/spreadinterp.cpp src/utils.cpp fortran/finufftfort.cpp)

# Set of compilers which behave like gcc
set(FINUFFT_GNU_LIKE_COMPILERS AppleClang Clang GNU)

# Utility function to enable ASAN on debug builds
function(enable_asan target)
    if (NOT FINUFFT_ENABLE_SANITIZERS)
        return()
    endif ()

    if (CMAKE_CXX_COMPILER_ID IN_LIST FINUFFT_GNU_LIKE_COMPILERS)
        # Enable only on clang / gcc compilers.
        target_compile_options(${target} PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=address>)
        target_link_options(${target} PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=address>)
    endif ()
endfunction()

# Utility function to set finufft compilation options.
function(set_finufft_options target)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${target} PROPERTY CMAKE_CXX_STANDARD 14)
    enable_asan(${target})

    target_compile_options(${target} PRIVATE SHELL:$<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Enable cx-limited-range on supported compilers
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:-fcx-limited-range>)
    endif ()

    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    if (FINUFFT_USE_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        # there are issues on windows with OpenMP and CMake, so we need to manually add the flags
        # otherwise there are link errors
        if(WIN32)
            target_link_options(${target} PRIVATE ${OpenMP_CXX_FLAGS})
        endif()
    else ()
        if (CMAKE_CXX_COMPILER_ID IN_LIST FINUFFT_GNU_LIKE_COMPILERS)
            # OpenMP disabled, suppress unknown pragma warnings to avoid spam.
            target_compile_options(${target} PRIVATE "-Wno-unknown-pragmas")
        endif ()
    endif ()

    # FFTW CMAKE file includes the APIs only as an install target, so we need to manually
    # include them since we need them for build not for install
    # trying to include them directly into the fftw and fftwf targets causes issues with
    # the latest version of cmake, so we do it here instead.
    if ( (NOT FFTW_FOUND ) OR (FINUFFT_FFTW_LIBRARIES STREQUAL DOWNLOAD))
        list (GET FINUFFT_FFTW_LIBRARIES 0 element)
        get_property(FFTW_SOURCE_DIR TARGET ${element} PROPERTY SOURCE_DIR)
        set(FFTW_INCLUDE_DIR ${FFTW_SOURCE_DIR}/api)
        target_include_directories(${target} PUBLIC ${FFTW_INCLUDE_DIR})
    endif()

endfunction()

if(FINUFFT_USE_CPU)
    # Main finufft libraries
    add_library(finufft_f32 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
    target_compile_definitions(finufft_f32 PRIVATE SINGLE)
    set_finufft_options(finufft_f32)
    target_link_libraries(finufft_f32 PUBLIC ${FINUFFT_FFTW_LIBRARIES})

    add_library(finufft_f64 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
    set_finufft_options(finufft_f64)
    target_link_libraries(finufft_f64 PUBLIC ${FINUFFT_FFTW_LIBRARIES})

    add_library(finufft SHARED src/utils_precindep.cpp contrib/legendre_rule_fast.cpp)
    set_finufft_options(finufft)
    target_link_libraries(finufft PUBLIC finufft_f32 finufft_f64)
    # windows does not have a math library, so we need to exclude it
    if(NOT WIN32)
        target_link_libraries(finufft PUBLIC m)
    endif()
    target_include_directories(finufft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

    add_library(finufft_static STATIC src/utils_precindep.cpp contrib/legendre_rule_fast.cpp)
    set_finufft_options(finufft)
    target_link_libraries(finufft_static PUBLIC finufft_f32 finufft_f64)
    # windows does not have a math library, so we need to exclude it
    if(NOT WIN32)
        target_link_libraries(finufft_static PUBLIC m)
    endif()
    target_include_directories(finufft_static PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

if(FINUFFT_USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message("FINUFFT WARNING: No CUDA architecture supplied via '-DCMAKE_CUDA_ARCHITECTURES=...', defaulting to '60;70;75;'")
    message("See: https://developer.nvidia.com/cuda-gpus for more details on what architecture to supply.")
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75" CACHE STRING "" FORCE)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(src/cuda)
  if (BUILD_TESTING AND FINUFFT_BUILD_TESTS)
    add_subdirectory(perftest/cuda)
  endif ()
endif()

# Add tests defined in their own directory
if (BUILD_TESTING AND FINUFFT_BUILD_TESTS AND FINUFFT_USE_CPU)
    add_subdirectory(test)
    add_subdirectory(perftest)
endif ()

if (BUILD_TESTING AND FINUFFT_BUILD_TESTS AND FINUFFT_USE_CUDA)
    add_subdirectory(test/cuda)
endif ()

if (FINUFFT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (FINUFFT_BUILD_FORTRAN)
    enable_language(Fortran)
    add_subdirectory(fortran)
endif ()

if (FINUFFT_BUILD_MATLAB)
    add_subdirectory(matlab)
endif ()
