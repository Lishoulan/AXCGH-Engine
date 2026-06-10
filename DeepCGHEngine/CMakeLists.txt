# ============================================================================
# DeepCGHEngine — CMake Build Configuration
# ============================================================================
#
# Build the core C++ library and optional Python bindings.
#
# Usage (MinGW):
#   cmake -B build -G "MinGW Makefiles" -DONNXRUNTIME_ROOT=<path>
#   cmake --build build --config Release
#
# Options:
#   DEEPCGH_BUILD_PYTHON=ON   Build PyBind11 Python module (default: ON)
#   DEEPCGH_BUILD_APPS=OFF    Build standalone demo applications (default: OFF)
#   DEEPCGH_BUILD_TESTS=OFF   Build unit tests (default: OFF)
#   ONNXRUNTIME_ROOT=""       Path to ONNX Runtime SDK root
#
# ============================================================================

cmake_minimum_required(VERSION 3.15)

project(DeepCGHEngine
    VERSION     1.0.0
    LANGUAGES   CXX
    DESCRIPTION "Deep-learning Computer-Generated Holography Engine")

# ---------------------------------------------------------------------------
# C++ Standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(DEEPCGH_BUILD_PYTHON "Build Python bindings via PyBind11" ON)
option(DEEPCGH_BUILD_TESTS  "Build unit tests" OFF)
option(DEEPCGH_BUILD_APPS   "Build standalone demo applications" OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime SDK")
set(FFTW3_ROOT "" CACHE PATH "Root directory of FFTW3 (containing include/ and lib/)")

# ---------------------------------------------------------------------------
# Find FFTW3
# ---------------------------------------------------------------------------
if(FFTW3_ROOT)
    set(FFTW3_INCLUDE_DIRS "${FFTW3_ROOT}")
    if(MINGW)
        set(FFTW3_LIBRARY "${FFTW3_ROOT}/libfftw3f-3.dll.a")
        if(NOT EXISTS "${FFTW3_LIBRARY}")
            message(FATAL_ERROR
                "FFTW3 import library not found at ${FFTW3_LIBRARY}. "
                "Please verify FFTW3_ROOT points to a valid FFTW3 installation.")
        endif()
        set(FFTW3_DLL "${FFTW3_ROOT}/libfftw3f-3.dll")
        add_library(FFTW3::FFTW3 SHARED IMPORTED)
        set_target_properties(FFTW3::FFTW3 PROPERTIES
            IMPORTED_LOCATION "${FFTW3_DLL}"
            IMPORTED_IMPLIB   "${FFTW3_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        )
        message(STATUS "FFTW3f found (MinGW): ${FFTW3_DLL}")
    else()
        find_library(FFTW3_LIBRARY
            NAMES fftw3f libfftw3f-3
            PATHS "${FFTW3_ROOT}"
            NO_DEFAULT_PATH)
        add_library(FFTW3::FFTW3 SHARED IMPORTED)
        set_target_properties(FFTW3::FFTW3 PROPERTIES
            IMPORTED_LOCATION "${FFTW3_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        )
        message(STATUS "FFTW3f found: ${FFTW3_LIBRARY}")
    endif()
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3f)
    endif()
    if(NOT FFTW3_FOUND)
        find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h)
        find_library(FFTW3_LIBRARY NAMES fftw3f libfftw3f-3)
        if(FFTW3_INCLUDE_DIRS AND FFTW3_LIBRARY)
            add_library(FFTW3::FFTW3 SHARED IMPORTED)
            set_target_properties(FFTW3::FFTW3 PROPERTIES
                IMPORTED_LOCATION "${FFTW3_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
            )
            message(STATUS "FFTW3f found: ${FFTW3_LIBRARY}")
        else()
            message(FATAL_ERROR
                "FFTW3 not found. Set -DFFTW3_ROOT=<path> containing fftw3.h and libfftw3f-3.dll.a")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Find ONNX Runtime
# ---------------------------------------------------------------------------
if(ONNXRUNTIME_ROOT)
    set(OnnxRuntime_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    set(OnnxRuntime_LIB_DIR      "${ONNXRUNTIME_ROOT}/lib")

    # For MinGW: the .lib is MSVC format, so we link directly against the DLL
    if(MINGW)
        # MinGW can link directly against the DLL
        set(OnnxRuntime_DLL "${OnnxRuntime_LIB_DIR}/onnxruntime.dll")

        if(NOT EXISTS "${OnnxRuntime_DLL}")
            message(FATAL_ERROR
                "ONNX Runtime DLL not found at ${OnnxRuntime_DLL}. "
                "Please verify ONNXRUNTIME_ROOT points to a valid SDK.")
        endif()

        # Generate import library for MinGW from the DLL
        execute_process(
            COMMAND ${CMAKE_DLLTOOL} --input-def "${OnnxRuntime_LIB_DIR}/onnxruntime.def"
                    --output-lib "${CMAKE_BINARY_DIR}/libonnxruntime.dll.a"
            RESULT_VARIABLE DLLTOOL_RESULT
            ERROR_QUIET
        )

        # If .def doesn't exist, generate it from the DLL using gendef or dlltool
        if(NOT DLLTOOL_RESULT EQUAL 0)
            # Try using the .lib file directly with MinGW (sometimes works)
            # Or generate a .def from the DLL
            find_program(GENDEF gendef PATHS "${OnnxRuntime_LIB_DIR}" NO_DEFAULT_PATH)
            if(GENDEF)
                execute_process(COMMAND ${GENDEF} "${OnnxRuntime_DLL}"
                                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
                execute_process(
                    COMMAND ${CMAKE_DLLTOOL} --input-def "${CMAKE_BINARY_DIR}/onnxruntime.def"
                            --output-lib "${CMAKE_BINARY_DIR}/libonnxruntime.dll.a"
                )
            endif()
        endif()

        # Create imported target
        add_library(OnnxRuntime::OnnxRuntime SHARED IMPORTED)
        set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
            IMPORTED_LOCATION "${OnnxRuntime_DLL}"
            IMPORTED_IMPLIB   "${CMAKE_BINARY_DIR}/libonnxruntime.dll.a"
            INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIRS}"
        )
        message(STATUS "ONNX Runtime found (MinGW): ${OnnxRuntime_DLL}")
    else()
        # MSVC or other: use the .lib file
        find_library(OnnxRuntime_LIBRARY
            NAMES onnxruntime onnxruntime_shared
            PATHS "${OnnxRuntime_LIB_DIR}"
            NO_DEFAULT_PATH)

        if(NOT OnnxRuntime_LIBRARY)
            message(FATAL_ERROR
                "ONNX Runtime library not found in ${OnnxRuntime_LIB_DIR}.")
        endif()

        add_library(OnnxRuntime::OnnxRuntime SHARED IMPORTED)
        set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
            IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIRS}"
        )
        message(STATUS "ONNX Runtime found: ${OnnxRuntime_LIBRARY}")
    endif()
else()
    find_package(OnnxRuntime QUIET)
    if(NOT OnnxRuntime_FOUND)
        message(FATAL_ERROR
            "ONNX Runtime not found. Set -DONNXRUNTIME_ROOT=<path>.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Core Library
# ---------------------------------------------------------------------------
set(DEEPCGH_SOURCES
    src/PreProcessor.cpp
    src/InferenceCore.cpp
    src/EngineAPI.cpp
)

set(DEEPCGH_HEADERS
    include/deepcgh/Types.h
    include/deepcgh/PreProcessor.h
    include/deepcgh/InferenceCore.h
    include/deepcgh/EngineAPI.h
)

add_library(deepcgh_engine STATIC ${DEEPCGH_SOURCES} ${DEEPCGH_HEADERS})

target_include_directories(deepcgh_engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(deepcgh_engine
    PUBLIC
        OnnxRuntime::OnnxRuntime
        FFTW3::FFTW3
)

# MinGW-specific: ensure the DLLs are copied next to the executable
if(MINGW AND ONNXRUNTIME_ROOT)
    add_custom_command(TARGET deepcgh_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OnnxRuntime_LIB_DIR}/onnxruntime.dll"
            "$<TARGET_FILE_DIR:deepcgh_engine>"
    )
endif()
if(MINGW AND FFTW3_ROOT)
    add_custom_command(TARGET deepcgh_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FFTW3_ROOT}/libfftw3f-3.dll"
            "$<TARGET_FILE_DIR:deepcgh_engine>"
    )
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(deepcgh_engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(deepcgh_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Python Bindings (PyBind11)
# ---------------------------------------------------------------------------
if(DEEPCGH_BUILD_PYTHON)
    find_package(pybind11 QUIET)

    if(NOT pybind11_FOUND)
        # Try fetching pybind11 via FetchContent as a fallback
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.12.0
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(_deepcgh_engine bindings/pybind_module.cpp)

    target_link_libraries(_deepcgh_engine
        PRIVATE
            deepcgh_engine
    )

    # Set the output name for the Python module
    set_target_properties(_deepcgh_engine PROPERTIES
        PREFIX ""
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )

    # Copy ORT DLL next to the Python module
    if(MINGW AND ONNXRUNTIME_ROOT)
        add_custom_command(TARGET _deepcgh_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OnnxRuntime_LIB_DIR}/onnxruntime.dll"
                "$<TARGET_FILE_DIR:_deepcgh_engine>"
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# Standalone Demo Application
# ---------------------------------------------------------------------------
if(DEEPCGH_BUILD_APPS)
    add_executable(deepcgh_demo apps/main.cpp)

    target_link_libraries(deepcgh_demo
        PRIVATE
            deepcgh_engine
    )

    target_include_directories(deepcgh_demo
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Copy DLLs next to the demo executable
    if(MINGW AND ONNXRUNTIME_ROOT)
        add_custom_command(TARGET deepcgh_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OnnxRuntime_LIB_DIR}/onnxruntime.dll"
                "$<TARGET_FILE_DIR:deepcgh_demo>"
        )
    endif()
    if(MINGW AND FFTW3_ROOT)
        add_custom_command(TARGET deepcgh_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW3_ROOT}/libfftw3f-3.dll"
                "$<TARGET_FILE_DIR:deepcgh_demo>"
        )
    endif()

    # Compiler warnings
    if(MSVC)
        target_compile_options(deepcgh_demo PRIVATE /W4 /permissive-)
    else()
        target_compile_options(deepcgh_demo PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(DEEPCGH_BUILD_TESTS)
    enable_testing()
endif()

# ---------------------------------------------------------------------------
# Install Rules
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS deepcgh_engine
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/deepcgh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
