# GenArt Solution

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# TODO: Debug
include(CMakePrintHelpers)
include(CheckLanguage)

set(EXE_NAME GenArt)

# CUDA
project(${EXE_NAME} LANGUAGES CXX CUDA)
enable_language(CUDA)
# set_property(TARGET ${EXE_NAME} PROPERTY CUDA_ARCHITECTURES 86)

# TODO: Debug
check_language(CUDA)
cmake_print_variables(CMAKE_CUDA_COMPILER)

set(CMAKE_CXX_STANDARD 11 CACHE STRING "" FORCE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# warning level, all warnings as errors, build in parallel, disable security warnings, latest language support
# add_compile_options(/W3 /WX /MP /sdl- /std:c++latest)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W3>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/WX>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/MP>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/sdl->)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/std:c++latest>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-std>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:c++17>)

set(__ARCH "x64")

# GLEW
set(GLEW_USE_STATIC_LIBS TRUE)
set(CMAKE_PREFIX_PATH "../Goodies/glew-2.2.0")
set(CMAKE_LIBRARY_PATH "../Goodies/glew-2.2.0/lib/Release/x64")

find_package(GLEW REQUIRED)
# TODO: Debug
cmake_print_variables(GLEW_INCLUDE_DIRS)
cmake_print_variables(GLEW_STATIC_LIBRARIES)

add_definitions(-DGLEW_STATIC)
include_directories(${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_STATIC_LIBRARIES})

# FreeGLUT
set(GLUT_INCLUDE_DIR "../Goodies/freeglut-3.2.2/include")
set(GLUT_LIBRARIES "optimized;C:/Users/davemc/source/repos/Goodies/freeglut-3.2.2/lib/Release/freeglut_static.lib;debug;C:/Users/davemc/source/repos/Goodies/freeglut-3.2.2/lib/Debug/freeglut_staticd.lib")

add_definitions(-DFREEGLUT_STATIC)
include_directories(${GLUT_INCLUDE_DIR})
link_libraries(${GLUT_LIBRARIES})

set(SOURCES
    src/AutoScorer.h
    src/AutoScorer_kernels.cu
    src/BinaryExprSubclasses.cpp
    src/BinaryExprSubclasses.h
    src/CUDAHelpers.cpp
    src/CUDAHelpers.h
    src/CUDARender.h
    src/Counters.cpp
    src/Counters.h
    src/Evaluator.h
    src/Evolver.cpp
    src/Evolver.h
    src/Expr.cpp
    src/Expr.h
    src/ExprHelpers.cpp
    src/ExprHelpers.h
    src/ExprImplementations.h
    src/ExprTools.cpp
    src/ExprTools.h
    src/Individual.cpp
    src/Individual.h
    src/Interop.cpp
    src/Interop.h
    src/Interval.cpp
    src/Interval.h
    src/IntervalImplementations.cpp
    src/IntervalImplementations.h
    src/Main.cpp
    src/MathHelpers.h
    src/MathIndividual.cpp
    src/MathIndividual.h
    src/MathStyle.cpp
    src/MathStyle.h
    src/MathStyleCUDARender.cpp
    src/MathStyleCUDARender.h
    src/MathStyleDefines.h
    src/MathStyleRender_kernel.cu
    src/MathStyleRender_kernel.h
    src/NonaryExprSubclasses.cpp
    src/NonaryExprSubclasses.h
    src/Population.cpp
    src/Population.h
    src/RenderHelpers.h
    src/RenderManager.cpp
    src/RenderManager.h
    src/Style.cpp
    src/Style.h
    src/Test.cpp
    src/Test.h
    src/TestExpressions.cpp
    src/UI.cpp
    src/UI.h
    src/UnaryExprSubclasses.cpp
    src/UnaryExprSubclasses.h
    src/Utils.h
    )

source_group("src"  FILES ${SOURCES})

add_subdirectory(${PROJECT_ROOT_DIR}/../DMcTools ${CMAKE_CURRENT_BINARY_DIR}/DMcTools)

add_executable(${EXE_NAME} ${SOURCES})

set_target_properties(${EXE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${PROJECT_ROOT_DIR} )
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${EXE_NAME})

target_link_libraries(${EXE_NAME} PRIVATE DMcTools)
