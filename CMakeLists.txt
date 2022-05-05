# GenArt Solution

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(EXE_NAME GenArt)

# CUDA
project(${EXE_NAME} LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# GLEW - setenv GLEW_HOME C:/Users/davemc/source/repos/Goodies/glew-2.2.0
set(CMAKE_PREFIX_PATH $ENV{GLEW_HOME})
set(CMAKE_LIBRARY_PATH $ENV{GLEW_HOME}/lib/Release/x64)
set(GLEW_USE_STATIC_LIBS TRUE)

find_package(GLEW REQUIRED)

add_definitions(-DGLEW_STATIC)
include_directories(${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_STATIC_LIBRARIES})

# FreeGLUT - setenv GLUT_HOME C:/Users/davemc/source/repos/Goodies/freeglut-3.2.2
set(GLUT_INCLUDE_DIR $ENV{GLUT_HOME}/include)
set(GLUT_LIBRARIES "optimized;$ENV{GLUT_HOME}/lib/Release/freeglut_static.lib;debug;$ENV{GLUT_HOME}/lib/Debug/freeglut_staticd.lib")

add_definitions(-DFREEGLUT_STATIC)
include_directories(${GLUT_INCLUDE_DIR})
link_libraries(${GLUT_LIBRARIES})

set(EXPR_SOURCES
    src/BinaryExprSubclasses.cpp
    src/BinaryExprSubclasses.h
    src/Expr.cpp
    src/Expr.h
    src/ExprHelpers.cpp
    src/ExprHelpers.h
    src/ExprImplementations.h
    src/ExprTools.cpp
    src/ExprTools.h
    src/Interval.cpp
    src/Interval.h
    src/IntervalImplementations.cpp
    src/IntervalImplementations.h
    src/NonaryExprSubclasses.cpp
    src/NonaryExprSubclasses.h
    src/TestExpressions.cpp
    src/UnaryExprSubclasses.cpp
    src/UnaryExprSubclasses.h
)

set(INDIV_SOURCES
    src/AutoScorer.h
    src/AutoScorer_kernels.cu
    src/Evaluator.h
    src/Individual.cpp
    src/Individual.h
    src/MathIndividual.cpp
    src/MathIndividual.h
    src/MathStyle.cpp
    src/MathStyle.h
    src/MathStyleCUDARender.cpp
    src/MathStyleCUDARender.h
    src/MathStyleDefines.h
    src/MathStyleRender_kernel.cu
    src/MathStyleRender_kernel.h
    src/Style.cpp
    src/Style.h
)

set(MAIN_SOURCES
    src/CUDAHelpers.cpp
    src/CUDAHelpers.h
    src/CUDARender.h
    src/Counters.cpp
    src/Counters.h
    src/Evolver.cpp
    src/Evolver.h
    src/Interop.cpp
    src/Interop.h
    src/Main.cpp
    src/MathHelpers.h
    src/Population.cpp
    src/Population.h
    src/RenderHelpers.h
    src/RenderManager.cpp
    src/RenderManager.h
    src/Test.cpp
    src/Test.h
    src/UI.cpp
    src/UI.h
    src/Utils.h
    )

source_group("Expr" FILES ${EXPR_SOURCES})
source_group("Indiv" FILES ${INDIV_SOURCES})
source_group("Main" FILES ${MAIN_SOURCES})

add_subdirectory(${PROJECT_ROOT_DIR}/../DMcTools ${CMAKE_CURRENT_BINARY_DIR}/DMcTools)

add_executable(${EXE_NAME} ${EXPR_SOURCES} ${INDIV_SOURCES} ${MAIN_SOURCES})

set_target_properties(${EXE_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${PROJECT_ROOT_DIR}
    CUDA_ARCHITECTURES 86
)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${EXE_NAME})

target_link_libraries(${EXE_NAME} PRIVATE DMcTools)
