add_library(ml_core STATIC
    ml_core.cpp
    Matrix.cpp
    LinearRegression.cpp
)

add_library(ml::core ALIAS ml_core)

target_include_directories(ml_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(ml_core PUBLIC cxx_std_17)
