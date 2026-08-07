add_executable(ml_core_matmul_benchmark
    matmul_benchmark.cpp
)

target_link_libraries(ml_core_matmul_benchmark
    PRIVATE
        ml::core
)
