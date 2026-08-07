add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE ${CMAKE_SOURCE_DIR}/third_party/doctest)

add_executable(ml_core_tests
    test_main.cpp
    test_matrix.cpp
    test_linear_regression.cpp
    test_kmeans.cpp
    test_validation.cpp
)

target_link_libraries(ml_core_tests
    PRIVATE
        ml::core
        doctest
)

# Points test_validation.cpp at the checked-in scikit-learn reference
# fixtures regardless of where the build directory lives.
target_compile_definitions(ml_core_tests
    PRIVATE
        ML_CORE_VALIDATION_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/validation/fixtures"
)

include(CTest)
add_test(NAME ml_core_tests COMMAND ml_core_tests)
