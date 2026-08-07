add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE ${CMAKE_SOURCE_DIR}/third_party/doctest)

add_executable(ml_core_tests
    test_main.cpp
    test_matrix.cpp
    test_linear_regression.cpp
)

target_link_libraries(ml_core_tests
    PRIVATE
        ml::core
        doctest
)

include(CTest)
add_test(NAME ml_core_tests COMMAND ml_core_tests)
