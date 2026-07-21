include(add_alquimia_unit_test)
add_alquimia_unit_test(test_alquimia_c_utils_exe test_alquimia_c_utils.c)

if (XSDK_WITH_ONNX)
  add_alquimia_unit_test(test_alquimia_onnx_exe test_alquimia_onnx.c)
  # Tag test_alquimia_onnx_exe for easy filtering with `ctest -L`
  # Set labels for test_alquimia_onnx_exe 
  # Allows easy categorization when filtering tests via `ctest -L <label>`
  set_tests_properties(test_alquimia_onnx_exe PROPERTIES
    LABELS "onnx;unit;condition;integration")

  add_test(NAME test_alquimia_onnx_manifest
    COMMAND test_alquimia_onnx_exe manifest)
  set_tests_properties(test_alquimia_onnx_manifest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    LABELS "onnx;unit;manifest;mapping")

  add_alquimia_unit_test(test_alquimia_onnx_failures_exe
    test_alquimia_onnx_failures.c)
  set_tests_properties(test_alquimia_onnx_failures_exe PROPERTIES
    LABELS "onnx;unit;manifest;condition;resource")

  add_executable(test_alquimia_onnx_routing_exe
    test_alquimia_onnx_routing.c)
  target_link_libraries(test_alquimia_onnx_routing_exe
    alquimia;${ALQUIMIA_TPLS})
  set_target_properties(test_alquimia_onnx_routing_exe PROPERTIES
    COMPILE_FLAGS
      "-DCMAKE_CURRENT_SOURCE_DIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")

  add_test(NAME test_alquimia_onnx_routing
    COMMAND test_alquimia_onnx_routing_exe routing)
  set_tests_properties(test_alquimia_onnx_routing PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    LABELS "onnx;unit;routing")

  add_test(NAME test_alquimia_onnx_model_family
    COMMAND test_alquimia_onnx_routing_exe model-family)
  set_tests_properties(test_alquimia_onnx_model_family PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    LABELS "onnx;integration;model-family")
endif()
