include(add_alquimia_unit_test)
add_alquimia_unit_test(test_alquimia_c_utils_exe test_alquimia_c_utils.c)

if (XSDK_WITH_ONNX)
  # test_alquimia_onnx.c
  add_alquimia_unit_test(test_alquimia_onnx_mapping_exe test_alquimia_onnx_mapping.c)
  # Tag test_alquimia_onnx_mapping_exe for easy filtering with `ctest -L`
  # Set labels for test_alquimia_onnx_mapping_exe 
  # Allows easy categorization when filtering tests via `ctest -L <label>`
  set_tests_properties(test_alquimia_onnx_mapping_exe PROPERTIES
    LABELS "onnx;unit;condition;integration")

  # Register a test case in the ctest
  add_test(NAME test_alquimia_onnx_config
    COMMAND test_alquimia_onnx_mapping_exe config)

  # Allows easy categorization when filtering tests via `ctest -L <label>`
  # Work directory has to be: ~/alquimia-ryan/build/alquimia-build/unit_tests
  set_tests_properties(test_alquimia_onnx_config PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    LABELS "onnx;unit;config;mapping")

  # test_alquimia_onnx_failures.c
  add_alquimia_unit_test(test_alquimia_onnx_failures_exe
    test_alquimia_onnx_failures.c)

  # Allows easy categorization when filtering tests via `ctest -L <label>`
  # Work directory has to be: ~/alquimia-ryan/build/alquimia-build/unit_tests
  set_tests_properties(test_alquimia_onnx_failures_exe PROPERTIES
    LABELS "onnx;unit;config;condition;resource")

  # ---------------------------------------------------------------------------
  # test_alquimia_onnx_inference_routing.c
  # Note: Cannot use add_alquimia_unit_test() here because this executable 
  # provides multiple sub-test suites via CLI arguments ("routing" and "model-family"), 
  # whereas add_alquimia_unit_test() rigidly registers a single no-argument test.
  # ---------------------------------------------------------------------------
  add_executable(test_alquimia_onnx_inference_routing_exe
    test_alquimia_onnx_inference_routing.c)
  target_link_libraries(test_alquimia_onnx_inference_routing_exe
    alquimia;${ALQUIMIA_TPLS})
  set_target_properties(test_alquimia_onnx_inference_routing_exe PROPERTIES
    COMPILE_FLAGS
      "-DCMAKE_CURRENT_SOURCE_DIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")

  add_test(NAME test_alquimia_onnx_routing
    COMMAND test_alquimia_onnx_inference_routing_exe routing)
  # Allows easy categorization when filtering tests via `ctest -L <label>`
  set_tests_properties(test_alquimia_onnx_routing PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    LABELS "onnx;unit;routing")

  add_test(NAME test_alquimia_onnx_model_family
    COMMAND test_alquimia_onnx_inference_routing_exe model-family)

  # Allows easy categorization when filtering tests via `ctest -L <label>`
  # Work directory has to be: ~/alquimia-ryan/build/alquimia-build/unit_tests
  set_tests_properties(test_alquimia_onnx_model_family PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    LABELS "onnx;integration;model-family")
endif()
