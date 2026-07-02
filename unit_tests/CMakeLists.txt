include(add_alquimia_unit_test)
add_alquimia_unit_test(test_alquimia_c_utils_exe test_alquimia_c_utils.c)

if (XSDK_WITH_ONNX)
  add_alquimia_unit_test(test_alquimia_onnx_exe test_alquimia_onnx.c)
endif()
