# This function adds a batch chemistry benchmark test for Alquimia.
# Additional arguments name optional files, relative to the working directory.
function(add_alquimia_batch_chem_benchmark benchmark input_file)
  set(exe ${PROJECT_BINARY_DIR}/drivers/batch_chem)
  if (ARGN)
    add_test(batch_chem_${benchmark} ${CMAKE_COMMAND}
      "-Dexe=${exe}" "-Dinput_file=${input_file}" "-Doptional_files=${ARGN}"
      -P ${PROJECT_SOURCE_DIR}/cmake/Modules/run_optional_batch_chem_benchmark.cmake)
    set_tests_properties(batch_chem_${benchmark} PROPERTIES
      SKIP_REGULAR_EXPRESSION "SKIPPED: optional benchmark file unavailable:")
  else()
    add_test(batch_chem_${benchmark} ${exe} ${input_file})
  endif()
  set_tests_properties(batch_chem_${benchmark} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()
