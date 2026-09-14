# Check at test time so optional files can be supplied without reconfiguring.
foreach(optional_file ${optional_files})
  if (NOT EXISTS "${optional_file}")
    message("SKIPPED: optional benchmark file unavailable: ${optional_file}")
    return()
  endif()
endforeach()

# Preserve failures from models that are present, including invalid models.
execute_process(COMMAND "${exe}" "${input_file}" RESULT_VARIABLE result)
if (NOT "${result}" STREQUAL "0")
  message(FATAL_ERROR "Batch chemistry benchmark failed: ${result}")
endif()
