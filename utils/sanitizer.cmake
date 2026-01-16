function(viennacore_enable_sanitizer)
  include(CheckLinkerFlag)

  add_compile_options(-fno-omit-frame-pointer)

  if(MSVC)
    set(flags "/fsanitize=address" "/fsanitize=undefined")
  else()
    set(flags "-fsanitize=address" "-fsanitize=undefined")
  endif()

  foreach(flag IN LISTS flags)
    string(MAKE_C_IDENTIFIER "${flag}" flag_id)
    check_linker_flag(CXX "${flag}" "VIENNACORE_LINKER_${flag_id}")

    if(NOT VIENNACORE_LINKER_${flag_id})
      message(WARNING "Sanitizer flag '${flag}' linker check failed.")
    else()
      add_compile_options("${flag}")
      add_link_options("${flag}")
    endif()
  endforeach()

  message(STATUS "Enabled Sanitizer")
endfunction()
