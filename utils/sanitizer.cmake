function(viennacore_enable_sanitizer)
  include(CheckCompilerFlag)
  include(CheckLinkerFlag)

  set(VIENNACORE_SANITIZER_CANDIDATES
      -fno-omit-frame-pointer
      # GCC / Clang
      -fsanitize=undefined
      -fsanitize=address
      -fsanitize=thread
      # Clang
      -fsanitize=memory
      # GCC
      -fsanitize=leak
      # MSVC
      /fsanitize=address)

  foreach(flag IN LISTS VIENNACORE_SANITIZER_CANDIDATES)
    list(FIND VIENNACORE_SANITIZER_CANDIDATES "${flag}" INDEX)

    check_linker_flag(CXX ${flag} VIENNACORE_LINKER_FLAG_${INDEX})
    check_compiler_flag(CXX ${flag} VIENNACORE_COMPILER_FLAG_${INDEX})

    if(VIENNACORE_LINKER_FLAG_${INDEX})
      add_link_options(${flag})
    endif()

    if(VIENNACORE_COMPILER_FLAG_${INDEX})
      add_compile_options(${flag})
    endif()
  endforeach()

  message(STATUS "Enabled Sanitizer")
endfunction()
