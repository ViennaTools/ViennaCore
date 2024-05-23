macro(viennacore_setup_bat TARGET PATH)
  message(STATUS "Generating Bat file for ${TARGET}")

  file(
    GENERATE
    OUTPUT "$<TARGET_FILE_DIR:${TARGET}>/${TARGET}.bat"
    CONTENT "set \"PATH=${PATH};%PATH%\"\n%~dp0${TARGET}.exe %*")
endmacro()
