find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

#get numpy include path
execute_process(
  COMMAND
  ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_PATH
  RESULT_VARIABLE NUMPY_ERR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
if(NUMPY_ERR)
  message(SEND_ERROR "WARNING: NumPy header not found.")
endif(NUMPY_ERR)

add_cython_target(encodec_cy CXX PY3)

add_library(encodec_cy MODULE ${encodec_cy})

# Same linking for both system-wide or local installation
target_link_libraries(encodec_cy ${ENCODEC_LIB})

# For system-linked installation, this will do nothing
target_include_directories(encodec_cy PUBLIC ${NUMPY_INCLUDE_PATH}) # ${ENCODEC_LIB_HEADERS})

python_extension_module(encodec_cy)
install(TARGETS encodec_cy LIBRARY DESTINATION encodecpp)