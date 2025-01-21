# Find the gloo libraries
#
# The following variables are optionally searched for defaults
#  GLOO_ROOT_DIR: Base directory where all gloo components are found
#  GLOO_INCLUDE_DIR: Directory where gloo headers are found
#  GLOO_LIB_DIR: Directory where gloo libraries are found

# The following are set after configuration is done:
#  GLOO_FOUND
#  GLOO_INCLUDE_DIRS
#  GLOO_LIBRARIES

find_path(GLOO_INCLUDE_DIRS
  NAMES gloo/config.h
  HINTS
  ${GLOO_INCLUDE_DIR}
  ${GLOO_ROOT_DIR}
  ${GLOO_ROOT_DIR}/include)

find_library(GLOO_LIBRARIES
  NAMES gloo
  HINTS
  ${GLOO_LIB_DIR}
  ${GLOO_ROOT_DIR}
  ${GLOO_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
# message(STATUS "after find gloo: " ${GLOO_ROOT_DIR} ": " ${GLOO_INCLUDE_DIR} ": " ${GLOO_LIBRARIES})
find_package_handle_standard_args(gloo DEFAULT_MSG GLOO_INCLUDE_DIRS GLOO_LIBRARIES)
mark_as_advanced(GLOO_INCLUDE_DIR GLOO_LIBRARIES)
