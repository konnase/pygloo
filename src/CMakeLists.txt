# PYGLOO_SRCS is the list of source files that we need to build libpygloo.
set(PYGLOO_SRCS)

# PYGLOO_HDRS is the list of header files that we need to install.
set(PYGLOO_HDRS)

# Compiled sources in root directory
list(APPEND PYGLOO_SRCS
  # "${CMAKE_CURRENT_SOURCE_DIR}/allgather.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/allreduce.cc"
  # "${CMAKE_CURRENT_SOURCE_DIR}/barrier.cc"
  # "${CMAKE_CURRENT_SOURCE_DIR}/broadcast.cc"
  # "${CMAKE_CURRENT_SOURCE_DIR}/gather.cc"
  # "${CMAKE_CURRENT_SOURCE_DIR}/recv.cc"
  # "${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter.cc"
  # "${CMAKE_CURRENT_SOURCE_DIR}/reduce.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/rendezvous.cc"
  # "${CMAKE_CURRENT_SOURCE_DIR}/scatter.cc"
  # "${CMAKE_CURRENT_SOURCE_DIR}/send.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/transport.cc"
  )

set(PYGLOO_SRCS ${PYGLOO_SRCS} PARENT_SCOPE)

# Install if necessary.
# If the Gloo build is included from another project's build, it may
# want to statically link with Gloo and not install any artifacts.
if(PYGLOO_INSTALL)
  install(TARGETS pygloo EXPORT GlooTargets
      DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/pygloo)
  foreach(HEADER ${PYGLOO_HDRS})
    string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
    string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "pygloo" DIR ${DIR})
    install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${DIR})
  endforeach()
endif()
