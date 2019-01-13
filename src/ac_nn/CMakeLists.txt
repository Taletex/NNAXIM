add_executable(ac_nn ac_nn.cpp ${tiny_dnn_headers})
target_link_libraries(ac_nn
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

cotire(ac_nn)