cmake_minimum_required(VERSION 3.14)
project(AutoMLDotaBot)

include(ExternalProject)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(THIRDPARTY_PATH ${CMAKE_SOURCE_DIR}/thirdparty)

#from paddlepaddle
FUNCTION(build_protobuf TARGET_NAME BUILD_FOR_HOST)
    set(THIRD_PARTY_PATH ${CMAKE_SOURCE_DIR}/thirdparty)
    STRING(REPLACE "extern_" "" TARGET_DIR_NAME "${TARGET_NAME}")
    SET(PROTOBUF_SOURCES_DIR ${THIRD_PARTY_PATH}/${TARGET_DIR_NAME})
    SET(PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${TARGET_DIR_NAME})

    SET(${TARGET_NAME}_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" PARENT_SCOPE)
    SET(PROTOBUF_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" PARENT_SCOPE)
    SET(${TARGET_NAME}_LITE_LIBRARY
            "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite${CMAKE_STATIC_LIBRARY_SUFFIX}"
            PARENT_SCOPE)
    SET(${TARGET_NAME}_LIBRARY
            "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX}"
            PARENT_SCOPE)
    SET(${TARGET_NAME}_PROTOC_LIBRARY
            "${PROTOBUF_INSTALL_DIR}/lib/libprotoc${CMAKE_STATIC_LIBRARY_SUFFIX}"
            PARENT_SCOPE)
    SET(${TARGET_NAME}_PROTOC_EXECUTABLE
            "${PROTOBUF_INSTALL_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}"
            PARENT_SCOPE)

    SET(OPTIONAL_CACHE_ARGS "")
    SET(OPTIONAL_ARGS "")
    IF(BUILD_FOR_HOST)
        SET(OPTIONAL_ARGS "-Dprotobuf_WITH_ZLIB=OFF")
    ELSE()
        SET(OPTIONAL_ARGS
                "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
                "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
                "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}"
                "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
                "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
                "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
                "-Dprotobuf_WITH_ZLIB=ON"
                "-DZLIB_ROOT:FILEPATH=${ZLIB_ROOT}"
                ${EXTERNAL_OPTIONAL_ARGS})
        SET(OPTIONAL_CACHE_ARGS "-DZLIB_ROOT:STRING=${ZLIB_ROOT}")
    ENDIF()
    IF(WIN32)
        SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} "-DCMAKE_GENERATOR_PLATFORM=x64")
    ENDIF()

    ExternalProject_Add(
            ${TARGET_NAME}
            PREFIX          ${PROTOBUF_SOURCES_DIR}
            DOWNLOAD_COMMAND  ""
            UPDATE_COMMAND  ""
            CONFIGURE_COMMAND
            ${CMAKE_COMMAND} ${PROTOBUF_SOURCES_DIR}/cmake
            ${OPTIONAL_ARGS}
            -Dprotobuf_BUILD_TESTS=OFF
            -DCMAKE_SKIP_RPATH=ON
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR}
            -DCMAKE_INSTALL_LIBDIR=lib
            -DBUILD_SHARED_LIBS=OFF
            CMAKE_CACHE_ARGS
            -DCMAKE_INSTALL_PREFIX:PATH=${PROTOBUF_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
            -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
            ${OPTIONAL_CACHE_ARGS}
    )
ENDFUNCTION()

build_protobuf(extern_protobuf TRUE)


SET(PROTOBUF_INCLUDE_DIR ${extern_protobuf_INCLUDE_DIR}
        CACHE PATH "protobuf include directory." FORCE)
SET(PROTOBUF_LITE_LIBRARY ${extern_protobuf_LITE_LIBRARY}
        CACHE FILEPATH "protobuf lite library." FORCE)
SET(PROTOBUF_LIBRARY ${extern_protobuf_LIBRARY}
        CACHE FILEPATH "protobuf library." FORCE)
SET(PROTOBUF_PROTOC_LIBRARY ${extern_protobuf_PROTOC_LIBRARY}
        CACHE FILEPATH "protoc library." FORCE)

set(PROTOBUF_PROTOC_EXECUTABLE ${extern_protobuf_PROTOC_EXECUTABLE}
        CACHE FILEPATH "protoc compiler." FORCE)


set(DOTASERVICE_PROTO_PATH ${CMAKE_SOURCE_DIR}/thirdparty/dotaservice)

set(JSON_BuildTests OFF CACHE INTERNAL "")

add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/json)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${PROTOBUF_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/protobuf/)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/json/include)

set(THIRDPARTY_INSTALL ${THIRDPARTY_PATH}/install/)

ExternalProject_Add(c-ares
        PREFIX c-ares
        SOURCE_DIR "${THIRDPARTY_PATH}/c-ares"
        DOWNLOAD_COMMAND  ""
        CMAKE_CACHE_ARGS
        -DCARES_SHARED:BOOL=OFF
        -DCARES_STATIC:BOOL=ON
        -DCARES_STATIC_PIC:BOOL=ON
        -DCMAKE_INSTALL_PREFIX:PATH=${THIRDPARTY_INSTALL}/c-ares
        )

# Builds zlib project from the git submodule.
ExternalProject_Add(zlib
        PREFIX zlib
        SOURCE_DIR "${THIRDPARTY_PATH}/zlib"
        DOWNLOAD_COMMAND  ""
        CMAKE_CACHE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${THIRDPARTY_INSTALL}/zlib
        )

# the location where protobuf-config.cmake will be installed varies by platform
set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${THIRDPARTY_INSTALL}/protobuf/lib/cmake/protobuf")

set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")

# Builds gRPC based on locally checked-out sources and set arguments so that all the dependencies
# are correctly located.

set(grpc_name extern_grpc)

ExternalProject_Add(${grpc_name}
        PREFIX grpc
        SOURCE_DIR "${THIRDPARTY_PATH}/grpc"
        DOWNLOAD_COMMAND  ""
        CMAKE_CACHE_ARGS
        -DgRPC_INSTALL:BOOL=ON
        -DgRPC_BUILD_TESTS:BOOL=OFF
        -DgRPC_PROTOBUF_PROVIDER:STRING=package
        -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -DgRPC_ZLIB_PROVIDER:STRING=package
        -DZLIB_ROOT:STRING=${THIRDPARTY_INSTALL}/zlib
        -DgRPC_CARES_PROVIDER:STRING=package
        -Dc-ares_DIR:PATH=${THIRDPARTY_INSTALL}/c-ares/lib/cmake/c-ares
        -DgRPC_SSL_PROVIDER:STRING=module
        -DBORINGSSL_ROOT_DIR:PATH=${THIRDPARTY_PATH}/boringssl
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        -DCMAKE_INSTALL_PREFIX:PATH=${THIRDPARTY_INSTALL}/grpc
        DEPENDS c-ares extern_protobuf zlib
        )

include_directories(${THIRDPARTY_INSTALL}/grpc/include)

set(PROTOS
        dota_gcmessages_common_bot_script.proto
        dota_shared_enums.proto
        DotaService.proto)

set(_GRPC_CPP_PLUGIN_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/grpc/src/${grpc_name}-build/grpc_cpp_plugin)
set(_GRPC_GRPCPP_LIBRARY
        grpc++
        grpc++_cronet
        grpc++_error_details
        grpc++_reflection
        grpc++_unsecure
        grpcpp_channelz
        grpc
        gpr
        address_sorting)

foreach(FIL ${PROTOS})
    STRING(REPLACE ".proto" ".pb.cc" SRC_NAME "${FIL}")
    STRING(REPLACE ".proto" ".pb.h" HDR_NAME "${FIL}")
    STRING(REPLACE ".proto" ".grpc.pb.cc" GRPC_SRC_NAME "${FIL}")
    STRING(REPLACE ".proto" ".grpc.pb.h" GRPC_HDR_NAME "${FIL}")

    set(OUTPUT_SRC ${CMAKE_SOURCE_DIR}/src/protobuf/dotaservice/protos/${SRC_NAME})
    set(OUTPUT_HDR ${CMAKE_SOURCE_DIR}/src/protobuf/dotaservice/protos/${HDR_NAME})
    set(OUTPUT_GRPC_SRC ${CMAKE_SOURCE_DIR}/src/protobuf/dotaservice/protos/${GRPC_SRC_NAME})
    set(OUTPUT_GRPC_HDR ${CMAKE_SOURCE_DIR}/src/protobuf/dotaservice/protos/${GRPC_HDR_NAME})

    list(APPEND DOTA_PROTO_SRCS ${OUTPUT_SRC})
    list(APPEND DOTA_PROTO_HDRS ${OUTPUT_HDR})
    list(APPEND DOTA_PROTO_SRCS ${OUTPUT_GRPC_SRC})
    list(APPEND DOTA_PROTO_HDRS ${OUTPUT_GRPC_HDR})

    add_custom_command(
            OUTPUT ${OUTPUT_SRC} ${OUTPUT_HDR}
            COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
            --plugin="protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}"
            --cpp_out ${CMAKE_SOURCE_DIR}/src/protobuf
            --grpc_out ${CMAKE_SOURCE_DIR}/src/protobuf
            -I ${DOTASERVICE_PROTO_PATH}

            ${CMAKE_SOURCE_DIR}/thirdparty/dotaservice/dotaservice/protos/${FIL}

            DEPENDS ${DOTASERVICE_PROTO} ${PROTOBUF_PROTOC_EXECUTABLE} ${grpc_name}
            COMMENT "${PROTOBUF_PROTOC_EXECUTABLE}
            --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}
            --cpp_out ${CMAKE_SOURCE_DIR}/src/protobuf
            --grpc_out ${CMAKE_SOURCE_DIR}/src/protobuf
            -I ${DOTASERVICE_PROTO_PATH}

            ${CMAKE_SOURCE_DIR}/thirdparty/dotaservice/dotaservice/protos/${FIL}"
            )

endforeach()

set_source_files_properties(
        ${DOTA_PROTO_SRCS}
        ${DOTA_PROTO_HDRS}
        PROPERTIES GENERATED TRUE
)


add_executable(AutoMLDotaBot main.cpp include/dotaclient/dotaservice.h include/dotaclient/env.h include/dotaclient/util.h src/dotaclient/env.cpp include/nn/nn.h include/nn/util.h include/nn/layer.h src/nn/layer.cpp include/nn/torch_layer.h)
target_link_libraries(AutoMLDotaBot nlohmann_json::nlohmann_json)

message(grpc::grpc++_unsecure)

link_directories(${CMAKE_CURRENT_BINARY_DIR}/grpc/src/${grpc_name}-build/)
link_directories(${THIRDPARTY_INSTALL}/protobuf/lib)
link_directories(${THIRDPARTY_INSTALL}/zlib/lib)
link_directories(${THIRDPARTY_INSTALL}/c-ares/lib)

add_executable(grpc_test "grpc_test.cpp" ${DOTA_PROTO_SRCS} ${DOTA_PROTO_HDRS})
target_link_libraries(grpc_test
        ${_GRPC_GRPCPP_LIBRARY}
        protobuf
        pthread
        z
        cares)
