# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_METAWARENN_CODEGEN STREQUAL "ON")
  add_definitions(-DUSE_JSON_RUNTIME=1)
  file(GLOB METAWARENN_RELAY_CONTRIB_SRC src/relay/backend/contrib/metawarenn/*.cc)
  list(APPEND COMPILER_SRCS ${METAWARENN_RELAY_CONTRIB_SRC})
  list(APPEND COMPILER_SRCS ${JSON_RELAY_CONTRIB_SRC})

  file(GLOB METAWARENN_LIB_SRC src/runtime/contrib/metawarenn/metawarenn_lib/*.cc
    src/runtime/contrib/metawarenn/metawarenn_lib/op/*.cc
    src/runtime/contrib/metawarenn/metawarenn_lib/optimizer/*.cc
    src/runtime/contrib/metawarenn/metawarenn_lib/kernels/*.cc
    src/runtime/contrib/metawarenn/metawarenn_lib/kernels/helpers/*.cc
    src/runtime/contrib/metawarenn/metawarenn_lib/executable_network/*.cc
    src/runtime/contrib/metawarenn/metawarenn_lib/mwnnconvert/*.cc
    src/runtime/contrib/metawarenn/metawarenn_lib/mwnnconvert/mwnn_protobuf/cpp_wrapper/*.cc
    src/runtime/contrib/metawarenn/metawarenn_lib/mwnnconvert/onnx_protobuf/*.cc
    src/runtime/contrib/metawarenn/metawarenn_lib/inference_engine/*.cc)

  list(APPEND RUNTIME_SRCS ${METAWARENN_LIB_SRC})
  file(GLOB METAWARENN_CONTRIB_SRC src/runtime/contrib/metawarenn/metawarenn_json_runtime.cc)
  list(APPEND RUNTIME_SRCS ${METAWARENN_CONTRIB_SRC})

  find_library(MWNN_PB_LIB_DIR NAMES protobuf HINTS metawarenn_inference)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${MWNN_PB_LIB_DIR})
endif()

