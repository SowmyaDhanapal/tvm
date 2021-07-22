/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/metawarenn/metawarenn_json_runtime.cc
 * \brief A simple JSON runtime for MetaWareNN.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/executable_network/metawarenn_executable_graph.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class MetaWareNNJSONRuntime : public JSONRuntimeBase {

 public:
  MetaWareNNJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "metawarenn_json"; }

  void Init(const Array<NDArray>& consts) override {
    std::cout << "\n In MetaWareNN INIT!!!";
    std::cout << "\n Total Constants : " << consts.size();
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    // Setup constants entries for weights.
    SetupConstants(consts);
    BuildMetaWareNNGraph();
  }

  void Run() override {
    std::cout << "\n In MetaWareNN RUNN!!!";
  }

 private:
  std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph_;
  std::shared_ptr<::metawarenn::MWNNExecutableGraph> mwnn_exe_graph_;
  // Build up the engine based on the input graph.
  void BuildMetaWareNNGraph() {
    std::cout << "\n In BuildMetaWareNNGraph " << symbol_name_;
    mwnn_graph_ = std::make_shared<::metawarenn::MWNNGraph>(nodes_, symbol_name_);
    // Add inputs and constants.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      const auto& node = nodes_[nid];
      std::string name = "node_" + std::to_string(nid);
      if (node.GetOpType() == "input") {
        mwnn_graph_->set_graph_inputs(name, node);
      }
      else if (node.GetOpType() == "const") {
        uint32_t eid = EntryID(nid, 0);
        std::string name = "node_" + std::to_string(nid);
        mwnn_graph_->set_graph_initializers(name, data_entry_[eid]);
      }
    }
    // Add outputs.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const auto& node = outputs_[i];
      std::cout << "\n Node op : " << node.id_;
      mwnn_graph_->set_graph_outputs(node);
    }
  }

};

runtime::Module MetaWareNNJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  std::cout << "\n In MetaWareNNJSONRuntimeCreate !!!";
  std::cout << "\n symbol_name : " << symbol_name;
  auto n = make_object<MetaWareNNJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.MetaWareNNJSONRuntimeCreate").set_body_typed(MetaWareNNJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_metawarenn_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<MetaWareNNJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
