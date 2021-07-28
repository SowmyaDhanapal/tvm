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
#include "metawarenn_lib/metawarenn_utils.h"
#include "metawarenn_lib/optimizer/pass_manager.h"
#include "metawarenn_lib/executable_network/metawarenn_executable_graph.h"
#include "metawarenn_lib/mwnn_inference_api/mwnn_inference_api.h"
#define CHW_TO_HWC 0

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
    //Generated MetaWareNN Graph
    BuildMetaWareNNGraph();
    //Optimize MetaWareNN Graph
    ApplyPasses();
    //Generate Executable Network
    mwnn_exe_graph_ = std::make_shared<metawarenn::MWNNExecutableGraph>(*mwnn_graph_);
  }

  void Run() override {
    std::cout << "\n In MetaWareNN RUNN!!!";
    std::unordered_map<std::string, float*> graph_inputs;
    std::unordered_map<std::string, float*> graph_outputs;

    for (auto g_ip : mwnn_graph_->get_graph_inputs()) {
      std::cout << "\n Graph Input  : " << g_ip.get_name();
      for (size_t i = 0; i < input_nodes_.size(); ++i) {
        auto nid = input_nodes_[i];
        if (nodes_[nid].GetOpType() == "input") {
          auto eid = EntryID(input_nodes_[i], 0);
          float *input_buf = (float*)(data_entry_[eid]->data);
          graph_inputs[g_ip.get_name()] = input_buf;
        }
      }
    }
    for (auto g_op : mwnn_graph_->get_graph_outputs()) {
      std::cout << "\n Graph Output  : " << g_op.get_name();
      for (size_t i = 0; i < outputs_.size(); ++i) {
        auto eid = EntryID(outputs_[i]);
        size_t buffer_size = GetDataSize(*data_entry_[eid]);
        std::cout << "\n Output eid : " << eid << " buffer_size : " << buffer_size;
        float *output_buf = (float*)(data_entry_[eid]->data);
        graph_outputs[g_op.get_name()] = output_buf;
      }
    }
    // **************************************** Calls to invoke the MetaWareNN Inference API ************************************

    /*::metawarenn::MWNNInferenceApi mwapi;

    std::string ip_name = mwnn_graph_->get_graph_ip_name();
    auto ip_shape = mwnn_graph_->get_graph_ip_tensor()[0].get_dims();
    mwapi.prepareInput(graph_inputs[ip_name], ip_shape);

    std::string op_name = mwnn_graph_->get_graph_op_name();
    auto op_shape = mwnn_graph_->get_graph_op_tensor()[0].get_dims();
    mwapi.prepareOutput(op_shape);

    mwapi.prepareGraph(mwnn_graph_->get_name());

    mwapi.runGraph();

    mwapi.getOutput(graph_outputs[op_name], op_shape);*/

    // ******************************************* Call to invoke the local run function *****************************************
    ::metawarenn::convert_to_mwnn_format(*mwnn_graph_, graph_inputs, graph_outputs, CHW_TO_HWC);
    //mwnn_exe_graph_->runGraph();
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
  }

  void ApplyPasses() {
    ::metawarenn::optimizer::PassManager manager;
    auto node_list = mwnn_graph_->get_graph_nodes();
    if(CHW_TO_HWC)
    {
      for (auto g_t : mwnn_graph_->get_graph_initializers()) {
        if(g_t.get_dims().size() == 4) {
          std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";
          ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph_, g_t, CHW_TO_HWC, 0);
          manager.register_pass(cl);
        }
      }
      for (auto g_t : mwnn_graph_->get_graph_inputs()) {
        if(g_t.get_dims().size() == 4) {
          std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";
          ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph_, g_t, CHW_TO_HWC, 0);
          manager.register_pass(cl);
        }
      }
    }
    for (int node_idx = 0; node_idx < mwnn_graph_->get_graph_nodes().size(); node_idx++) {
      auto g_n = node_list[node_idx];
      /*if(g_n.get_op_type() == "Reshape") {
        ::metawarenn::optimizer::RemoveReshape rr(mwnn_graph_, g_n);
        std::cout << "\n MetaWareNNCC : " << rr.get_name();
        manager.register_pass(rr);
      }
      else*/
      if(g_n.get_op_type() == "BatchNorm") {
        ::metawarenn::optimizer::FuseBatchNorm fbn(mwnn_graph_, g_n);
        std::cout << "\n MetaWareNNCC : " << fbn.get_name();
        manager.register_pass(fbn);
      }
      else  if(g_n.get_op_type() == "Relu") {
        ::metawarenn::optimizer::FuseRelu fr(mwnn_graph_, g_n);
        std::cout << "\n MetaWareNNCC : " << fr.get_name();
        manager.register_pass(fr);
      }
    }
    ::metawarenn::optimizer::CalculateOffset co(mwnn_graph_);
    manager.register_pass(co);
    manager.run_passes();
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
