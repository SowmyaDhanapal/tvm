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
#include <fcntl.h>
#include <unistd.h>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/metawarenn_utils.h"
#include "metawarenn_lib/optimizer/pass_manager.h"
#include "metawarenn_lib/executable_network/metawarenn_executable_graph.h"
#include "metawarenn_lib/mwnn_inference_api/mwnn_inference_api.h"
#include "metawarenn_lib/mwnnconvert/mwnn_protobuf/cpp_wrapper/MWNN.pb.h"
#define CHW_TO_HWC 0
#define INVOKE_NNAC 0

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;
static int graph_count;

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
    #if INVOKE_NNAC
      InvokeNNAC();
    #endif
    //Generate Executable Network
    mwnn_exe_graph_ = std::make_shared<metawarenn::MWNNExecutableGraph>(*mwnn_graph_);
  }

  void Run() override {
    std::cout << "\n In MetaWareNN RUNN!!!";
    std::unordered_map<std::string, float*> graph_inputs;
    std::unordered_map<std::string, float*> graph_outputs;

    for (auto g_ip : mwnn_graph_->get_graph_ip_tensor()) {
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
    for (auto g_op : mwnn_graph_->get_graph_op_tensor()) {
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

    ::metawarenn::MWNNInferenceApi mwapi;

    for (auto g_ip : mwnn_graph_->get_graph_ip_tensor()) {
      auto ip_shape = g_ip.get_dims();
      mwapi.prepareInput(graph_inputs[g_ip.get_name()], ip_shape);
    }

    for (auto g_op : mwnn_graph_->get_graph_op_tensor()) {
      auto op_shape = g_op.get_dims();
      mwapi.prepareOutput(op_shape);
    }

    mwapi.prepareGraph(mwnn_graph_->get_name());

    mwapi.runGraph();

    for (auto g_op : mwnn_graph_->get_graph_op_tensor()) {
      auto op_shape = g_op.get_dims();
      mwapi.getOutput(graph_outputs[g_op.get_name()], op_shape);
    }


    // ******************************************* Call to invoke the local run function *****************************************
    //::metawarenn::convert_to_mwnn_format(*mwnn_graph_, graph_inputs, graph_outputs, CHW_TO_HWC);
    //mwnn_exe_graph_->runGraph();
  }

 private:
  std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph_;
  std::shared_ptr<::metawarenn::MWNNExecutableGraph> mwnn_exe_graph_;
  // Build up the engine based on the input graph.
  void BuildMetaWareNNGraph() {
    graph_count++;
    std::string subgraph_name = "MetaWareNN_" + std::to_string(graph_count);
    mwnn_graph_ = std::make_shared<::metawarenn::MWNNGraph>(nodes_, subgraph_name);
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
          ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph_, g_t, CHW_TO_HWC, 0, true);
          manager.register_pass(cl);
        }
      }
      for (auto g_t : mwnn_graph_->get_graph_ip_tensor()) {
        if(g_t.get_dims().size() == 4) {
          std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";
          ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph_, g_t, CHW_TO_HWC, 0, false);
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

  void InvokeNNAC() {
    std::cout << "\n ---------------------------Graph----------------------------- \n";
    std::cout << "\n Graph Name : " << mwnn_graph_->get_name();
    ::MWNN::MWNNGraphProto mwnn_graph_proto;
    mwnn_graph_proto.set_name(mwnn_graph_->get_name());
    for (auto g_ip : mwnn_graph_->get_graph_ip_names())
      mwnn_graph_proto.add_ip_name((g_ip));
    for (auto g_op : mwnn_graph_->get_graph_op_names())
      mwnn_graph_proto.add_op_name((g_op));

    std::cout << "\n -----------------------Graph Inputs-------------------------- \n";
    for (auto g_ip : mwnn_graph_->get_graph_ip_tensor()) {
      std::cout << "\n Input Name : " << g_ip.get_name();
      std::cout << "\n Data Type : " << g_ip.get_type();
      std::cout << "\n Input Dims : ";
      auto input = mwnn_graph_proto.add_input();
      input->set_name(g_ip.get_name());
      input->set_type(g_ip.get_type());
      for (auto dim : g_ip.get_dims()) {
        std::cout << dim << ",";
        input->add_dims(dim);
      }
    }
    std::cout << "\n -----------------------Graph Outputs-------------------------- \n";
    for (auto g_op : mwnn_graph_->get_graph_op_tensor()) {
      std::cout << "\n Output Name : " << g_op.get_name();
      std::cout << "\n Data Type : " << g_op.get_type();
      std::cout << "\n Output Dims : ";
      auto output = mwnn_graph_proto.add_output();
      output->set_name(g_op.get_name());
      output->set_type(g_op.get_type());
      for (auto dim : g_op.get_dims()) {
        std::cout << dim << ",";
        output->add_dims(dim);
      }
    }
    std::cout << "\n -----------------------Graph Nodes-------------------------- \n";
    for (auto g_n : mwnn_graph_->get_graph_nodes()) {
      std::cout << "\n ================================================================ \n";
      std::cout << "\n Node Name : " << g_n.get_name();
      std::cout << "\n Op Type : " << g_n.get_op_type();
      auto node = mwnn_graph_proto.add_node();
      node->set_name(g_n.get_name());
      auto op_type = g_n.get_op_type();
      node->set_op_type(op_type == "DepthwiseConv" ? "Conv" : op_type);
      for (auto n_ip : g_n.get_inputs()) {
        std::cout << "\n Input : n_ip : " << n_ip;
        node->add_ip_name((n_ip));
      }
      for (auto n_op : g_n.get_outputs()) {
        std::cout << "\n Output : n_op : " << n_op;
        node->add_op_name((n_op));
      }
      std::cout << "\n ---------------------------------------------------------------- ";
      for (auto attribute : g_n.get_attributes()) {
        std::cout << "\n Attribute Name : " << attribute.get_name();
        std::cout << "\n Attribute Data : ";
        auto attr = node->add_attribute();
        attr->set_name(attribute.get_name());
        attr->set_type(attribute.get_type());
        if(attribute.get_type() == 6) { //int data
          for(int i = 0; i < attribute.get_int_data().size(); i++){
            attr->add_int_data(attribute.get_int_data()[i]);
            std::cout << attribute.get_int_data()[i] << ",";
          }
        }
        else if(attribute.get_type() == 3) { //float data
          for(int i = 0; i < attribute.get_float_data().size(); i++){
            attr->add_float_data(attribute.get_float_data()[i]);
            std::cout << attribute.get_float_data()[i] << ",";
          }
        }
        else if(attribute.get_type() == 12) { //string data
          for(int i = 0; i < attribute.get_string_data().size(); i++){
            attr->add_string_data(attribute.get_string_data()[i]);
            std::cout << attribute.get_string_data()[i] << ",";
          }
        }
      }
    }
    std::cout << "\n -----------------------Graph Tensors-------------------------- \n";
    for (auto g_t : mwnn_graph_->get_graph_initializers()) {
      auto initializer = mwnn_graph_proto.add_initializer();
      initializer->set_name(g_t.get_name());
      initializer->set_type(g_t.get_type());
      std::cout << "\n Name : " << g_t.get_name();
      std::cout << "\n Type : " << g_t.get_type();
      std::cout << "\n Dims : ";
      for (auto dim : g_t.get_dims()) {
        std::cout << dim << ",";
        initializer->add_dims(dim);
      }
      //std::cout << "\n Tensor values : ";
      for (auto t_val : g_t.get_tensor()) {
        //std::cout << t_val << ",";
        initializer->add_float_data(t_val);
      }
    }
    std::cout << "\n Graph Name : " << mwnn_graph_->get_name();
    std::string name = mwnn_graph_->get_name();
    auto mwnn_op_path = "/path/to/tvm/metawarenn_inference/EVDumps/";
    auto check = mkdir(mwnn_op_path, 0777);
    if(check != 0)
    {
      std::cout << "\nPlease check the directory path to store the serialized binary!!!!!";
      exit(1);
    }
    auto mwnn_proto_bin = std::string(mwnn_op_path) + std::string(name) + ".bin";

    int fp = open(mwnn_proto_bin.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    std::cout << fp;
    std::cout << mwnn_graph_proto.SerializeToFileDescriptor(fp);
    close(fp);

    std::cout << "\n\n=================Initiating NNAC python script via shell script======================\n";
    std::string cmd = "bash /path/to/tvm/src/runtime/contrib/metawarenn/metawarenn_lib/mwnnconvert/mwnn_convert.sh " + mwnn_proto_bin + " " + mwnn_op_path + " " + name + " " + std::to_string(graph_count);
    const char *command = cmd.c_str();
    system(command);
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
