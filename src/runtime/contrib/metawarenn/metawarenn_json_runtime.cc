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
    exe_graph_ = std::make_shared<metawarenn::ExecutableGraph>(*graph_);
  }

  void Run() override {
    std::cout << "\n In MetaWareNN RUNN!!!";
    std::unordered_map<std::string, float*> graph_inputs;
    std::unordered_map<std::string, float*> graph_outputs;

    for (auto g_ip : graph_->get_graph_ip_tensor()) {
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
    for (auto g_op : graph_->get_graph_op_tensor()) {
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

    ::metawarenn::InferenceApi mwapi;

    for (auto g_ip : graph_->get_graph_ip_tensor()) {
      auto ip_shape = g_ip.get_dims();
      mwapi.prepareInput(graph_inputs[g_ip.get_name()], ip_shape);
    }

    for (auto g_op : graph_->get_graph_op_tensor()) {
      auto op_shape = g_op.get_dims();
      mwapi.prepareOutput(op_shape);
    }

    mwapi.prepareGraph(graph_->get_name());

    mwapi.runGraph();

    for (auto g_op : graph_->get_graph_op_tensor()) {
      auto op_shape = g_op.get_dims();
      mwapi.getOutput(graph_outputs[g_op.get_name()], op_shape);
    }


    // ******************************************* Call to invoke the local run function *****************************************
    //::metawarenn::convert_to_mwnn_format(*graph_, graph_inputs, graph_outputs, CHW_TO_HWC);
    //exe_graph_->runGraph();
  }

 private:
  std::shared_ptr<::metawarenn::Graph> graph_;
  std::shared_ptr<::metawarenn::ExecutableGraph> exe_graph_;
  // Build up the engine based on the input graph.
  void BuildMetaWareNNGraph() {
    graph_count++;
    std::string subgraph_name = "MetaWareNN_" + std::to_string(graph_count);
    graph_ = std::make_shared<::metawarenn::Graph>();
    graph_->set_name(subgraph_name);
    int layer_count = 0;
    std::vector<std::vector<int64_t>> op_shape;
    std::vector<DLDataType> dtypes;
    std::string op_name;
    for (int id = 0; id < nodes_.size(); id++) {
      const auto& node = nodes_[id];
      //std::cout << "\n Node Op Type : " << node.GetOpType() << " Name : " << node.GetOpName();
      if (node.GetOpType() == "kernel") {
        std::string node_name;
        std::string node_op_type;
        std::vector<std::string> node_inputs;
        std::vector<std::string> node_outputs;
        std::vector<metawarenn::Attribute> node_attributes;
        int out_index = 1;

        //Node Inputs Parsing
        for (size_t i = 0; i < node.GetInputs().size(); ++i) {
          auto in_node = node.GetInputs()[i];
          if(in_node.id_ > out_index)
            out_index = in_node.id_;
          std::string ip_name = "node_" + std::to_string(in_node.id_);
          node_inputs.emplace_back(ip_name);
        }
        //Node Output Parsing
        op_name = "node_" + std::to_string(out_index+1);
        node_outputs.emplace_back(op_name);

        //Node Output Shape & Type Parsing
        op_shape = node.GetOpShape();
        dtypes = node.GetOpDataType();
        //Setting MetaWareNN Op Type & Node Name
        if (node.GetOpName() == "nn.conv2d") {
          node_op_type = "Conv";
          node_name = node_op_type + std::to_string(layer_count++);
          std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
          std::vector<std::string> pads = node.GetAttr<std::vector<std::string>>("padding");
          std::vector<std::string> dilations = node.GetAttr<std::vector<std::string>>("dilation");
          int group = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
          auto weight_entry = node.GetInputs()[1];
          std::vector<long int> kernel_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

          metawarenn::Attribute attr_dilate("dilations", std::vector<int>({std::stoi(dilations[0]), std::stoi(dilations[1])}));
          node_attributes.emplace_back(attr_dilate);
          metawarenn::Attribute attr_stride("strides", std::vector<int>({std::stoi(strides[0]), std::stoi(strides[1])}));
          node_attributes.emplace_back(attr_stride);
          metawarenn::Attribute attr_pad("pads", std::vector<int>({std::stoi(pads[0]), std::stoi(pads[1]), std::stoi(pads[2]), std::stoi(pads[3])}));
          node_attributes.emplace_back(attr_pad);
          metawarenn::Attribute attr_group("group", std::vector<int>({group}));
          node_attributes.emplace_back(attr_group);
          metawarenn::Attribute attribute("activation", std::vector<int>({0}));
          node_attributes.emplace_back(attribute);
          metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int>({kernel_shape[2], kernel_shape[3]}));
          node_attributes.emplace_back(attr_kernel_shape);
        }
        else if (node.GetOpName() == "nn.batch_norm") {
          node_op_type = "BatchNormalization";
          node_name = node_op_type + std::to_string(layer_count++);

          float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);
          metawarenn::Attribute attr_epsilon("epsilon", std::vector<float>({epsilon}));
          node_attributes.emplace_back(attr_epsilon);
        }
        else if (node.GetOpName() == "nn.relu") {
          node_op_type = "Relu";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "add") {
          node_op_type = "Add";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "nn.global_avg_pool2d") {
          node_op_type = "GlobalAveragePool";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "nn.avg_pool2d" || node.GetOpName() == "nn.max_pool2d") {
          node_op_type = node.GetOpName() == "nn.avg_pool2d" ? "AveragePool" : "MaxPool";
          node_name = node_op_type + std::to_string(layer_count++);
          auto pool_size = node.GetAttr<std::vector<std::string>>("pool_size");
          auto padding = node.GetAttr<std::vector<std::string>>("padding");
          auto strides = node.GetAttr<std::vector<std::string>>("strides");
          metawarenn::Attribute attr_pool_size("kernel_shape", std::vector<int>({std::stoi(pool_size[0]), std::stoi(pool_size[1])}));
          node_attributes.emplace_back(attr_pool_size);
          metawarenn::Attribute attr_pad("pads", std::vector<int>({std::stoi(padding[0]), std::stoi(padding[1]), std::stoi(padding[2]), std::stoi(padding[3])}));
          node_attributes.emplace_back(attr_pad);
          metawarenn::Attribute attr_stride("strides", std::vector<int>({std::stoi(strides[0]), std::stoi(strides[1])}));
          node_attributes.emplace_back(attr_stride);
        }
        else if (node.GetOpName() == "nn.lrn") {
          node_op_type = "LRN";
          node_name = node_op_type + std::to_string(layer_count++);
          auto alpha = node.GetAttr<std::vector<std::string>>("alpha");
          auto beta = node.GetAttr<std::vector<std::string>>("beta");
          auto size = node.GetAttr<std::vector<std::string>>("size");
          auto axis = node.GetAttr<std::vector<std::string>>("axis");
          auto bias = node.GetAttr<std::vector<std::string>>("bias");
          metawarenn::Attribute attr_alpha("alpha", std::vector<int>({std::stoi(alpha[0])}));
          node_attributes.emplace_back(attr_alpha);
          metawarenn::Attribute attr_beta("beta", std::vector<int>({std::stoi(beta[0])}));
          node_attributes.emplace_back(attr_beta);
          metawarenn::Attribute attr_size("size", std::vector<int>({std::stoi(size[0])}));
          node_attributes.emplace_back(attr_size);
          metawarenn::Attribute attr_axis("axis", std::vector<int>({std::stoi(axis[0])}));
          node_attributes.emplace_back(attr_axis);
          metawarenn::Attribute attr_bias("bias", std::vector<int>({std::stoi(bias[0])}));
          node_attributes.emplace_back(attr_bias);
        }
        else if (node.GetOpName() == "nn.batch_flatten") {
          node_op_type = "BatchFlatten";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "nn.dense") {
          node_op_type = "Dense";
          node_name = node_op_type + std::to_string(layer_count++);
          /*auto units = node.GetAttr<std::vector<std::string>>("units");
          metawarenn::Attribute attr_units("units", std::vector<int>({std::stoi(units[0])}));
          node_attributes.emplace_back(attr_units);*/
        }
        else if (node.GetOpName() == "nn.bias_add") {
          node_op_type = "BiasAdd";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "clip") {
          node_op_type = "Clip";
          node_name = node_op_type + std::to_string(layer_count++);
          auto min = node.GetAttr<std::vector<std::string>>("a_min");
          metawarenn::Attribute attr_min("min", std::vector<int>({std::stoi(min[0])}));
          node_attributes.emplace_back(attr_min);
          auto max = node.GetAttr<std::vector<std::string>>("a_max");
          metawarenn::Attribute attr_max("max", std::vector<int>({std::stoi(max[0])}));
          node_attributes.emplace_back(attr_max);
        }
        else if (node.GetOpName() == "squeeze") {
          node_op_type = "Squeeze";
          node_name = node_op_type + std::to_string(layer_count++);
          auto axis = node.GetAttr<std::vector<std::string>>("axis");
          metawarenn::Attribute attr_axis("axis", std::vector<int>({std::stoi(axis[0])}));
          node_attributes.emplace_back(attr_axis);
        }
        else if (node.GetOpName() == "transpose") {
          node_op_type = "Transpose";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "concatenate") {
          node_op_type = "Concat";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "max") {
          node_op_type = "Max";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "subtract") {
          node_op_type = "Subtract";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "exp") {
          node_op_type = "Exp";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "maximum") {
          node_op_type = "Maximum";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "minimum") {
          node_op_type = "Minimum";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "sum") {
          node_op_type = "Sum";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "divide") {
          node_op_type = "Divide";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "multiply") {
          node_op_type = "Mul";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "mean") {
          node_op_type = "Mean";
          node_name = node_op_type + std::to_string(layer_count++);
          auto axis = node.GetAttr<std::vector<std::string>>("axis");
          std::vector<int> int_axis(axis.size());
          std::transform(axis.begin(), axis.end(), std::back_inserter(int_axis),
                        [](const std::string& str) { return std::stoi(str); });
          metawarenn::Attribute attr_axis("axis", int_axis);
          node_attributes.emplace_back(attr_axis);
        }
        else if (node.GetOpName() == "split") {
          node_op_type = "Split";
          node_name = node_op_type + std::to_string(layer_count++);
          auto indices_or_sections = node.GetAttr<std::vector<std::string>>("indices_or_sections");
          auto axis = node.GetAttr<std::vector<std::string>>("axis");
          metawarenn::Attribute attr_ios("indices_or_sections", std::vector<int>({std::stoi(indices_or_sections[0])}));
          node_attributes.emplace_back(attr_ios);
          metawarenn::Attribute attr_axis("axis", std::vector<int>({std::stoi(axis[0])}));
          node_attributes.emplace_back(attr_axis);
        }
        else if (node.GetOpName() == "strided_slice") {
          node_op_type = "StridedSlice";
          node_name = node_op_type + std::to_string(layer_count++);
          auto begin = node.GetAttr<std::vector<std::string>>("begin");
          auto end = node.GetAttr<std::vector<std::string>>("end");
          auto strides = node.GetAttr<std::vector<std::string>>("strides");

          std::vector<int> int_begin(begin.size());
          std::vector<int> int_end(end.size());
          std::vector<int> int_strides(strides.size());

          std::transform(begin.begin(), begin.end(), std::back_inserter(int_begin),
                        [](const std::string& str) { return std::stoi(str); });
          std::transform(end.begin(), end.end(), std::back_inserter(int_end),
                        [](const std::string& str) { return std::stoi(str); });
          std::transform(strides.begin(), strides.end(), std::back_inserter(int_strides),
                        [](const std::string& str) { return std::stoi(str); });

          metawarenn::Attribute attr_begin("begin_mask", int_begin);
          node_attributes.emplace_back(attr_begin);
          metawarenn::Attribute attr_end("end_mask", int_end);
          node_attributes.emplace_back(attr_end);
          metawarenn::Attribute attr_strides("strides", int_strides);
          node_attributes.emplace_back(attr_strides);
        }
        else if (node.GetOpName() == "nn.softmax") {
          node_op_type = "Softmax";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "reshape") {
          node_op_type = "Reshape";
          node_name = node_op_type + std::to_string(layer_count++);
          std::vector<std::string> shape = node.GetAttr<std::vector<std::string>>("newshape");
          metawarenn::Attribute attr_shape("shape", std::vector<int>({std::stoi(shape[0]), std::stoi(shape[1])}));
          node_attributes.emplace_back(attr_shape);
        }
        else {
          std::cout << "\n Unsupported Op in MetaWareNN backend : " << node.GetOpName();
          exit(1);
        }
        /*std::cout << "\n ================================Node=============================\n";
        std::cout << "\n Name : " << node_name;
        std::cout << "\n Type : " << node_op_type;
        for (auto nip: node_inputs)
          std::cout << "\n Inputs : " << nip;
        for (auto nop: node_outputs)
          std::cout << "\n Outputs : " << nop;*/

        metawarenn::Node m_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
        graph_->set_graph_nodes(m_node);
        auto node = m_node.get_node();
        graph_->graph_nodes[m_node.get_name()] = std::move(node);
      }
    }
    std::vector<int> dims;
    for(int m = 0; m < op_shape.size(); m++)
      for(int n = 0; n < op_shape[m].size(); n++) {
        dims.push_back(op_shape[m][n]);
      }

    //Add Outputs
    auto m_type = get_mwnn_type_tvm(dtypes[0].code);
    //Fills Graph Output Tensor Details - Name, Dims
    metawarenn::Tensor m_op_tensor(op_name, m_type, dims);
    graph_->set_graph_op_tensor(m_op_tensor);
    graph_->set_graph_op_names(m_op_tensor.get_name());

    // Add inputs and constants.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      const auto& node = nodes_[nid];
      std::string ip_name = "node_" + std::to_string(nid);
      if (node.GetOpType() == "input") {
        auto shapes = node.GetOpShape();
        auto dtypes = node.GetOpDataType();

        for (size_t j = 0; j < shapes.size(); ++j) {
          auto shape = shapes[j];
          int size = shape.size();
          std::vector<int> dims(size);
          for(int d = 0; d < size; d++)
            dims[d] = shape[d];
          std::cout << "\nInput Name : " << ip_name;
          std::cout << "\nInput Dims : ";
          for(int k=0; k < dims.size(); k++)
            std::cout << dims[k] << " ";

          auto m_type = get_mwnn_type_tvm(dtypes[j].code);
          std::cout << "\nInput Type : " << (int)m_type;

          //Fills Graph Input Tensor Details - Name, Dims
          metawarenn::Tensor m_ip_tensor(ip_name, m_type, dims);
          graph_->set_graph_ip_names(ip_name);
          auto ip_node = m_ip_tensor.get_node();
          graph_->graph_nodes[ip_name] = std::move(ip_node);
          graph_->set_graph_ip_tensor(m_ip_tensor);
        }
      }
      else if (node.GetOpType() == "const") {
        uint32_t eid = EntryID(nid, 0);
        std::string name = "node_" + std::to_string(nid);
        const DLTensor* data = data_entry_[eid];
        std::vector<int> dims(data->shape, data->shape + data->ndim);
        auto total_elements = std::accumulate(begin(dims), end(dims), 1, std::multiplies<int>());
        std::vector<float> tensor_vec(((float*)(data->data)), ((float*)(data->data)) + total_elements);

        auto m_type = get_mwnn_type_tvm(data->dtype.code);
        metawarenn::Tensor m_tensor(name, dims, m_type, tensor_vec);
        graph_->set_graph_initializers(m_tensor);
        graph_->initializer_names.insert(name);
        auto const_node = m_tensor.get_constant_node();
        graph_->graph_nodes[m_tensor.get_name()] = std::move(const_node);
      }
    }
  }

  static metawarenn::ElementType::element_type get_mwnn_type_tvm(uint8_t tvm_type) {
      switch (tvm_type) {
          case kDLFloat:
              return metawarenn::ElementType::element_type::float_;
          case kDLInt:
              return metawarenn::ElementType::element_type::int32_;
          case kDLUInt:
              return metawarenn::ElementType::element_type::uint32_;
          default:
              return metawarenn::ElementType::element_type::dynamic_;
      }
  }

  void ApplyPasses() {
    ::metawarenn::optimizer::PassManager manager;
    auto node_list = graph_->get_graph_nodes();
    if(CHW_TO_HWC)
    {
      for (auto g_t : graph_->get_graph_initializers()) {
        if(g_t.get_dims().size() == 4) {
          std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";
          ::metawarenn::optimizer::ConvertLayout cl(graph_, g_t, CHW_TO_HWC, 0, true);
          manager.register_pass(cl);
        }
      }
      for (auto g_t : graph_->get_graph_ip_tensor()) {
        if(g_t.get_dims().size() == 4) {
          std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";
          ::metawarenn::optimizer::ConvertLayout cl(graph_, g_t, CHW_TO_HWC, 0, false);
          manager.register_pass(cl);
        }
      }
    }
    for (int node_idx = 0; node_idx < graph_->get_graph_nodes().size(); node_idx++) {
      auto g_n = node_list[node_idx];
      /*if(g_n.get_op_type() == "Reshape") {
        ::metawarenn::optimizer::RemoveReshape rr(graph_, g_n);
        std::cout << "\n MetaWareNNCC : " << rr.get_name();
        manager.register_pass(rr);
      }
      else*/
      if(g_n.get_op_type() == "BatchNormalization") {
        ::metawarenn::optimizer::FuseBatchNorm fbn(graph_, g_n);
        std::cout << "\n MetaWareNNCC : " << fbn.get_name();
        manager.register_pass(fbn);
      }
      else  if(g_n.get_op_type() == "Relu") {
        ::metawarenn::optimizer::FuseRelu fr(graph_, g_n);
        std::cout << "\n MetaWareNNCC : " << fr.get_name();
        manager.register_pass(fr);
      }
    }
    ::metawarenn::optimizer::CalculateOffset co(graph_);
    manager.register_pass(co);
    manager.run_passes();

    auto graph_ip_names = graph_->get_graph_ip_names();
    for (auto g_n : graph_->get_graph_nodes()) {
      for (auto n_ip : g_n.get_inputs()) {
        if(!(graph_->initializer_names.count(n_ip)) && !(std::count(graph_ip_names.begin(), graph_ip_names.end(), n_ip))) {
          if (graph_->get_node_producers().count(n_ip)) {
            graph_->set_node_consumer(n_ip, g_n.get_name());
          }
        }
      }
      for (auto n_op : g_n.get_outputs()) {
        graph_->set_node_producer(n_op, g_n.get_name());
      }
    }
    for (auto itr : graph_->get_node_producers()) {
      std::cout << "\n Produced Tensor : " << itr.first;
      std::cout << "\n      Producer Node : " << itr.second;
    }
    for (auto itr : graph_->get_node_consumers()) {
      std::cout << "\n Consumed Tensor : " << itr.first;
      auto& vitr = itr.second;
      for (auto node_name : vitr) {
          std::cout << "\n      Consumer Node - " << node_name;
      }
    }
  }

  void InvokeNNAC() {
    std::cout << "\n ---------------------------Graph----------------------------- \n";
    std::cout << "\n Graph Name : " << graph_->get_name();
    ::MWNN::MWNNGraphProto graph_proto;
    graph_proto.set_name(graph_->get_name());
    for (auto g_ip : graph_->get_graph_ip_names())
      graph_proto.add_ip_name((g_ip));
    for (auto g_op : graph_->get_graph_op_names())
      graph_proto.add_op_name((g_op));

    std::cout << "\n -----------------------Graph Inputs-------------------------- \n";
    for (auto g_ip : graph_->get_graph_ip_tensor()) {
      std::cout << "\n Input Name : " << g_ip.get_name();
      std::cout << "\n Data Type : " << g_ip.get_type();
      std::cout << "\n Input Dims : ";
      auto input = graph_proto.add_input();
      input->set_name(g_ip.get_name());
      input->set_type(g_ip.get_type());
      for (auto dim : g_ip.get_dims()) {
        std::cout << dim << ",";
        input->add_dims(dim);
      }
    }
    std::cout << "\n -----------------------Graph Outputs-------------------------- \n";
    for (auto g_op : graph_->get_graph_op_tensor()) {
      std::cout << "\n Output Name : " << g_op.get_name();
      std::cout << "\n Data Type : " << g_op.get_type();
      std::cout << "\n Output Dims : ";
      auto output = graph_proto.add_output();
      output->set_name(g_op.get_name());
      output->set_type(g_op.get_type());
      for (auto dim : g_op.get_dims()) {
        std::cout << dim << ",";
        output->add_dims(dim);
      }
    }
    std::cout << "\n -----------------------Graph Nodes-------------------------- \n";
    for (auto g_n : graph_->get_graph_nodes()) {
      std::cout << "\n ================================================================ \n";
      std::cout << "\n Node Name : " << g_n.get_name();
      std::cout << "\n Op Type : " << g_n.get_op_type();
      auto node = graph_proto.add_node();
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
    for (auto g_t : graph_->get_graph_initializers()) {
      auto initializer = graph_proto.add_initializer();
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
    std::cout << "\n -----------------------Graph Tensor Producers-------------------------- \n";
    for (auto producer : graph_->get_node_producers()) {
      std::cout << "\n Produced Tensor : " << producer.first;
      std::cout << "\n      Producer Node : " << producer.second;
      auto pnode = graph_proto.add_producers();
      pnode->set_tensor_name(producer.first);
      pnode->add_node_name(producer.second);
    }
    std::cout << "\n -----------------------Graph Tensor Consumers-------------------------- \n";
    for (auto consumer : graph_->get_node_consumers()) {
      std::cout << "\n Consumed Tensor : " << consumer.first;
      auto& consumer_nodes = consumer.second;
      auto cnode = graph_proto.add_consumers();
      cnode->set_tensor_name(consumer.first);
      for (auto node_name : consumer_nodes) {
        std::cout << "\n      Consumer Node - " << node_name;
        cnode->add_node_name(node_name);
        }
    }

    std::string name = graph_->get_name();
    char* op_path = nullptr;
    op_path = getenv("NNAC_DUMPS_PATH");
    if(!IsPathExist(std::string(op_path))) {
      int check = mkdir(op_path, 0777);
      if(check != 0) {
        std::cout << "\nPlease check the directory path to store the serialized binary!!!!!";
        exit(1);
      }
    }
    auto proto_bin = std::string(op_path) + std::string(name) + ".bin";

    int fp = open(proto_bin.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    std::cout << fp;
    std::cout << graph_proto.SerializeToFileDescriptor(fp);
    close(fp);

    char* lib_path = nullptr;
    lib_path = getenv("METAWARENN_LIB_PATH");
    if(!IsPathExist(std::string(lib_path)))
      std::cout << "\nPlease check the MetaWareNN Library path!!!";
    std::cout << "\n\n=================Initiating NNAC python script via shell script======================\n";
    std::string cmd = "bash " + std::string(lib_path) +"/mwnnconvert/mwnn_convert.sh " + proto_bin + " " + op_path + " " + name + " " + std::to_string(graph_count);
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
