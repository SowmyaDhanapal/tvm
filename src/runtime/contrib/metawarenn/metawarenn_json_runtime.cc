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
#include "metawarenn_lib/mwnnconvert/mwnn_to_onnx_proto.h"
#define CHW_TO_HWC 0
#define HWC_TO_CHW 0
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
    #if !EXECUTABLE_GRAPH_SERIALIZATION
    write_onnx_proto(graph_);
    #endif
    //Generate Executable Network
    #if EXECUTABLE_GRAPH_SERIALIZATION
    exe_graph_ = std::make_shared<metawarenn::ExecutableGraph>(*graph_);
    #endif
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

    #if EXECUTABLE_GRAPH_SERIALIZATION
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
    #endif


    // ******************************************* Call to invoke the local run function *****************************************
    //::metawarenn::convert_to_mwnn_format(*graph_, graph_inputs, graph_outputs, CHW_TO_HWC);
    //exe_graph_->runGraph();
  }

 private:
  std::shared_ptr<::metawarenn::Graph> graph_;
  #if EXECUTABLE_GRAPH_SERIALIZATION
  std::shared_ptr<::metawarenn::ExecutableGraph> exe_graph_;
  #endif
  bool quant_model = false;
  std::string quant_prev_scale;
  std::string quant_prev_zp;
  std::string last_qdq_node;//quant, dequant

  void CreateMWNNNode(const std::string &node_name_,
                      const std::string &node_op_type_,
                      const std::vector<::metawarenn::Attribute> &node_attributes_,
                      const std::vector<std::string> &node_inputs_,
                      const std::vector<std::string> &node_outputs_) {
    metawarenn::Node m_node(node_name_, node_op_type_, node_attributes_, node_inputs_, node_outputs_);
    graph_->set_graph_nodes(m_node);

    std::cout << "\n ================================Node=============================\n";
    std::cout << "\n Name : " << node_name_;
    std::cout << "\n Type : " << node_op_type_;
    for (auto nip: node_inputs_)
      std::cout << "\n Inputs : " << nip;
    for (auto nop: node_outputs_)
      std::cout << "\n Outputs : " << nop;
  }

  void CreateQDQNodes(std::string ip_name, std::string op_name,
                      std::string scale_name, std::string zp_name) {
    std::string quant_node_op_type = "QuantizeLinear";
    std::string quant_node_name = quant_node_op_type + "_" + ip_name;
    std::vector<std::string> quant_node_inputs;
    std::vector<std::string> quant_node_outputs;
    std::vector<::metawarenn::Attribute> quant_node_attributes;
    quant_node_inputs.push_back(ip_name);
    quant_node_inputs.push_back(scale_name);//Output Scale
    quant_node_inputs.push_back(zp_name);//Output ZeroPoint
    quant_node_outputs.push_back(quant_node_name);
    CreateMWNNNode(quant_node_name, quant_node_op_type, quant_node_attributes, quant_node_inputs, quant_node_outputs);

    std::string dequant_node_op_type = "DequantizeLinear";
    std::string dequant_node_name = dequant_node_op_type + "_" + ip_name;
    std::vector<std::string> dequant_node_inputs;
    std::vector<std::string> dequant_node_outputs;
    std::vector<::metawarenn::Attribute> dequant_node_attributes;
    dequant_node_inputs.push_back(quant_node_outputs[0]);
    dequant_node_inputs.push_back(scale_name);//Output Scale
    dequant_node_inputs.push_back(zp_name);//Output ZeroPoint
    dequant_node_outputs.push_back(op_name);
    CreateMWNNNode(dequant_node_name, dequant_node_op_type, dequant_node_attributes, dequant_node_inputs, dequant_node_outputs);
    quant_prev_scale = scale_name;
    quant_prev_zp = zp_name;
    last_qdq_node = "dequant";
  }

  void RetrieveQuantParams(std::string cur_ip_name, std::string *dq_scale_name, std::string *dq_zp_name) {
    for(auto node : graph_->get_graph_nodes()) {
      if (node.get_op_type() == "DequantizeLinear") {
        for (auto n_op : node.get_outputs()) { //Iterate over the dequant node op's
          if(n_op == cur_ip_name) { //If Dequant is the producer of current node ip, then get `scale` & `zp`
            std::vector<std::string> ip_names = node.get_inputs();
            *dq_scale_name = ip_names[1];
            *dq_zp_name = ip_names[2];
          }
        }
      }
    }
  }

  template <class T1, class T2>
  void CreateMWNNTensor(const DLTensor *data, std::string name, metawarenn::ElementType::element_type m_type, std::vector<int> dims) {
    auto total_elements = std::accumulate(begin(dims), end(dims), 1, std::multiplies<int>());
    std::vector<T2> tensor_data = std::vector<T2>(((T1*)(data->data)), ((T1*)(data->data)) + total_elements);
    metawarenn::Tensor m_tensor(name, dims, m_type, tensor_data);
    graph_->set_graph_initializers(m_tensor);
    graph_->initializer_names.insert(name);
    std::cout << "\n Const Node : " << name << " Dims : ";
    for (auto di : dims)
        std::cout << di << ",";
  }

  // Build up the engine based on the input graph.
  void BuildMetaWareNNGraph() {
    graph_count++;
    std::string subgraph_name = "MetaWareNN_" + std::to_string(graph_count);
    graph_ = std::make_shared<::metawarenn::Graph>();
    graph_->set_name(subgraph_name);
    auto tf_flag = std::getenv("TF_TVM_TO_ONNX");
    bool tf_tvm_to_onnx = atoi(tf_flag);
    int layer_count = 0;
    std::vector<std::vector<int64_t>> op_shape;
    std::vector<DLDataType> dtypes;
    std::string op_name;
    std::set<int> merged_node_ids;
    int prev_out_index = 0;
    std::map<std::string, std::string> quant_ip_mapper;
    std::cout << "\n nodes_.size() : " << nodes_.size();
    std::set<int> nid_set;
    // Add inputs and constants.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      const auto& node = nodes_[nid];
      nid_set.insert(nid);
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

          if(static_cast<int>(dtypes[j].code) == kDLUInt) {
            quant_model = true;
            m_type = metawarenn::ElementType::element_type::uint8_;
            std::string dequant_node_op_type = "DequantizeLinear";
            std::string dequant_node_name = dequant_node_op_type + "_" + ip_name;
            std::vector<std::string> dequant_node_inputs;
            std::vector<std::string> dequant_node_outputs;
            std::vector<::metawarenn::Attribute> dequant_node_attributes;

            std::string scale_name = dequant_node_name + std::string("_scale");
            std::vector<float> tensor_vec_scale = {0.0078125};
            ::metawarenn::Tensor scale_tensor(scale_name, std::vector<int>({tensor_vec_scale.size()}), metawarenn::ElementType::element_type::float_, tensor_vec_scale);
            graph_->set_graph_initializers(scale_tensor);
            graph_->initializer_names.insert(scale_name);

            std::string zp_name = dequant_node_name + std::string("_zero_point");
            std::vector<int32_t> tensor_vec_zp = {128};
            ::metawarenn::Tensor zp_tensor(zp_name, std::vector<int>({tensor_vec_zp.size()}), metawarenn::ElementType::element_type::uint8_, tensor_vec_zp);
            graph_->set_graph_initializers(zp_tensor);
            graph_->initializer_names.insert(zp_name);

            dequant_node_inputs.push_back(ip_name);
            dequant_node_inputs.push_back(scale_name);//Scale
            dequant_node_inputs.push_back(zp_name);//ZeroPoint
            dequant_node_outputs.push_back(dequant_node_name);
            CreateMWNNNode(dequant_node_name, dequant_node_op_type, dequant_node_attributes, dequant_node_inputs, dequant_node_outputs);
            quant_ip_mapper[ip_name] = dequant_node_name;
            last_qdq_node = "dequant";
        }

          //Fills Graph Input Tensor Details - Name, Dims
          metawarenn::Tensor m_ip_tensor(ip_name, m_type, dims);
          graph_->set_graph_ip_names(ip_name);
          graph_->set_graph_ip_tensor(m_ip_tensor);
        }
      }
      else if (node.GetOpType() == "const") {
        uint32_t eid = EntryID(nid, 0);
        std::string name = "node_" + std::to_string(nid);
        const DLTensor* data = data_entry_[eid];
        std::vector<int> dims(data->shape, data->shape + data->ndim);
        metawarenn::ElementType::element_type m_type;
        if(static_cast<int>(data->dtype.code) == kDLUInt) {
          m_type = metawarenn::ElementType::element_type::uint8_;
          CreateMWNNTensor<uint8_t,int32_t>(data, name, m_type, dims);
        }
        else if(static_cast<int>(data->dtype.code) == kDLFloat) {
          m_type = metawarenn::ElementType::element_type::float_;
          CreateMWNNTensor<float,float>(data, name, m_type, dims);
        }
        else if(static_cast<int>(data->dtype.code) == kDLInt) {
          m_type = metawarenn::ElementType::element_type::int32_;
          CreateMWNNTensor<int32_t,int32_t>(data, name, m_type, dims);
        }
        else{
          std::cout << "\n Error: Unhandled constant datatype!!!";
          exit(1);
        }
        //if(data->shape == 0 && data->ndim == 0)
        //  continue;
      }
    }
    for (int id = 0; id < nodes_.size(); id++) {
      const auto& node = nodes_[id];
      //std::cout << "\n Node Op Type : " << node.GetOpType() << " Name : " << node.GetOpName();
      if (node.GetOpType() == "kernel") {
        if(merged_node_ids.count(id))
          continue;
        std::string node_name;
        std::string node_op_type;
        std::vector<std::string> node_inputs;
        std::vector<std::string> node_outputs;
        std::vector<metawarenn::Attribute> node_attributes;
        int out_index = 1;
        //Node Inputs Parsing
        for (size_t i = 0; i < node.GetInputs().size(); ++i) {
          auto in_node = node.GetInputs()[i];
          if(in_node.id_ >= out_index)
            out_index = in_node.id_ + 1;
          std::string ip_name = "";
          // Check if the input is an initializer
          if (std::count(input_nodes_.begin(), input_nodes_.end(), in_node.id_))
            ip_name = "node_" + std::to_string(in_node.id_);
          // Check if the input is computational node output & append the index with ip_name
          else
            ip_name = "node_" + std::to_string(in_node.id_) + "_" + std::to_string(in_node.index_);
          if(quant_ip_mapper.count(ip_name) == 0)
            node_inputs.emplace_back(ip_name);
          else
            node_inputs.emplace_back(quant_ip_mapper[ip_name]);
        }
        if(prev_out_index >= out_index)
          out_index = prev_out_index + 1;
        // Check if node output index clashes with the const/Input index, if so use the next index for node output
        while(nid_set.count(out_index)) {
          out_index = out_index + 1;
        }
        prev_out_index = out_index;
        // Node Output Parsing
        for (int i = 0; i < node.GetNumOutput(); ++i) {
          // Avoid adding training related outputs in batchnorm node
          if(node.GetOpName() == "nn.batch_norm" && i == 1)
            break;
          else {
            op_name = "node_" + std::to_string(out_index) + "_" + std::to_string(i);
            node_outputs.emplace_back(op_name);
          }
        }
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
          std::vector<std::string> kernel_size = node.GetAttr<std::vector<std::string>>("kernel_size");
          int64_t group = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
          auto weight_entry = node.GetInputs()[1];

          metawarenn::Attribute attr_dilate("dilations", std::vector<int64_t>({std::stoi(dilations[0]), std::stoi(dilations[1])}));
          node_attributes.emplace_back(attr_dilate);
          metawarenn::Attribute attr_group("group", group);
          node_attributes.emplace_back(attr_group);
          metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>({std::stoi(kernel_size[0]), std::stoi(kernel_size[1])}));
          node_attributes.emplace_back(attr_kernel_shape);
          metawarenn::Attribute attr_pad("pads", std::vector<int64_t>({std::stoi(pads[0]), std::stoi(pads[1]), std::stoi(pads[2]), std::stoi(pads[3])}));
          node_attributes.emplace_back(attr_pad);
          metawarenn::Attribute attr_stride("strides", std::vector<int64_t>({std::stoi(strides[0]), std::stoi(strides[1])}));
          node_attributes.emplace_back(attr_stride);
        }
        else if (node.GetOpName() == "nn.conv2d_transpose") {
          node_op_type = "ConvTranspose";
          node_name = node_op_type + std::to_string(layer_count++);
          std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
          std::vector<std::string> pads = node.GetAttr<std::vector<std::string>>("padding");
          std::vector<std::string> dilations = node.GetAttr<std::vector<std::string>>("dilation");
          std::vector<std::string> op_padding = node.GetAttr<std::vector<std::string>>("output_padding");
          std::vector<std::string> kernel_size = node.GetAttr<std::vector<std::string>>("kernel_size");
          int64_t group = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
          auto weight_entry = node.GetInputs()[1];
          std::vector<int64_t> op_dims;
          for(int m = 0; m < 1; m++)
            for(int n = 0; n < op_shape[m].size(); n++) {
              op_dims.push_back(op_shape[m][n]);
            }

          metawarenn::Attribute attr_dilate("dilations", std::vector<int64_t>({std::stoi(dilations[0]), std::stoi(dilations[1])}));
          node_attributes.emplace_back(attr_dilate);
          metawarenn::Attribute attr_group("group", group);
          node_attributes.emplace_back(attr_group);
          metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>({std::stoi(kernel_size[0]), std::stoi(kernel_size[1])}));
          node_attributes.emplace_back(attr_kernel_shape);
          metawarenn::Attribute attr_op_padding("output_padding", std::vector<int64_t>({std::stoi(op_padding[0]), std::stoi(op_padding[1])}));
          node_attributes.emplace_back(attr_op_padding);
          metawarenn::Attribute attr_op_shape("output_shape", op_dims);
          node_attributes.emplace_back(attr_op_shape);
          metawarenn::Attribute attr_pad("pads", std::vector<int64_t>({std::stoi(pads[0]), std::stoi(pads[1]), std::stoi(pads[2]), std::stoi(pads[3])}));
          node_attributes.emplace_back(attr_pad);
          metawarenn::Attribute attr_stride("strides", std::vector<int64_t>({std::stoi(strides[0]), std::stoi(strides[1])}));
          node_attributes.emplace_back(attr_stride);
        }
        else if (node.GetOpName() == "nn.batch_norm") {
          node_op_type = "BatchNormalization";
          node_name = node_op_type + std::to_string(layer_count++);
          float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);
          metawarenn::Attribute attr_epsilon("epsilon", epsilon);
          node_attributes.emplace_back(attr_epsilon);
        }
        else if (node.GetOpName() == "nn.instance_norm") {
          node_op_type = "InstanceNormalization";
          node_name = node_op_type + std::to_string(layer_count++);
          float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);
          metawarenn::Attribute attr_epsilon("epsilon", epsilon);
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
        else if (node.GetOpName() == "nn.max_pool2d") {
          node_op_type = "MaxPool";
          node_name = node_op_type + std::to_string(layer_count++);
          auto dilations = node.GetAttr<std::vector<std::string>>("dilation");
          auto pool_size = node.GetAttr<std::vector<std::string>>("pool_size");
          auto padding = node.GetAttr<std::vector<std::string>>("padding");
          auto strides = node.GetAttr<std::vector<std::string>>("strides");
          int64_t ceil_mode = std::stoi(node.GetAttr<std::vector<std::string>>("ceil_mode")[0]);

          metawarenn::Attribute attr_ceil_model("ceil_mode", ceil_mode);
          node_attributes.emplace_back(attr_ceil_model);
          metawarenn::Attribute attr_dilations("dilations", std::vector<int64_t>({std::stoi(dilations[0]), std::stoi(dilations[1])}));
          node_attributes.emplace_back(attr_dilations);
          metawarenn::Attribute attr_pool_size("kernel_shape", std::vector<int64_t>({std::stoi(pool_size[0]), std::stoi(pool_size[1])}));
          node_attributes.emplace_back(attr_pool_size);
          metawarenn::Attribute attr_pad("pads", std::vector<int64_t>({std::stoi(padding[0]), std::stoi(padding[1]), std::stoi(padding[2]), std::stoi(padding[3])}));
          node_attributes.emplace_back(attr_pad);
          metawarenn::Attribute attr_stride("strides", std::vector<int64_t>({std::stoi(strides[0]), std::stoi(strides[1])}));
          node_attributes.emplace_back(attr_stride);

          std::string scale_name, zp_name;
          RetrieveQuantParams(node_inputs[0], &scale_name, &zp_name);
          if(quant_model) {
            std::string dequant_node_op_type = "DequantizeLinear";
            std::string dequant_node_name = dequant_node_op_type + "_" + node_outputs[0];
            CreateQDQNodes(node_outputs[0], dequant_node_name, scale_name, zp_name);
            quant_ip_mapper[node_outputs[0]] = dequant_node_name;
          }
        }
        else if (node.GetOpName() == "nn.avg_pool2d") {
          node_op_type = "AveragePool";
          node_name = node_op_type + std::to_string(layer_count++);
          auto pool_size = node.GetAttr<std::vector<std::string>>("pool_size");
          auto padding = node.GetAttr<std::vector<std::string>>("padding");
          auto strides = node.GetAttr<std::vector<std::string>>("strides");
          int64_t ceil_mode = std::stoi(node.GetAttr<std::vector<std::string>>("ceil_mode")[0]);
          int64_t count_include_pad = std::stoi(node.GetAttr<std::vector<std::string>>("count_include_pad")[0]);

          metawarenn::Attribute attr_ceil_model("ceil_mode", ceil_mode);
          node_attributes.emplace_back(attr_ceil_model);
          metawarenn::Attribute attr_count_include_pad("count_include_pad", count_include_pad);
          node_attributes.emplace_back(attr_count_include_pad);
          metawarenn::Attribute attr_pool_size("kernel_shape", std::vector<int64_t>({std::stoi(pool_size[0]), std::stoi(pool_size[1])}));
          node_attributes.emplace_back(attr_pool_size);
          metawarenn::Attribute attr_pad("pads", std::vector<int64_t>({std::stoi(padding[0]), std::stoi(padding[1]), std::stoi(padding[2]), std::stoi(padding[3])}));
          node_attributes.emplace_back(attr_pad);
          metawarenn::Attribute attr_stride("strides", std::vector<int64_t>({std::stoi(strides[0]), std::stoi(strides[1])}));
          node_attributes.emplace_back(attr_stride);

          std::string scale_name, zp_name;
          RetrieveQuantParams(node_inputs[0], &scale_name, &zp_name);
          if(quant_model) {
            std::string dequant_node_op_type = "DequantizeLinear";
            std::string dequant_node_name = dequant_node_op_type + "_" + node_outputs[0];
            CreateQDQNodes(node_outputs[0], dequant_node_name, scale_name, zp_name);
            quant_ip_mapper[node_outputs[0]] = dequant_node_name;
          }
        }
        else if (node.GetOpName() == "nn.lrn") {
          node_op_type = "LRN";
          node_name = node_op_type + std::to_string(layer_count++);
          auto alpha = node.GetAttr<std::vector<std::string>>("alpha");
          auto beta = node.GetAttr<std::vector<std::string>>("beta");
          auto size = node.GetAttr<std::vector<std::string>>("size");
          auto bias = node.GetAttr<std::vector<std::string>>("bias");

          metawarenn::Attribute attr_alpha("alpha", std::stof(alpha[0]));
          node_attributes.emplace_back(attr_alpha);
          metawarenn::Attribute attr_beta("beta", std::stof(beta[0]));
          node_attributes.emplace_back(attr_beta);
          metawarenn::Attribute attr_bias("bias", std::stof(bias[0]));
          node_attributes.emplace_back(attr_bias);
          metawarenn::Attribute attr_size("size", (int64_t)std::stoi(size[0]));
          node_attributes.emplace_back(attr_size);
        }
        else if (node.GetOpName() == "nn.batch_flatten") {
          node_op_type = "Flatten";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "nn.dense") {
          node_op_type = "Gemm";
          node_name = node_op_type + std::to_string(layer_count++);
          metawarenn::Attribute attr_transB("transB", (int64_t)1);//TODO - Do Check & Pass flags
          node_attributes.emplace_back(attr_transB);
          if(id+2 < nodes_.size()) {
            for(int k = id+1; k <= id+2; k++) {
              const auto& bias_node = nodes_[k];
              if(bias_node.GetOpType() == "kernel" && bias_node.GetOpName() == "add" || bias_node.GetOpName() == "nn.bias_add") {
                //BiasNode Input Parsing = (Gemm)(onnx) -> (Flatten + Dense + Add) in TVM
                auto bias_in_node = bias_node.GetInputs()[1];//index-0 --> Feature Tensor, index-1 Bias Values
                std::string ip_name = "node_" + std::to_string(bias_in_node.id_);
                node_inputs.emplace_back(ip_name);
                merged_node_ids.insert(k);
                op_name = "node_" + std::to_string(bias_in_node.id_+1) + "_" + std::to_string(bias_in_node.index_);
                node_outputs[0] = op_name;
                prev_out_index = bias_in_node.id_+1;
              }
            }
          }
        }
        else if (node.GetOpName() == "clip") {
          node_op_type = "Clip";
          node_name = node_op_type + std::to_string(layer_count++);
          auto min = node.GetAttr<std::vector<std::string>>("a_min");
          auto max = node.GetAttr<std::vector<std::string>>("a_max");

          std::string clip_ip_min = node_name + "_min";
          metawarenn::Tensor min_tensor(clip_ip_min, std::vector<int>({1}), metawarenn::ElementType::element_type::float_, std::vector<float>({std::stof(min[0])}));
          graph_->set_graph_initializers(min_tensor);
          graph_->initializer_names.insert(clip_ip_min);
          node_inputs.emplace_back(clip_ip_min);

          std::string clip_ip_max = node_name + "_max";
          metawarenn::Tensor max_tensor(clip_ip_max, std::vector<int>({1}), metawarenn::ElementType::element_type::float_, std::vector<float>({std::stof(max[0])}));
          graph_->set_graph_initializers(max_tensor);
          graph_->initializer_names.insert(clip_ip_max);
          node_inputs.emplace_back(clip_ip_max);
        }
        else if (node.GetOpName() == "squeeze") {
          node_op_type = "Squeeze";
          node_name = node_op_type + std::to_string(layer_count++);
          auto axes = node.GetAttr<std::vector<std::string>>("axis");
          std::vector<int64_t> tensor_axes(axes.size());
          for(int itr = 0; itr < axes.size(); itr++) {
            if(tf_tvm_to_onnx) //To handle the layout from HWC(TFLite) to CHW(ONNX)
              tensor_axes[itr] = std::stof(axes[itr]) + 1;
            else
              tensor_axes[itr] = std::stof(axes[itr]);
           }

          std::string axes_ip_name = node_name + "axes";
          metawarenn::Tensor axes_tensor(axes_ip_name, std::vector<int>({tensor_axes.size()}), metawarenn::ElementType::element_type::int64_, tensor_axes);
          graph_->set_graph_initializers(axes_tensor);
          graph_->initializer_names.insert(axes_ip_name);
          node_inputs.emplace_back(axes_ip_name);;
        }
        else if (node.GetOpName() == "transpose") {
          node_op_type = "Transpose";
          node_name = node_op_type + std::to_string(layer_count++);
          auto perm = node.GetAttr<std::vector<std::string>>("axes");
          std::vector<int64_t> int_perm(perm.size());
          for(int itr = 0; itr < perm.size(); itr++) {
            int_perm[itr] = std::stoi(perm[itr]);
           }
          metawarenn::Attribute attr_perm("perm", int_perm);
          node_attributes.emplace_back(attr_perm);
        }
        else if (node.GetOpName() == "concatenate") {
          node_op_type = "Concat";
          node_name = node_op_type + std::to_string(layer_count++);
          auto axis = node.GetAttr<std::vector<std::string>>("axis");
          metawarenn::Attribute attr_axis;
          if(tf_tvm_to_onnx) //To handle the layout from HWC(TFLite) to CHW(ONNX)
            attr_axis = metawarenn::Attribute("axis", (int64_t)std::stoi(axis[0])-2);
          else
            attr_axis = metawarenn::Attribute("axis", (int64_t)std::stoi(axis[0]));
          node_attributes.emplace_back(attr_axis);
        }
        else if (node.GetOpName() == "max") {
          if((id+4 < nodes_.size()) &&
             (nodes_[id+1].GetOpType() == "kernel" && nodes_[id+1].GetOpName() == "subtract") &&
             (nodes_[id+2].GetOpType() == "kernel" && nodes_[id+2].GetOpName() == "exp") &&
             (nodes_[id+3].GetOpType() == "kernel" && nodes_[id+3].GetOpName() == "sum") &&
             (nodes_[id+4].GetOpType() == "kernel" && nodes_[id+4].GetOpName() == "divide")) {
               node_op_type = "Softmax";
               node_name = node_op_type + std::to_string(layer_count++);
               metawarenn::Attribute attr_axis("axis", (int64_t)1);//Defaults to 1(C) because, 0th axis mostly describes the batch_size(N)
               node_attributes.emplace_back(attr_axis);
               id = id + 4;
               const auto& node = nodes_[id];
               //Node Output Shape & Type Parsing
               op_shape = node.GetOpShape();
               dtypes = node.GetOpDataType();
             }
           else {
             node_op_type = "Max";
             node_name = node_op_type + std::to_string(layer_count++);
           }
        }
        else if (node.GetOpName() == "subtract") {
          node_op_type = "Sub";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "exp") {
          node_op_type = "Exp";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "nn.matmul") {
          node_op_type = "MatMul";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "nn.leaky_relu") {
          node_op_type = "LeakyRelu";
          node_name = node_op_type + std::to_string(layer_count++);
          auto alpha = node.GetAttr<std::vector<std::string>>("alpha");

          metawarenn::Attribute attr_alpha("alpha", std::stof(alpha[0]));
          node_attributes.emplace_back(attr_alpha);
        }
        else if (node.GetOpName() == "nn.prelu") {
          node_op_type = "PRelu";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "nn.dropout") {
          node_op_type = "Dropout";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "nn.upsampling" || node.GetOpName() == "nn.upsampling3d") {
          node_op_type = "Resize";
          node_name = node_op_type + std::to_string(layer_count++);
          auto method = node.GetAttr<std::vector<std::string>>("method");
          auto scale_h = node.GetAttr<std::vector<std::string>>("scale_h");
          auto scale_w = node.GetAttr<std::vector<std::string>>("scale_w");
          std::vector<float> scales = {1, 1, std::stof(scale_h[0]), std::stof(scale_w[0])};
          if(method[0] == "nearest_neighbor")
            method[0] = "nearest";
          else
            method[0] = "linear"; //Gets handled in onnx based on dims

          metawarenn::Attribute attr_method("mode", method[0]);
          node_attributes.emplace_back(attr_method);
          metawarenn::Tensor roi_tensor(node_name + "_roi", std::vector<int>({0}), metawarenn::ElementType::element_type::float_, std::vector<float>{});
          graph_->set_graph_initializers(roi_tensor);
          graph_->initializer_names.insert(roi_tensor.get_name());
          node_inputs.emplace_back(roi_tensor.get_name());
          metawarenn::Tensor scale_tensor(node_name + "_scale", std::vector<int>({4}), metawarenn::ElementType::element_type::float_, scales);
          graph_->set_graph_initializers(scale_tensor);
          graph_->initializer_names.insert(scale_tensor.get_name());
          node_inputs.emplace_back(scale_tensor.get_name());
        }
        else if (node.GetOpName() == "sum") {
          node_op_type = "Sum";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "divide") {
          node_op_type = "Div";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "multiply") {
          node_op_type = "Mul";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "mean") {
          auto axis = node.GetAttr<std::vector<std::string>>("axis");
          int64_t keepdims = std::stoi(node.GetAttr<std::vector<std::string>>("keepdims")[0]);
          int64_t exclude = std::stoi(node.GetAttr<std::vector<std::string>>("exclude")[0]);
          //Ensure the HWC layout for the reduction from TFLite model
          if(tf_tvm_to_onnx && std::stoi(axis[0]) == 1 && std::stoi(axis[1]) == 2) {
            node_op_type = "GlobalAveragePool";
            node_name = node_op_type + std::to_string(layer_count++);
          }
          else {
            node_op_type = "ReduceMean"; // keepdims = 0 reduces the shape according to axes & keepdims = 1 maintains same shape
            node_name = node_op_type + std::to_string(layer_count++);
            std::vector<int64_t> int_axis(axis.size());
            for(int i = 0; i < axis.size(); i++)
              int_axis[i] = std::stoi(axis[i]);
            metawarenn::Attribute attr_axis("axes", int_axis);
            node_attributes.emplace_back(attr_axis);
            metawarenn::Attribute attr_keepdims("keepdims", keepdims);
            node_attributes.emplace_back(attr_keepdims);
          }
        }
        else if (node.GetOpName() == "split") {
          node_op_type = "Split";
          node_name = node_op_type + std::to_string(layer_count++);
          auto axis = node.GetAttr<std::vector<std::string>>("axis");
          auto indices_or_sections = node.GetAttr<std::vector<std::string>>("indices_or_sections");
          int num_splits = node.GetNumOutput();
          int split_val = std::stoi(indices_or_sections[0]);
          std::vector<int64_t> split(num_splits);
          auto out_shape = node.GetOpShape();
          int i = 0;
          // To Handle more than 2 splits & output shape has the ONNX required split info in index 4
          if(num_splits > 2) {
            for(auto shape: out_shape)
              split[i++] = shape[4];
          }
          // To Handle 2 splits and indices_or_sections carries the split size
          else
            split = {split_val, split_val};
          metawarenn::Attribute attr_axis;
          if(tf_tvm_to_onnx)
            attr_axis = metawarenn::Attribute("axis", (int64_t)std::stoi(axis[0])-2); //To handle the layout from HWC(TFLite) to CHW(ONNX)
          else {
            attr_axis = metawarenn::Attribute("axis", (int64_t)std::stoi(axis[0]));
            metawarenn::Tensor split_tensor(node_name + "_split", std::vector<int>{num_splits}, metawarenn::ElementType::element_type::int64_, split);
            graph_->set_graph_initializers(split_tensor);
            graph_->initializer_names.insert(split_tensor.get_name());
            node_inputs.emplace_back(split_tensor.get_name());
            node_attributes.emplace_back(attr_axis);
          }
        }
        else if (node.GetOpName() == "strided_slice") {
          node_op_type = "Slice";
          node_name = node_op_type + std::to_string(layer_count++);
          auto begin = node.GetAttr<std::vector<std::string>>("begin");
          auto end = node.GetAttr<std::vector<std::string>>("end");
          std::vector<int64_t> tensor_begin(begin.size());
          std::vector<int64_t> tensor_end(begin.size());
          if(tf_tvm_to_onnx) {
            tensor_begin[0] = std::stoi(begin[0]); tensor_begin[1] = std::stoi(begin[3]);
            tensor_begin[2] = std::stoi(begin[1]); tensor_begin[3] = std::stoi(begin[2]);
            tensor_end[0] = std::stoi(end[0]); tensor_end[1] = std::stoi(end[3]);
            tensor_end[2] = std::stoi(end[1]); tensor_end[3] = std::stoi(end[2]);
          }
          else {
            std::transform(begin.begin(), begin.end(), std::back_inserter(tensor_begin),
                           [](const std::string& str) { return std::stoi(str); });
            std::transform(end.begin(), end.end(), std::back_inserter(tensor_end),
                           [](const std::string& str) { return std::stoi(str); });
          }

          std::string begin_ip_name = node_name + "_ip_begin";
          metawarenn::Tensor begin_tensor(begin_ip_name, std::vector<int>({tensor_begin.size()}), metawarenn::ElementType::element_type::int64_, tensor_begin);
          graph_->set_graph_initializers(begin_tensor);
          graph_->initializer_names.insert(begin_ip_name);
          node_inputs.emplace_back(begin_ip_name);

          std::string end_ip_name = node_name + "_ip_end";
          metawarenn::Tensor end_tensor(end_ip_name, std::vector<int>({tensor_end.size()}), metawarenn::ElementType::element_type::int64_, tensor_end);
          graph_->set_graph_initializers(end_tensor);
          graph_->initializer_names.insert(end_ip_name);
          node_inputs.emplace_back(end_ip_name);
        }
        else if (node.GetOpName() == "nn.softmax") {
          node_op_type = "Softmax";
          node_name = node_op_type + std::to_string(layer_count++);
          auto axis = node.GetAttr<std::vector<std::string>>("axis");
          metawarenn::Attribute attr_axis("axis", (int64_t)std::stoi(axis[0]));
          node_attributes.emplace_back(attr_axis);
        }
        else if (node.GetOpName() == "reshape") {
          node_op_type = "Reshape";
          node_name = node_op_type + std::to_string(layer_count++);
          std::string reshape_ip_name = node_name + "_ip";
          auto new_shape = node.GetAttr<std::vector<std::string>>("newshape");
          std::vector<int64_t> tensor_vec(new_shape.size(), 0);
          if(tf_tvm_to_onnx & new_shape.size()==4) { //NHWC -> NCHW
            tensor_vec[0] = std::stoi(new_shape[0]);//N
            tensor_vec[1] = std::stoi(new_shape[3]);//C
            tensor_vec[2] = std::stoi(new_shape[1]);//H
            tensor_vec[3] = std::stoi(new_shape[2]);//W
          }
          else{
            for(int i=0; i<new_shape.size(); i++)
              tensor_vec[i] = std::stoi(new_shape[i]);
          }
          metawarenn::Tensor reshape_tensor(reshape_ip_name, std::vector<int>({tensor_vec.size()}), metawarenn::ElementType::element_type::int64_, tensor_vec);
          graph_->set_graph_initializers(reshape_tensor);
          graph_->initializer_names.insert(reshape_ip_name);
          node_inputs.emplace_back(reshape_ip_name);

          std::string scale_name, zp_name;
          RetrieveQuantParams(node_inputs[0], &scale_name, &zp_name);
          if(quant_model) {
            std::string dequant_node_op_type = "DequantizeLinear";
            std::string dequant_node_name = dequant_node_op_type + "_" + node_outputs[0];
            CreateQDQNodes(node_outputs[0], dequant_node_name, scale_name, zp_name);
            quant_ip_mapper[node_outputs[0]] = dequant_node_name;
          }
        }
        else if (node.GetOpName() == "topk") {
          node_op_type = "TopK";
          node_name = node_op_type + std::to_string(layer_count++);
          auto axis = node.GetAttr<std::vector<std::string>>("axis");
          int64_t is_ascend = std::stoi(node.GetAttr<std::vector<std::string>>("is_ascend")[0]);

          metawarenn::Attribute attr_axis("axis", (int64_t)std::stoi(axis[0]));
          node_attributes.emplace_back(attr_axis);
          metawarenn::Attribute attr_largest("largest", (int64_t)!is_ascend);
          node_attributes.emplace_back(attr_largest);
          }
        else if (node.GetOpName() == "image.resize1d" || node.GetOpName() == "image.resize2d" || node.GetOpName() == "image.resize3d") {
          node_op_type = "Resize";
          node_name = node_op_type + std::to_string(layer_count++);
          auto cord_trans_mode = node.GetAttr<std::vector<std::string>>("coordinate_transformation_mode");
          auto cubic_alpha = node.GetAttr<std::vector<std::string>>("cubic_alpha");
          auto cubic_exclude = node.GetAttr<std::vector<std::string>>("cubic_exclude");
          auto method = node.GetAttr<std::vector<std::string>>("method");
          auto size_param = node.GetAttr<std::vector<std::string>>("size");
          auto out_shape = node.GetOpShape();
          std::vector<int64_t> size_vec(4);
          int i = 0;
          for(auto shape: out_shape)
            for(auto sh: shape)
              size_vec[i++] = sh;
          if(method[0] == "nearest_neighbor")
            method[0] = "nearest";
          auto rounding_method = node.GetAttr<std::vector<std::string>>("rounding_method");
          if(rounding_method[0] == "round")
            rounding_method[0] = "round_prefer_floor";

          metawarenn::Attribute attr_cord_trans_mode("coordinate_transformation_mode", cord_trans_mode[0]);
          node_attributes.emplace_back(attr_cord_trans_mode);
          metawarenn::Attribute attr_cubic_alpha("cubic_coeff_a", std::stof(cubic_alpha[0]));
          node_attributes.emplace_back(attr_cubic_alpha);
          metawarenn::Attribute attr_cubic_exclude("exclude_outside", (int64_t)std::stoi(cubic_exclude[0]));
          node_attributes.emplace_back(attr_cubic_exclude);
          metawarenn::Attribute attr_method("mode", method[0]);
          node_attributes.emplace_back(attr_method);
          metawarenn::Attribute attr_rounding_method("nearest_mode", rounding_method[0]);
          node_attributes.emplace_back(attr_rounding_method);
          // ROI, Scale is not available in node attribute. Adding empty vector to maintain the tensor order in ONNX
          metawarenn::Tensor roi_tensor(node_name + "_roi", std::vector<int>({0}), metawarenn::ElementType::element_type::float_, std::vector<float>{});
          graph_->set_graph_initializers(roi_tensor);
          graph_->initializer_names.insert(roi_tensor.get_name());
          node_inputs.emplace_back(roi_tensor.get_name());
          metawarenn::Tensor scale_tensor(node_name + "_scale", std::vector<int>({0}), metawarenn::ElementType::element_type::float_, std::vector<float>{});
          graph_->set_graph_initializers(scale_tensor);
          graph_->initializer_names.insert(scale_tensor.get_name());
          node_inputs.emplace_back(scale_tensor.get_name());
          metawarenn::Tensor size_tensor(node_name + "_size", std::vector<int>({size_vec.size()}), metawarenn::ElementType::element_type::int64_, size_vec);
          graph_->set_graph_initializers(size_tensor);
          graph_->initializer_names.insert(size_tensor.get_name());
          node_inputs.emplace_back(size_tensor.get_name());
          }
        else if (node.GetOpName() == "image.crop_and_resize") {
          node_op_type = "Resize";
          node_name = node_op_type + std::to_string(layer_count++);
          auto extrapolation_value = node.GetAttr<std::vector<std::string>>("extrapolation_value");
          auto method = node.GetAttr<std::vector<std::string>>("method");
          if(method[0] == "nearest_neighbor")
            method[0] = "nearest";
          else if(method[0] == "bilinear")
            method[0] = "linear"; //Gets handled in onnx

          metawarenn::Attribute attr_cord_trans_mode("coordinate_transformation_mode", "tf_crop_and_resize");
          node_attributes.emplace_back(attr_cord_trans_mode);
          metawarenn::Attribute attr_extrapolation_value("extrapolation_value", std::stof(extrapolation_value[0]));
          node_attributes.emplace_back(attr_extrapolation_value);
          metawarenn::Attribute attr_method("mode", method[0]);
          node_attributes.emplace_back(attr_method);
        }
        else if (node.GetOpName() == "nn.pad") {
          node_op_type = "Pad";
          node_name = node_op_type + std::to_string(layer_count++);
          std::string pad_ip_name = node_name + "_ip";
          auto padding = node.GetAttr<std::vector<std::string>>("padding");
          std::vector<int> dims{padding.size()};
          std::vector<int64_t> tensor_vec;
          for(auto i = padding.begin(); i != padding.end(); i++)
            tensor_vec.push_back(std::stoi(*i));
          metawarenn::Tensor reshape_tensor(pad_ip_name, dims, metawarenn::ElementType::element_type::int64_, tensor_vec);
          graph_->set_graph_initializers(reshape_tensor);
          graph_->initializer_names.insert(pad_ip_name);
          node_inputs[1] = pad_ip_name;
        }
        else if (node.GetOpName() == "sigmoid") {
          node_op_type = "Sigmoid";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "log") {
          node_op_type = "Log";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "tanh") {
          node_op_type = "Tanh";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if (node.GetOpName() == "nn.bias_add") {
          node_op_type = "Add";
          node_name = node_op_type + std::to_string(layer_count++);
        }
        else if(node.GetOpName() == "qnn.conv2d") {
          float ip_scale=0.0, w_scale=0.0;
          graph_->update_initializer_tensor_type(node_inputs[3], metawarenn::ElementType::element_type::uint8_);
          for(auto g_t : graph_->get_graph_initializers()) {
            if(g_t.get_name() == node_inputs[4]) { //Input Scale
              ip_scale = g_t.get_tensor<float>(g_t.get_type())[0];
            }
            else if(g_t.get_name() == node_inputs[5]) { //Weight Scale
              w_scale = g_t.get_tensor<float>(g_t.get_type())[0];
            }
          }

          node_op_type = "Conv";
          node_name = node_op_type + std::to_string(layer_count++);
          std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
          std::vector<std::string> pads = node.GetAttr<std::vector<std::string>>("padding");
          std::vector<std::string> dilations = node.GetAttr<std::vector<std::string>>("dilation");
          std::vector<std::string> kernel_size = node.GetAttr<std::vector<std::string>>("kernel_size");
          int64_t group = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

          if(tf_tvm_to_onnx && group > 1) {
            //https://github.com/apache/tvm/blob/26281792e92ae24ec7a14b11e8df8fbacf9c4882/python/tvm/relay/frontend/tflite.py#L2088
            //Assigns Input channels to group in tvm parser
            //To handle the depth multiplier values correctly
            for (auto g_t : graph_->get_graph_initializers()) {
              auto k_dims = g_t.get_dims();
              if(k_dims.size() == 4 && g_t.get_name() == node_inputs[1]) {
                std::vector<int> new_dims = {k_dims[0], k_dims[1], k_dims[2]/group, k_dims[3]*group};
                auto type = g_t.get_type();
                if(type == (int)metawarenn::ElementType::element_type::float_) {
                  std::vector<float> data = g_t.get_tensor<float>(type);
                  graph_->update_initializer_tensors(g_t.get_name(), new_dims, data, type);
                }
                else if(type == (int)metawarenn::ElementType::element_type::int32_ ||
                  type == (int)metawarenn::ElementType::element_type::int8_ ||
                  type == (int)metawarenn::ElementType::element_type::int16_ ||
                  type == (int)metawarenn::ElementType::element_type::uint8_ ||
                  type == (int)metawarenn::ElementType::element_type::uint16_) {
                  std::vector<int32_t> data = g_t.get_tensor<int32_t>(type);
                  graph_->update_initializer_tensors(g_t.get_name(), new_dims, data, type);
                  }
                else if(type == (int)metawarenn::ElementType::element_type::int64_) {
                  std::vector<int64_t> data = g_t.get_tensor<int64_t>(type);
                  graph_->update_initializer_tensors(g_t.get_name(), new_dims, data, type);
                }
                break;
              }
            }
          }

          std::string dequant_node_op_type = "DequantizeLinear";
          std::string dequant_node_name = dequant_node_op_type + "_" + node_inputs[1];
          std::vector<std::string> dequant_node_inputs;
          std::vector<std::string> dequant_node_outputs;
          std::vector<::metawarenn::Attribute> dequant_node_attributes;
          dequant_node_inputs.push_back(node_inputs[1]);
          dequant_node_inputs.push_back(node_inputs[5]);//Weight Scale
          dequant_node_inputs.push_back(node_inputs[3]);//Weight ZeroPoint
          dequant_node_outputs.push_back(dequant_node_name);
          CreateMWNNNode(dequant_node_name, dequant_node_op_type, dequant_node_attributes, dequant_node_inputs, dequant_node_outputs);
          node_inputs.erase(node_inputs.begin()+2, node_inputs.begin()+6);
          node_inputs[1] = dequant_node_outputs[0];//Replace with Dequant node output
          last_qdq_node = "dequant";
          auto weight_entry = node.GetInputs()[1];

          metawarenn::Attribute attr_dilate("dilations", std::vector<int64_t>({std::stoi(dilations[0]), std::stoi(dilations[1])}));
          node_attributes.emplace_back(attr_dilate);
          metawarenn::Attribute attr_group("group", group);
          node_attributes.emplace_back(attr_group);
          metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>({std::stoi(kernel_size[0]), std::stoi(kernel_size[1])}));
          node_attributes.emplace_back(attr_kernel_shape);
          metawarenn::Attribute attr_pad("pads", std::vector<int64_t>({std::stoi(pads[0]), std::stoi(pads[1]), std::stoi(pads[2]), std::stoi(pads[3])}));
          node_attributes.emplace_back(attr_pad);
          metawarenn::Attribute attr_stride("strides", std::vector<int64_t>({std::stoi(strides[0]), std::stoi(strides[1])}));
          node_attributes.emplace_back(attr_stride);

          if(id+2 < nodes_.size()) {
            for(int k = id+1; k <= id+2; k++) {
              const auto& bias_node = nodes_[k];
              if(bias_node.GetOpType() == "kernel" && bias_node.GetOpName() == "nn.bias_add") {
                //BiasNode Input Parsing
                auto bias_in_node = bias_node.GetInputs()[1];//index-0 --> Feature Tensor, index-1 Bias Values
                std::string ip_name = "node_" + std::to_string(bias_in_node.id_);

                std::string dequant_node_op_type = "DequantizeLinear";
                std::string dequant_node_name = dequant_node_op_type + "_" + ip_name;
                std::vector<std::string> dequant_node_inputs;
                std::vector<std::string> dequant_node_outputs;
                std::vector<::metawarenn::Attribute> dequant_node_attributes;

                std::string scale_name = ip_name + std::string("_scale");
                std::vector<float> tensor_vec_scale = {ip_scale*w_scale};
                ::metawarenn::Tensor scale_tensor(scale_name, std::vector<int>({tensor_vec_scale.size()}), ::metawarenn::ElementType::element_type::float_, tensor_vec_scale);
                graph_->set_graph_initializers(scale_tensor);
                graph_->initializer_names.insert(scale_name);

                std::string zp_name = ip_name + std::string("_zero_point");
                std::vector<int32_t> tensor_vec_zp = {0};
                ::metawarenn::Tensor zp_tensor(zp_name, std::vector<int>({tensor_vec_zp.size()}), ::metawarenn::ElementType::element_type::int32_, tensor_vec_zp);
                graph_->set_graph_initializers(zp_tensor);
                graph_->initializer_names.insert(zp_name);

                dequant_node_inputs.push_back(ip_name);
                dequant_node_inputs.push_back(scale_name);//Bias Scale
                dequant_node_inputs.push_back(zp_name);//Bias ZeroPoint
                dequant_node_outputs.push_back(dequant_node_name);
                CreateMWNNNode(dequant_node_name, dequant_node_op_type, dequant_node_attributes, dequant_node_inputs, dequant_node_outputs);
                last_qdq_node = "dequant";

                node_inputs.emplace_back(dequant_node_outputs[0]);
                merged_node_ids.insert(k);
                op_name = "node_" + std::to_string(bias_in_node.id_+1) + "_" + std::to_string(bias_in_node.index_);
                node_outputs[0] = op_name;
                prev_out_index = bias_in_node.id_+1;
              }
            }
          }
        }
        else if (node.GetOpName() ==  "qnn.requantize") {
          //https://github.com/apache/tvm/blob/26281792e92ae24ec7a14b11e8df8fbacf9c4882/python/tvm/relay/frontend/tflite.py#L2196
          graph_->update_initializer_tensor_type(node_inputs[4], metawarenn::ElementType::element_type::uint8_);//Update ZeroPoint Datatype
          CreateQDQNodes(node_inputs[0], node_outputs[0], node_inputs[3], node_inputs[4]);
          node_inputs.erase(node_inputs.begin()+1, node_inputs.begin()+5);
          continue;
        }
        else if (node.GetOpName() ==  "qnn.dequantize") {
          if(last_qdq_node == "quant") {
            node_op_type = "DequantizeLinear";
            node_name = node_op_type + std::to_string(layer_count++);
            last_qdq_node = "dequant";
          }
          else {
            quant_ip_mapper[node_outputs[0]] = node_inputs[0];
            continue;
          }
        }
        else if (node.GetOpName() ==  "qnn.quantize") {
          if(last_qdq_node == "dequant") {
            node_op_type = "QuantizeLinear";
            node_name = node_op_type + std::to_string(layer_count++);
            graph_->update_initializer_tensor_type(node_inputs[2], metawarenn::ElementType::element_type::uint8_);
            last_qdq_node = "quant";
          }
          else {
            quant_ip_mapper[node_outputs[0]] = node_inputs[0];
            continue;
          }
        }
        else if (node.GetOpName() ==  "qnn.add") {
          //https://github.com/apache/tvm/blob/26281792e92ae24ec7a14b11e8df8fbacf9c4882/python/tvm/relay/frontend/tflite.py#L1270
          node_op_type = "Add";
          node_name = node_op_type + std::to_string(layer_count++);
          graph_->update_initializer_tensor_type(node_inputs[7], metawarenn::ElementType::element_type::uint8_);

          std::string dequant_node_op_type = "DequantizeLinear";
          std::string dequant_node_name = dequant_node_op_type + "_" + node_outputs[0];
          CreateQDQNodes(node_outputs[0], dequant_node_name, node_inputs[6], node_inputs[7]);
          node_inputs.erase(node_inputs.begin()+2, node_inputs.begin()+8);
          quant_ip_mapper[node_outputs[0]] = dequant_node_name;
        }
        else if (node.GetOpName() == "qnn.concatenate") {
          //https://github.com/apache/tvm/blob/26281792e92ae24ec7a14b11e8df8fbacf9c4882/python/tvm/relay/frontend/tflite.py#L1082
          node_op_type = "Concat";
          node_name = node_op_type + std::to_string(layer_count++);
          auto axis = node.GetAttr<std::vector<std::string>>("axis");
          metawarenn::Attribute attr_axis;
          if(tf_tvm_to_onnx) //To handle the layout from HWC(TFLite) to CHW(ONNX)
            attr_axis = metawarenn::Attribute("axis", (int64_t)std::stoi(axis[0])-2);
          else
            attr_axis = metawarenn::Attribute("axis", (int64_t)std::stoi(axis[0]));
          node_attributes.emplace_back(attr_axis);

          int total_ips = node_inputs.size();
          std::string op_scale_name = node_inputs[total_ips-2];
          std::string op_zp_name = node_inputs[total_ips-1];
          int num_inputs = (total_ips - 2) / 3;
          graph_->update_initializer_tensor_type(node_inputs[total_ips-1], metawarenn::ElementType::element_type::uint8_);
          std::string dequant_node_op_type = "DequantizeLinear";
          std::string dequant_node_name = dequant_node_op_type + "_" + node_outputs[0];
          CreateQDQNodes(node_outputs[0], dequant_node_name, op_scale_name, op_zp_name);
          node_inputs.erase(node_inputs.begin()+num_inputs, node_inputs.begin()+total_ips);
          quant_ip_mapper[node_outputs[0]] = dequant_node_name;
        }
        else if (node.GetOpName() == "qnn.dense") {
          //https://github.com/apache/tvm/blob/26281792e92ae24ec7a14b11e8df8fbacf9c4882/python/tvm/relay/frontend/tflite.py#L1902

          float ip_scale=0.0, w_scale=0.0;
          graph_->update_initializer_tensor_type(node_inputs[3], metawarenn::ElementType::element_type::uint8_);
          for(auto g_t : graph_->get_graph_initializers()) {
            if(g_t.get_name() == node_inputs[4]) { //Input Scale
              ip_scale = g_t.get_tensor<float>(g_t.get_type())[0];
            }
            else if(g_t.get_name() == node_inputs[5]) { //Weight Scale
              w_scale = g_t.get_tensor<float>(g_t.get_type())[0];
            }
          }

          std::string dequant_node_op_type = "DequantizeLinear";
          std::string dequant_node_name = dequant_node_op_type + "_" + node_inputs[1];
          std::vector<std::string> dequant_node_inputs;
          std::vector<std::string> dequant_node_outputs;
          std::vector<::metawarenn::Attribute> dequant_node_attributes;
          dequant_node_inputs.push_back(node_inputs[1]);
          dequant_node_inputs.push_back(node_inputs[5]);//Weight Scale
          dequant_node_inputs.push_back(node_inputs[3]);//Weight ZeroPoint
          dequant_node_outputs.push_back(dequant_node_name);
          CreateMWNNNode(dequant_node_name, dequant_node_op_type, dequant_node_attributes, dequant_node_inputs, dequant_node_outputs);
          node_inputs.erase(node_inputs.begin()+2, node_inputs.begin()+6);
          node_inputs[1] = dequant_node_outputs[0];//Replace with Dequant node output
          last_qdq_node = "dequant";

          node_op_type = "Gemm";
          node_name = node_op_type + std::to_string(layer_count++);
          metawarenn::Attribute attr_transB("transB", (int64_t)1);//TODO - Do Check & Pass flags
          node_attributes.emplace_back(attr_transB);

          if(id+2 < nodes_.size()) {
            for(int k = id+1; k <= id+2; k++) {
              const auto& bias_node = nodes_[k];
              if(bias_node.GetOpType() == "kernel" && bias_node.GetOpName() == "add" || bias_node.GetOpName() == "nn.bias_add") {
                //BiasNode Input Parsing = (Gemm)(onnx) -> (Flatten + Dense + Add) in TVM
                auto bias_in_node = bias_node.GetInputs()[1];//index-0 --> Feature Tensor, index-1 Bias Values
                std::string ip_name = "node_" + std::to_string(bias_in_node.id_);

                std::string dequant_node_op_type = "DequantizeLinear";
                std::string dequant_node_name = dequant_node_op_type + "_" + ip_name;
                std::vector<std::string> dequant_node_inputs;
                std::vector<std::string> dequant_node_outputs;
                std::vector<::metawarenn::Attribute> dequant_node_attributes;

                std::string scale_name = ip_name + std::string("_scale");
                std::vector<float> tensor_vec_scale = {ip_scale*w_scale};
                ::metawarenn::Tensor scale_tensor(scale_name, std::vector<int>({tensor_vec_scale.size()}), ::metawarenn::ElementType::element_type::float_, tensor_vec_scale);
                graph_->set_graph_initializers(scale_tensor);
                graph_->initializer_names.insert(scale_name);

                std::string zp_name = ip_name + std::string("_zero_point");
                std::vector<int32_t> tensor_vec_zp = {0};
                ::metawarenn::Tensor zp_tensor(zp_name, std::vector<int>({tensor_vec_zp.size()}), ::metawarenn::ElementType::element_type::int32_, tensor_vec_zp);
                graph_->set_graph_initializers(zp_tensor);
                graph_->initializer_names.insert(zp_name);

                dequant_node_inputs.push_back(ip_name);
                dequant_node_inputs.push_back(scale_name);//Bias Scale
                dequant_node_inputs.push_back(zp_name);//Bias ZeroPoint
                dequant_node_outputs.push_back(dequant_node_name);
                CreateMWNNNode(dequant_node_name, dequant_node_op_type, dequant_node_attributes, dequant_node_inputs, dequant_node_outputs);
                last_qdq_node = "dequant";

                node_inputs.emplace_back(dequant_node_name);
                merged_node_ids.insert(k);
                op_name = "node_" + std::to_string(bias_in_node.id_+1) + "_" + std::to_string(bias_in_node.index_);
                node_outputs[0] = op_name;
                prev_out_index = bias_in_node.id_+1;
              }
            }
          }
        }
        else if (node.GetOpName() == "cast") {
          if (quant_model && !node.HasAttr("dtype")) {
            quant_ip_mapper[node_outputs[0]] = node_inputs[0];
            continue;
          }
          node_op_type = "Cast";
          node_name = node_op_type + std::to_string(layer_count++);
          int data_type = std::stoi(node.GetAttr<std::vector<std::string>>("dtype")[0]);
          //TODO - Handle for non-quantized model & map data type to onnx
          //metawarenn::Attribute attr_axis("to", std::stoi(axis[0]));
          //node_attributes.emplace_back(attr_axis);
        }
        else {
          std::cout << "\n Unsupported Op in MetaWareNN backend : " << node.GetOpName();
          exit(1);
        }
        CreateMWNNNode(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
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
    std::cout << "\n Graph outputs: ";
    std::string out_name;
    // TODO: Handle node output name difference from graph output id & combine checks
    // For multiple output nodes
    if(outputs_.size() > 1) {
      for (auto output : outputs_) {
        out_name = "node_" + std::to_string(output.id_) + "_" + std::to_string(output.index_);
        std::cout << "\n" << out_name << ", ";
        auto out_shape = nodes_[output.id_].GetOpShape();
        std::vector<int> dims;
        for(int m = 0; m < out_shape.size(); m++)
          for(int n = 0; n < out_shape[m].size(); n++) {
            dims.push_back(out_shape[m][n]);
          }
        metawarenn::Tensor m_op_tensor(out_name, m_type, dims);
        graph_->set_graph_op_tensor(m_op_tensor);
        graph_->set_graph_op_names(m_op_tensor.get_name());
      }
    }
    // For single output nodes
    else {
        if(static_cast<int>(dtypes[0].code) == kDLUInt) {
          if(last_qdq_node != "quant") {
            m_type = metawarenn::ElementType::element_type::uint8_;
            if(quant_ip_mapper.count(op_name) != 0)
              op_name = quant_ip_mapper[op_name];

            std::string quant_node_op_type = "QuantizeLinear";
            std::string quant_node_name = quant_node_op_type + "_" + op_name;
            std::vector<std::string> quant_node_inputs;
            std::vector<std::string> quant_node_outputs;
            std::vector<::metawarenn::Attribute> quant_node_attributes;

            quant_node_inputs.push_back(op_name);
            quant_node_inputs.push_back(quant_prev_scale);//Scale
            quant_node_inputs.push_back(quant_prev_zp);//ZeroPoint
            quant_node_outputs.push_back(quant_node_name);
            CreateMWNNNode(quant_node_name, quant_node_op_type, quant_node_attributes, quant_node_inputs, quant_node_outputs);
            op_name = quant_node_name;
            last_qdq_node = "quant";
          }
          else
           m_type = metawarenn::ElementType::element_type::uint8_;
           if(quant_ip_mapper.count(op_name) != 0)
             op_name = quant_ip_mapper[op_name];
        }
      metawarenn::Tensor m_op_tensor(op_name, m_type, dims);
      graph_->set_graph_op_tensor(m_op_tensor);
      graph_->set_graph_op_names(m_op_tensor.get_name());
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

void register_expand_dim(::metawarenn::optimizer::PassManager *manager, metawarenn::Tensor g_t) {
  for (auto g_n : graph_->get_graph_nodes()) {
    if(g_n.get_op_type() == "Mul" || g_n.get_op_type() == "Add") {
      for (auto n_ip : g_n.get_inputs()) {
        if(g_t.get_name() == n_ip) {
          std::cout << "\n Less Dimensiosna Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";
          ::metawarenn::optimizer::ExpandDimension ed(graph_, g_t);
          manager->register_pass(ed);
        }
      }
    }
  }
}

  void ApplyPasses() {
    ::metawarenn::optimizer::PassManager manager;
    auto node_list = graph_->get_graph_nodes();
    auto tf_flag = std::getenv("TF_TVM_TO_ONNX");
    bool tf_tvm_to_onnx = atoi(tf_flag);
    if(tf_tvm_to_onnx)
    {
      for (auto g_t : graph_->get_graph_initializers()) {
        if(g_t.get_dims().size() == 4) {
          std::cout << "\n ConvertLayout Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";
          ::metawarenn::optimizer::ConvertLayout cl(graph_, g_t, CHW_TO_HWC, HWC_TO_CHW, tf_tvm_to_onnx, true);
          manager.register_pass(cl);
        }
        else {
          register_expand_dim(&manager, g_t);
        }
      }
      for (auto g_t : graph_->get_graph_ip_tensor()) {
        if(g_t.get_dims().size() == 4) {
          std::cout << "\n Name : " << g_t.get_name();
          std::cout << "\t Dims : ";
          for (auto dim : g_t.get_dims())
            std::cout << dim << ",";
          ::metawarenn::optimizer::ConvertLayout cl(graph_, g_t, CHW_TO_HWC, HWC_TO_CHW, tf_tvm_to_onnx, false);
          manager.register_pass(cl);
        }
      }
    }
    else {
      for (auto g_t : graph_->get_graph_initializers()) {
        if(g_t.get_dims().size() < 4) {
          register_expand_dim(&manager, g_t);
        }
      }
    }
    /*for (int node_idx = 0; node_idx < graph_->get_graph_nodes().size(); node_idx++) {
      auto g_n = node_list[node_idx];
      if(g_n.get_op_type() == "Reshape") {
        ::metawarenn::optimizer::RemoveReshape rr(graph_, g_n);
        std::cout << "\n MetaWareNNCC : " << rr.get_name();
        manager.register_pass(rr);
      }
      else if(g_n.get_op_type() == "BatchNormalization") {
        ::metawarenn::optimizer::FuseBatchNorm fbn(graph_, g_n);
        std::cout << "\n MetaWareNNCC : " << fbn.get_name();
        manager.register_pass(fbn);
      }
      else if(g_n.get_op_type() == "Relu") {
        ::metawarenn::optimizer::FuseRelu fr(graph_, g_n);
        std::cout << "\n MetaWareNNCC : " << fr.get_name();
        manager.register_pass(fr);
      }
    }*/
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

  #if INVOKE_NNAC
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
      // TODO: template the get_tensor() call if NNAC flow to be used
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
  #endif

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
