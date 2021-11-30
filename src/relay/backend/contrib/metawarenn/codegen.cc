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
 * \file src/relay/backend/contrib/metawarenn/codegen.cc
 * \brief Implementation of MetaWareNN codegen APIs.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>

#include "../../utils.h"
#include "../../../../runtime/contrib/json/json_node.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class MetaWareNNJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  MetaWareNNJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    Expr expr = GetRef<Expr>(cn);
    std::string name;
    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
    }
    else {
      LOG(FATAL) << "MetaWareNN JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }

    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    //std::cout << "\n In VisitExpr_ : " << name;
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    if (name == "nn.pad") {
      SetPadNodeAttribute(node, cn);
    }
    else {
      SetCallNodeAttribute(node, cn);
    }
    return AddNode(node, GetRef<Expr>(cn));
  }

  void SetPadNodeAttribute(std::shared_ptr<JSONGraphNode> node, const CallNode* cn) {
    const auto* pad_attr = cn->attrs.as<PadAttrs>();
    ICHECK(pad_attr);
    auto p = pad_attr->pad_width;
    const int dim_h = p.size();
    const int dim_w = p[0].size();
    std::vector<std::string> padding;
    std::vector<std::string> NDimValue(2);
    std::vector<std::string> CDimValue(2);
    std::vector<std::string> HDimValue(2);
    std::vector<std::string> WDimValue(2);

    auto tf_flag = std::getenv("TF_TVM_TO_ONNX");
    bool tf_tvm_to_onnx = atoi(tf_flag);

    if(tf_tvm_to_onnx) {
    NDimValue = {std::to_string(p[0][0].as<IntImmNode>()->value), std::to_string(p[0][1].as<IntImmNode>()->value)};
    HDimValue = {std::to_string(p[1][0].as<IntImmNode>()->value), std::to_string(p[1][1].as<IntImmNode>()->value)};
    WDimValue = {std::to_string(p[2][0].as<IntImmNode>()->value), std::to_string(p[2][1].as<IntImmNode>()->value)};
    CDimValue = {std::to_string(p[3][0].as<IntImmNode>()->value), std::to_string(p[3][1].as<IntImmNode>()->value)};
    }
    else {
    NDimValue = {std::to_string(p[0][0].as<IntImmNode>()->value), std::to_string(p[0][1].as<IntImmNode>()->value)};
    CDimValue = {std::to_string(p[1][0].as<IntImmNode>()->value), std::to_string(p[1][1].as<IntImmNode>()->value)};
    HDimValue = {std::to_string(p[2][0].as<IntImmNode>()->value), std::to_string(p[2][1].as<IntImmNode>()->value)};
    WDimValue = {std::to_string(p[3][0].as<IntImmNode>()->value), std::to_string(p[3][1].as<IntImmNode>()->value)};
    }
    //Pad layer NHWC --> NCHW
    //TFLite Format (NStart, NEnd, HStart, HEnd, WStart, WEnd, CStart, CEnd) (0, 1, 2, 3, 4, 5, 6, 7)
    //ONNX Format   (NStart, CStart, HStart, WStart, NEnd, CEnd, HEnd, WEnd) (0, 6, 2, 4, 1, 7, 3, 5)

    padding.emplace_back(NDimValue[0]);
    padding.emplace_back(CDimValue[0]);
    padding.emplace_back(HDimValue[0]);
    padding.emplace_back(WDimValue[0]);
    padding.emplace_back(NDimValue[1]);
    padding.emplace_back(CDimValue[1]);
    padding.emplace_back(HDimValue[1]);
    padding.emplace_back(WDimValue[1]);

    std::vector<dmlc::any> padding_attr;
    padding_attr.emplace_back(padding);
    node->SetAttr("padding", padding_attr);
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */

runtime::Module MetaWareNNCompiler(const ObjectRef& ref) {
  std::cout << "\n In MetaWareNNCompiler !!!";
  CHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  MetaWareNNJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();
  const auto* pf = runtime::Registry::Get("runtime.MetaWareNNJSONRuntimeCreate");
  CHECK(pf != nullptr) << "Cannot find MetaWareNN JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.metawarenn").set_body_typed(MetaWareNNCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
