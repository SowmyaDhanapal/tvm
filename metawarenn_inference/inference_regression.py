import onnx
import tvm
from tvm import relay
import numpy as np
import os
from typing import List

dtype = "float32"
tvm_path = os.environ['FRAMEWORK_PATH']
model_dir = tvm_path + "/onnx_models/"
f = open("models.txt", "r")

for line in f:
  inp_shape = []
  model_name = line.strip()
  model_path = model_dir + model_name
  print("Model path: ", model_path)
  if(os.path.exists(model_path)):
    onnx_model = onnx.load(model_dir + model_name)
  else:
    print("Please check the model path")
    exit(1)

  input_all = [node.name for node in onnx_model.graph.input]
  input_initializer =  [node.name for node in onnx_model.graph.initializer]
  net_feed_input = list(set(input_all)  - set(input_initializer))
  for node in onnx_model.graph.input:
    if node.name == net_feed_input[0]:
      input_node = node
      break

  shape = input_node.type.tensor_type.shape
  for dim in shape.dim:
      if (dim.HasField("dim_value")):
          inp_shape.append(dim.dim_value)

  # To handle unknown batch size
  if len(inp_shape) < 4:
    inp_shape.insert(0, 1)
  print("Input shape: ", inp_shape)

  mod, params = relay.frontend.from_onnx(onnx_model,shape={net_feed_input[0]: inp_shape}, dtype=dtype, freeze_params=True)

  print("Mod : ", type(mod)) #Mod :  <class 'tvm.ir.module.IRModule'>
  print("Params : ", type(params)) #Params :  <class 'dict'>
  print("=======================================================================")

  from tvm.relay.op.contrib.metawarenn import partition_for_metawarenn
  mod = partition_for_metawarenn(mod, params)
  print("Mod : ", type(mod)) #Mod :  <class 'tvm.ir.module.IRModule'>
  print("=======================================================================")

  if not tvm.get_global_func("relay.ext.metawarenn"):
    print("MetaWareNN Backend Function not exists!!!")
  else:
    print("MetaWareNN Backend Function exists!!!")

  print("============================ In Build =================================")
  #In Subgraphs case, export_library() successfully exports the .so file with target "llvm"  
  target = "llvm"
  with tvm.transform.PassContext():
      lib = relay.build(mod, target=target, params=params)
      print("lib: " , lib) #<tvm.relay.backend.executor_factory.GraphExecutorFactoryModule object at 0x7f768c01be80>
  print("=======================================================================")

  # load the module into memory
  from tvm.contrib import graph_executor
  print("======================== In load_module ===============================")
  print("lib[default]: " , lib["default"])
  print("======================== In GraphModule ===============================")
  #Loading lib directly avoiding save & load of .so file & fix for nodes mix up between one model to next
  module = graph_executor.GraphModule(lib["default"](tvm.cpu())) #<tvm.runtime.packed_func.PackedFunc object at 0x7fe592053db8>
  print("module : ", type(module)) #module :  <class 'tvm.contrib.graph_executor.GraphModule'>

  inference_inp_shape = (1, 3, 224, 224)
  input_data = np.random.uniform(0, 1, inference_inp_shape).astype(dtype)
  print("========================== In Run =====================================")
  module.run(data=input_data)