import tvm
from tvm import relay
import tflite
import numpy as np
import os
from typing import List

dtype = "float32"
tvm_path = os.environ['FRAMEWORK_PATH']
model_dir = tvm_path + "/tflite_models/"
f = open("tflite_models.txt", "r")

for line in f:
  inp_shape = []
  model_name = line.strip()
  model_path = model_dir + model_name
  print("Model path: ", model_path)
  if(os.path.exists(model_path)):
    tflite_model_buf = open(model_path, "rb").read()
    # Get TFLite model from buffer
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
  else:
    print("Please check the model path")
    exit(1)

  mod, params = relay.frontend.from_tflite(tflite_model)

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