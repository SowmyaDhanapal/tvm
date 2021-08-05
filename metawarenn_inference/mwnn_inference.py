import onnx
import tvm
from tvm import relay
import numpy as np

dtype = "float32"
input_shape = (1, 3, 224, 224)
onnx_model = onnx.load("/path/to/mobilenetv2-7.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, shape={'data': input_shape}, dtype=dtype)
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
    print(lib) #<tvm.relay.backend.executor_factory.GraphExecutorFactoryModule object at 0x7f768c01be80>
    lib.export_library('mwnn_compiled.so')
print("=======================================================================")

# load the module into memory
from tvm.contrib import graph_executor
print("======================== In load_module ===============================")
loaded_lib = tvm.runtime.load_module('mwnn_compiled.so')
print("loaded_lib : ", type(loaded_lib)) #loaded_lib :  <class 'tvm.runtime.module.Module'>
print("======================== In GraphModule ===============================")
module = graph_executor.GraphModule(loaded_lib["default"](tvm.cpu()))
print("module : ", type(module)) #module :  <class 'tvm.contrib.graph_executor.GraphModule'>
input_data = np.random.uniform(0, 1, input_shape).astype(dtype)
print("========================== In Run =====================================")
module.run(data=input_data)
