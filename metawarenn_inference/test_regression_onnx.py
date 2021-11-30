import os
import shutil
import onnx
import tvm
from tvm import relay
import numpy as np
from PIL import Image
import onnxruntime

tvm_path = os.environ['FRAMEWORK_PATH']
op_dump_folder = tvm_path + '/metawarenn_inference/op_onnx_models'
if os.path.exists(op_dump_folder):
    shutil.rmtree(op_dump_folder)
os.makedirs(op_dump_folder)
op_file = open(op_dump_folder + "/validation_result.txt", 'w')

img = Image.open("kitten.jpg") 
img = img.resize((224, 224))

# add N dim
input_data = np.expand_dims(img, axis=0)
input_data = (np.float32(input_data) - 124) / 120
print(np.shape(input_data))

dtype = "float32"
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
  if(model_name == "mnist-7.onnx"):
    input_data = np.random.random_sample((1,1,28,28)).astype(np.float32)
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
  input_data_gen = np.random.uniform(0, 1, inference_inp_shape).astype(dtype)
  print("========================== In Run =====================================")
  module.run(data=input_data_gen)
  gen_model_name = op_dump_folder + "/model_" + model_name
  os.rename("model.onnx", gen_model_name)

  if(model_name == "efficientnet-lite4-11.onnx" or model_name == "mnist-7.onnx"):
    new_data = input_data
  else:
    new_data = np.rollaxis(input_data, 3, 1)
  print(np.shape(new_data))

  #======================================================================================================
  session = onnxruntime.InferenceSession(model_path, None) #Updated model name
  input_name = session.get_inputs()[0].name
  output_name = []
  for out in session.get_outputs():
      output_name.append(out.name)
  print('input_name :', input_name)
  print('output_name :', output_name)
  result  = session.run(output_name, {input_name: new_data}) #Layer dump name same as above
  result_arr = np.array(result)
  flat_result = result_arr.flatten()
  flat_result[::-1].sort()
  #=======================================================================================================
  session_mwnn = onnxruntime.InferenceSession(gen_model_name, None)  #Updated model name
  input_name_mwnn = session_mwnn.get_inputs()[0].name
  output_name_mwnn = []
  for out in session_mwnn.get_outputs():
      output_name_mwnn.append(out.name)
  print('input_name_mwnn :', input_name_mwnn)
  print('output_name_mwnn :', output_name_mwnn)
  result_mwnn = session_mwnn.run(output_name_mwnn, {input_name_mwnn: new_data}) #layer name

  result_arr_mwnn = np.array(result_mwnn)
  flat_result_mwnn = result_arr_mwnn.flatten()
  flat_result_mwnn[::-1].sort()
  op_file.write("\n=====================================================================================================\n")
  op_file.write(model_path)
  for idx in range(0, 5):
    op_file.write("\nDefault : " + str(flat_result[idx]) + "      New : " + str(flat_result_mwnn[idx]))
    error = abs(flat_result[idx]-flat_result_mwnn[idx])
    if(error > 0.001):
      op_file.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MISMATCH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
op_file.close()
