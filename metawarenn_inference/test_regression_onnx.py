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
  input_nodes = []
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
  inp_shape_list = []
  input_shape_dict = dict()

  for node in onnx_model.graph.input:
    if node.name in net_feed_input:
      input_nodes.append(node)

  for inp_node in input_nodes:
    print("input node: ", inp_node.name)
    shape = inp_node.type.tensor_type.shape
    unknown = False
    inp_shape = []
    for dim in shape.dim:
      if (dim.HasField("dim_value")):
          print (dim.dim_value, end=", ")  # known dimension
          inp_shape.append(dim.dim_value)
      elif (dim.HasField("dim_param")):
          print (dim.dim_param, end=", ")  # unknown dimension with symbolic name
          if dim.dim_param.count("unk"):
            print("unknown exist - ", dim.dim_param)
            unknown = True
      else:
          print("unknown exist - ?")
          unknown = True

    # To handle unknown batch size
    if len(inp_shape) < 4:
      inp_shape.insert(0, 1)

    if unknown and len(shape.dim)  == 4 :
      if(model_name == "yolov3-10.onnx"):
        inp_shape = [1,3,608,608]
      if(model_name == "tiny-yolov3-11.onnx"):
        inp_shape = [1,3,416,416]
    inp_shape_list.append(inp_shape)
    input_shape_dict[inp_node.name] = inp_shape

  print("input_shape_list: ", inp_shape_list)
  print("input_shape_dict: ", input_shape_dict)
  mod, params = relay.frontend.from_onnx(onnx_model,shape=input_shape_dict, dtype=dtype, freeze_params=True)
  print("input shape: ", inp_shape_list[0])

  input_data = np.random.random_sample((inp_shape_list[0])).astype(np.float32)
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

  #======================================================================================================
  session = onnxruntime.InferenceSession(model_path, None) #Updated model name
  input_name = session.get_inputs()[0].name
  output_name = []
  for out in session.get_outputs():
      output_name.append(out.name)
  print('input_name :', input_name)
  print('output_name :', output_name)
  result  = session.run(output_name, {input_name: input_data}) #Layer dump name same as above
  flat_result = []
  if len(result) > 1:
    for res in result:
      flat_result.append(np.array(res).flatten())
  else:
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
  result_mwnn = session_mwnn.run(output_name_mwnn, {input_name_mwnn: input_data}) #layer name
  flat_result_mwnn = []
  op_file.write("\n=====================================================================================================\n")
  op_file.write(model_path)
  if len(output_name_mwnn) > 1:
    for res_mwnn in result_mwnn:
      flat_result_mwnn.append(np.array(res_mwnn).flatten())
    for idx in range(0, len(flat_result_mwnn)):
      op_file.write("\nDefault : " + str(flat_result[idx][0]) + "      New : " + str(flat_result_mwnn[idx][0]))
      error = abs(flat_result[idx][0]-flat_result_mwnn[idx][0])
      if(error > 0.001):
        op_file.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MISMATCH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
  else:
    result_arr_mwnn = np.array(result_mwnn)
    flat_result_mwnn = result_arr_mwnn.flatten()
    flat_result_mwnn[::-1].sort()
    for idx in range(0, 5):
      op_file.write("\nDefault : " + str(flat_result[idx]) + "      New : " + str(flat_result_mwnn[idx]))
      error = abs(flat_result[idx]-flat_result_mwnn[idx])
      if(error > 0.001):
        op_file.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MISMATCH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
op_file.close()
