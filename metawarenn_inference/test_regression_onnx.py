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

dtype = "float32"
model_dir = tvm_path + "/onnx_models/"
f = open("models.txt", "r")

for line in f:
  input_nodes = []
  model_name = line.strip()
  model_path = model_dir + model_name
  os.system('python onnx_inference.py ' + model_path)

op_file.close()
