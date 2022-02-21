import tensorflow as tf
import tvm
from tvm import relay
import tflite
import numpy as np
import os
import shutil
from PIL import Image
import onnxruntime

tvm_path = os.environ['FRAMEWORK_PATH']
op_dump_folder = tvm_path + '/metawarenn_inference/op_tflite_models'
if os.path.exists(op_dump_folder):
    shutil.rmtree(op_dump_folder)
os.makedirs(op_dump_folder)
op_file = open(op_dump_folder + "/validation_result.txt", 'w')

dtype = "float32"
model_dir = tvm_path + "/tflite_models/"
f = open("tflite_models.txt", "r")

for line in f:
  inp_shape = []
  model_name = line.strip()
  model_path = model_dir + model_name
  print("Model path: ", model_path)
  os.system('python tflite_inference.py ' + model_path)
op_file.close()
