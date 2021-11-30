#!/bin/sh

########### Executable Networks flow ##############
#set the path to tvm
export FRAMEWORK_PATH=/Path/to/tvm/
export PYTHONPATH=$FRAMEWORK_PATH/python:${PYTHONPATH}
export TF_TVM_TO_ONNX=0
export METAWARENN_LIB_PATH=$FRAMEWORK_PATH"/src/runtime/contrib/metawarenn/metawarenn_lib/"
export EXEC_DUMPS_PATH=$FRAMEWORK_PATH"/EXEC_DUMPS/"

########### NNAC - EV binary generation flow ##############
#set the path to ARC directory
export ARC_PATH=/Path/to/ARC/
export EV_CNNMODELS_HOME=/Path/to/cnn_models/
export NNAC_DUMPS_PATH=$FRAMEWORK_PATH"/NNAC_DUMPS/"
