#!/bin/sh

########### Executable Networks flow ##############
#set the path to tvm
export FRAMEWORK_PATH=/Path/to/tvm/
export METAWARENN_LIB_PATH=$FRAMEWORK_PATH"/src/runtime/contrib/metawarenn/metawarenn_lib/"
export EXEC_DUMPS_PATH=$FRAMEWORK_PATH"/EXEC_DUMPS/"

########### NNAC - EV binary generation flow ##############
#set the path to ARC directory
export ARC_PATH=/Path/to/ARC/
export NNAC_DUMPS_PATH=$FRAMEWORK_PATH"/NNAC_DUMPS/"