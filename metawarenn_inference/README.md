## Build TVM MetaWareNN Backend

### Install Python Virtual Environment
   1. virtualenv --python=/usr/bin/python3.6 ./tvm_env
   2. source ./tvm_env/bin/activate

### Install TVM & Build along with dependencies
   1. `git clone --recursive https://github.com/SowmyaDhanapal/tvm.git tvm`
   2. cd tvm
   3. git checkout metawarenn_dev
   4. git submodule sync
   5. git submodule update
   6. sudo apt-get update
   7. sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
   8. mkdir build
   9. cp cmake/config.cmake build
   10. cd build
   11. cmake ..
   12. make -j4
   13. Install the following python packages to meet the dependencies,
       1. pip3 install numpy decorator attrs
       2. pip3 install tornado
       3. pip3 install onnx
       4. pip3 install psutil xgboost cloudpickle
   14. cd tvm/python
   15. python setup.py install --user

### To Run the Inference Script 
   1. export TVM_HOME=/path/to/tvm
   2. export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
   3. cd /path/to/tvm/metawarenn_inference
   4. Update the ONNX model path in line no: 9
   5. python mwnn_inference.py
