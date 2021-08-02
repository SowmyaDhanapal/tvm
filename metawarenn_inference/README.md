## Build TVM MetaWareNN Backend

### Prerequisites
#### Clone the TVM & Dependencies
   1. `git clone --recursive https://github.com/SowmyaDhanapal/tvm.git tvm`
   2. cd tvm
   3. git checkout metawarenn_dev
   4. git submodule sync
   5. git submodule update
   6. sudo apt-get update
   7. sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
#### Install Python Virtual Environment
   1. virtualenv --python=/usr/bin/python3.6 ./tvm_env
   2. source ./tvm_env/bin/activate
#### Protobuf library dependencies
   1. Requird Protobuf Version - 3.11.3. Check with the following command:  
      `protoc --version`  
      + Steps to install Protobuf 3.11.3:  
```
      wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-all-3.11.3.tar.gz
      tar -xf protobuf-all-3.11.3.tar.gz
      cd protobuf-3.11.3
      ./configure [--prefix=install_protobuf_folder]
      make
      make check
      sudo make install
      cd ./python
      python3 setup.py build
      python3 setup.py test
      sudo python3 setup.py install
      sudo ldconfig
      # if not installed with sudo  
      export PATH=install_protobuf_folder/bin:${PATH}  
      export LD_LIBRARY_PATH=install_protobuf_folder/lib:${LD_LIBRARY_PATH}  
```
    2. Download protobuf library version 3.11.3 from the egnyte link https://multicorewareinc.egnyte.com/dl/FjljPlgjlI  
    3. Unzip and move the "libprotobuf.so" to "/path/to/tvm/metawarenn_inference"  

### Modifications to make before build
#### To Load MetaWareNN Executable Graph in Shared Memory [Default flow]
   1. Update the "tvm/src/runtime/contrib/metawarenn/metawarenn_lib/executable_network/metawarenn_executable_graph.cc" with path to store the MWNNExecutableNetwork.bin in line no: 401 and 414  
   2. Update the "tvm/src/runtime/contrib/metawarenn/metawarenn_lib/mwnn_inference_api/mwnn_inference_api.cc" file with saved file path of MWNNExecutableNetwork.bin in line no: 51  
#### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file
   1. Update the "tvm/src/runtime/contrib/metawarenn/metawarenn_json_runtime.cc" file as follows:  
      i. Set the INVOKE_NNAC macro to 1 in line no: 44  
      ii. Set the path to store the MWNN file dumps in line no: 310  
      iii. Set the path to tvm in line no: 319  
   2. Update the "tvm/src/runtime/contrib/metawarenn/metawarenn_lib/mwnnconvert/mwnn_convert.sh" file as follows:  
      i. Set the $EV_CNNMODELS_HOME path in line no: 3  
      ii. Set the absolute path for ARC/cnn_tools/setup.sh file in line no: 4  
      iii. Update the path to tvm with MWNN support in line no: 9 and line no: 20  
      iv. Update the path to evgencnn executable in line no: 10  
      v. Update the Imagenet images path in line no: 18  
   [Note] : Generated EV Binary file for MetaWareNN SubGraph will be stored in evgencnn/scripts folder.  

### Steps to build
   1. mkdir build
   2. cp cmake/config.cmake build
   3. cd build
   4. cmake ..
   5. make -j4
   6. Install the following python packages to meet the dependencies,
       1. pip3 install numpy decorator attrs
       2. pip3 install tornado
       3. pip3 install onnx
       4. pip3 install psutil xgboost cloudpickle
   7. cd tvm/python
   8. python setup.py install --user

### To Run the Inference Script 
   1. export TVM_HOME=/path/to/tvm
   2. export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
   3. cd /path/to/tvm/metawarenn_inference
   4. Update the ONNX model path in line no: 9
   5. python mwnn_inference.py
