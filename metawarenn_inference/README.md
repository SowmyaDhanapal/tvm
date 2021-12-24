## Build TVM MetaWareNN Backend

## Use Docker Setup to build and run the TVM
##### Check on the [tvm/metawarenn_inference/Docker/README.md](https://github.com/SowmyaDhanapal/tvm/tree/metawarenn_dev/metawarenn_inference/Docker/README.md)

## No Docker Process
### Get TVM

### Prerequisites
#### Clone the TVM & Dependencies
   ### Initial Setup
      1. `git clone --recursive https://github.com/SowmyaDhanapal/tvm.git tvm`
      2. cd tvm
      3. git checkout metawarenn_dev
      4. git submodule sync
      5. git submodule update
      6. In case if tvm is cloned without metawarenn_lib submodule, use below commands to pull MetaWareNN Library Submodule for the first time
         * `git pull`
         * `git submodule update --init --recursive`
         *  Move to metawarenn_lib submodule and checkout to metawarenn_dev branch
            a. `cd tvm/src/runtime/contrib/metawarenn/metawarenn_lib`
            b. `git checkout metawarenn_dev`
   ### Using Existing Setup to pull submodule changes [Docker / Non-Docker]
      1. `cd tvm`
      2. `git pull`
      3. `cd tvm/src/runtime/contrib/metawarenn/metawarenn_lib`
      4. `git checkout metawarenn_dev`
      5. `git pull`
   ### Install Dependent System Libraries
      1. sudo apt-get update
      2. sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
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
      export CPLUS_INCLUDE_PATH=install_protobuf_folder/include:${CPLUS_INCLUDE_PATH}
```

#### Install dependent python packages
   1. pip3 install numpy decorator attrs
   2. pip3 install tornado
   3. pip3 install onnx
   4. pip3 install psutil xgboost cloudpickle
   5. pip3 install tflite==2.3.0

### Modifications to make before build
#### To Load MetaWareNN Executable Graph in Shared Memory [Default flow]
  ```
   1. Set path to tvm in tvm/metawarenn_inference/env.sh line no: 5
  ```
#### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file
  ```
   1. Update the "tvm/src/runtime/contrib/metawarenn/metawarenn_json_runtime.cc" file as follows:
      i. Set the INVOKE_NNAC macro to 1 in line no: 47
   2. Set path to ARC/ directory in tvm/metawarenn_inference/env.sh line no: 11
   3. Set path to EV_CNNMODELS_HOME/ directory in tvm/metawarenn_inference/env.sh line no: 12
   [Note] : Generated EV Binary file for MetaWareNN SubGraph will be stored in evgencnn/scripts folder and all intermediate files will get stored in `/path/to/tvm/NNAC_DUMPS` folder
  ```
#### To use metawarenn_lib as shared library
   1. Rename tvm/cmake/modules/contrib/MetaWareNN.cmake to MetaWareNN_original.cmake
      `mv tvm/cmake/modules/contrib/MetaWareNN.cmake tvm/cmake/modules/contrib/MetaWareNN_original.cmake`
   2. Rename tvm/cmake/modules/contrib/MetaWareNN_shared_lib.cmake to MetaWareNN.cmake
      `mv tvm/cmake/modules/contrib/MetaWareNN_shared_lib.cmake tvm/cmake/modules/contrib/MetaWareNN.cmake`
   3. Download the metawarenn shared library from egnyte link https://multicorewareinc.egnyte.com/dl/n31afFTwP9 and place it in `tvm/src/runtime/contrib/metawarenn/metawarenn_lib/lib`
   4. Also download the dependent protobuf library from egnyte link https://multicorewareinc.egnyte.com/dl/kpRzPTSFdx and place it in `tvm/src/runtime/contrib/metawarenn/metawarenn_lib/lib`

### Steps to build
   1. mkdir build
   2. cp cmake/config.cmake build
   3. cd build
   4. cmake ..
   5. make -j4
   6. cd tvm/python
   7. python3 setup.py install --user

### To Run the Inference Script 
   1. cd /path/to/tvm/metawarenn_inference
   2. source env.sh
   3. Update the ONNX model path in line no: 8
   4. python mwnn_inference.py

### To Run the Inference for multiple ONNX models
   1. cd /path/to/tvm/metawarenn_inference
   2. source env.sh
   3. sh download_onnx_models.sh # Creates onnx_models directory inside tvm/ & downloads models into it
   4. python inference_regression.py

### To Run the Inference for multiple TFLite models
   1. cd /path/to/tvm/metawarenn_inference
   2. source env.sh
   3. sh download_tflite_models.sh # Creates tflite_models directory inside tvm/ & downloads models into it
   4. python inference_regression_tflite.py

### To Generate the ONNXProto from multiple ONNX models & Verify
   1. cd /path/to/tvm/metawarenn_inference
   2. source env.sh
   3. sh download_onnx_models.sh # (For First time) - Creates onnx_models directory inside tvm/ & downloads models into it
   4. python test_regression_onnx.py # Creates a `op_onnx_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original & generated onnx model

### To Generate the ONNXProto from multiple TFLite models & Verify
   1. cd /path/to/tvm/metawarenn_inference
   2. Set `TF_TVM_TO_ONNX` macro to 1 in `/path/to/tvm/metawarenn_inference/env.sh` line no: 7
   3. source env.sh
   4. sh download_tflite_models.sh # (For First time) - Creates tflite_models directory inside tvm/ & downloads models into it
   5. python test_regression_tflite.py # Creates a `op_tflite_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original tflite & generated onnx model

### To Generate the ONNXProto from multiple Quantized TFLite models & Verify
   1. cd /path/to/tvm/metawarenn_inference
   2. Set `TF_TVM_TO_ONNX` macro to 1 in `/path/to/tvm/metawarenn_inference/env.sh` line no: 7
   3. source env.sh
   4. sh download_tflite_quantized_models.sh # (For First time) - Creates tflite_quantized_models directory inside tvm/ & downloads models into it
   5. python test_regression_quantized_tflite.py # Creates a `op_tflite_quantized_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original tflite & generated onnx model
[Note] - Enable `FLOAT_OUTPUT` flag in line no: 12 of `test_regression_quantized_tflite.py` file to get the final outputs in float numbers, by defualt it is uint8
