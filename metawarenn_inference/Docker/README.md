## Steps to use the docker setup to build and run the TVM
1. To create a docker container with Ubuntu 18.04 as base, run
     * `sudo bash Docker.sh`
2. Copy the shell script to docker folder
     * `cp /path/to/local/machine/tvm_deps.sh /path/to/docker/folder/root`
3. Run the shell script to install the TVM related dependencies
     * `cd /path/to/docker/folder/root`
     * `bash tvm_deps.sh`
       - [Note]: The above commands will install all TVM related dependencies including onnx, tflite, numpy, cmake, protobuf, etc., and clones the TVM repository. It will take around an hour to finish the installation.
4. Update necessary paths
    * To create ONNX Proto from MWNNGraph [Default flow]
        1. By default, `INFERENCE_ENGINE` flag is set to zero in metawarenn_lib/metawarenn_common.h, which will create ONNXProto directly from MWNNGraph and store it in inference/op_onnx_models
        2. Enable `INFERENCE_ENGINE` flag in metawarenn_lib/metawarenn_common.h, to convert MWNNGraph to ExecutableGraph and then create Inference Engine & Execution Context and finally creates the output ONNXProto in inference/op_onnx_models
    * To Load MetaWareNN Executable Graph in Shared Memory [ OutDated & Optional ]
       1. Set the absolute path to onnxruntime in onnxruntime/onnxruntime/core/providers/metawarenn/inference/env.sh line no: 5
           [Note] : Executable MetaWareNN Binary Graph will get stored in `/path/to/onnxruntime/EXEC_DUMPS`
    * To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file - [ OutDated & Optional ]
       1. Enable the `INVOKE_NNAC` macro in onnxruntime/onnxruntime/core/providers/metawarenn/metawarenn.h line no: 17
       2. Update onnxruntime/onnxruntime/core/providers/metawarenn/inference/env.sh file
           i. Set the absolute path to ARC/ directory in line no: 11
           ii. Set the absolute path to cnn_models/ directory in line no: 12
           * [Note] : Generated EV Binary file for MetaWareNN SubGraph will store in evgencnn/scripts and all intermediate files will get stored in `/path/to/onnxruntime/NNAC_DUMPS` folder
5. Set the Environmental Variables for Build & Inference
      * `source /path/to/docker/tvm/metawarenn_inference/env.sh`
6. To build the TVM
      * `cd /path/to/tvm/`
      * `mkdir build`
      * `cp cmake/config.cmake build`
      * `cd build`
      * `cmake ..`
      * `make -j4`
      * `cd ../python`
      * `python3 setup.py install --user`


### To Run the Inference Script 
* Download the MobileNet-V2 model using the zip file from egnyte link - https://multicorewareinc.egnyte.com/dl/2JAUNXlGg0 and unzip the same
* `scp uname@ip_address:/path/to/local/machine/mobilenetv2-7.onnx /path/to/docker/folder/`
   1. `cd $FRAMEWORK_PATH/metawarenn_inference`
   2. `python3 mwnn_inference.py /path/to/mobilenetv2-7.onnx`

### To Generate the ONNXProto from multiple Float ONNX models & Verify
   1. cd /path/to/tvm/metawarenn_inference
   2. source env.sh
   3. sh download_onnx_models.sh or Download from Egnyte link - https://multicorewareinc.egnyte.com/fl/1hIpgufAHp # (For First time) - Creates onnx_models directory inside tvm/ & downloads models into it
   4. python test_regression_onnx.py # Creates a `op_onnx_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original & generated onnx model

### To Generate the ONNXProto from multiple Float TFLite models & Verify
   1. cd /path/to/tvm/metawarenn_inference
   2. Set `TF_TVM_TO_ONNX` macro to 1 in `/path/to/tvm/metawarenn_inference/env.sh` line no: 7
   3. source env.sh
   4. sh download_tflite_models.sh or Download float TFLite Models from Egnyte Link - https://multicorewareinc.egnyte.com/fl/k0wHUistvX # (For First time) - Creates tflite_models directory inside tvm/ & downloads models into it
   5. python test_regression_tflite.py # Creates a `op_tflite_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original tflite & generated onnx model

### To Generate the ONNXProto from multiple Quantized TFLite models & Verify
   1. cd /path/to/tvm/metawarenn_inference
   2. Set `TF_TVM_TO_ONNX` macro to 1 in `/path/to/tvm/metawarenn_inference/env.sh` line no: 7
   3. source env.sh
   4. sh download_tflite_quantized_models.sh or Download Quantized TFLite Models from Egnyte Link - https://multicorewareinc.egnyte.com/fl/7uaWWI9PNi # (For First time) - Creates tflite_quantized_models directory inside tvm/ & downloads models into it
   5. python test_regression_quantized_tflite.py # Creates a `op_tflite_quantized_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original tflite & generated onnx model
```
[Note] - Enable `FLOAT_OUTPUT` flag in line no: 12 of `test_regression_quantized_tflite.py` file to get the final outputs in float numbers, by defualt it is uint8
