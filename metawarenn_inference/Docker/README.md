## Steps to use the docker setup to build and run the TVM
1. To create a docker container with Ubuntu 18.04 as base, run
        * `sudo bash Docker.sh`
2. Copy the shell script to docker folder
        * `cp /path/to/local/machine/tvm_deps.sh /path/to/docker/folder/root`
3. Run the shell script to install the TVM related dependencies
        * `cd /path/to/docker/folder/root`
        * `bash tvm_deps.sh`
        [Note]: The above commands will install all TVM related dependencies including onnx, tflite, numpy, cmake, protobuf, etc., and clones the TVM repository. It will take around an hour to finish the installation.
4. Update necessary paths
        * To Load MetaWareNN Executable Graph in Shared Memory [Default flow]
            1. Set path to tvm in metawarenn_inference/env.sh line no: 5
        * To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file
            1. Update the "tvm/src/runtime/contrib/metawarenn/metawarenn_json_runtime.cc" file as follows:
                 i. Set the INVOKE_NNAC macro to 1 in line no: 44
            2. Set path to ARC/ directory in tvm/metawarenn_inference/env.sh line no: 11
            3. Set path to EV_CNNMODELS_HOME/ directory in tvm/metawarenn_inference/env.sh line no: 12
            [Note] : Generated EV Binary file for MetaWareNN SubGraph will be stored in evgencnn/scripts folder and all intermediate files will get stored in `/path/to/tvm/NNAC_DUMPS` folder
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
   2. `Update the docker path of mobilenetv2-7.onnx in script mwnn_inference.py in line no:8`
   3. `python3 mwnn_inference.py`

### To Run the Inference for Multiple ONNX models
   1. `cd $FRAMEWORK_PATH/metawarenn_inference`
   2. `sh download_onnx_models.sh` # Creates onnx_models directory inside tvm/ & downloads models into it
   3. `python3 inference_regression.py`

### To Run the Inference for Multiple TFLite models
   1. `cd $FRAMEWORK_PATH/metawarenn_inference`
   2. `sh download_tflite_models.sh` # Creates tflite_models directory inside tvm/ & downloads models into it
   3. `python3 inference_regression_tflite.py`
