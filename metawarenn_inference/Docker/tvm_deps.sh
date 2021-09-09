mkdir TVM
cd TVM
apt -y update
apt -y install git
apt-get -y install wget
apt-get -y install unzip
apt-get -y install openssh-client
apt-get -y install gedit vim
apt-get -y install build-essential
apt-get -y install libssl-dev
apt-get -y install python3-pip
apt-get -y install llvm
apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev cmake libedit-dev libxml2-dev
wget https://github.com/git-lfs/git-lfs/releases/download/v2.13.3/git-lfs-linux-amd64-v2.13.3.tar.gz
tar -xf git-lfs-linux-amd64-v2.13.3.tar.gz
chmod 755 install.sh
./install.sh
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-all-3.11.3.tar.gz
tar -xf protobuf-all-3.11.3.tar.gz
cd protobuf-3.11.3
./configure
make
make check
make install
cd ./python
python3 setup.py build
python3 setup.py test
python3 setup.py install
export PATH=/usr/local/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/include:${CPLUS_INCLUDE_PATH}
cd ../..
pip3 install numpy decorator attrs
pip3 install tornado
pip3 install onnx
pip3 install psutil xgboost cloudpickle
pip3 install tflite==2.3.0
git clone --recursive https://github.com/SowmyaDhanapal/tvm.git tvm
cd tvm
git checkout metawarenn_dev
git submodule sync
git submodule update
git submodule update --init --recursive
cd src/runtime/contrib/metawarenn/metawarenn_lib
git checkout metawarenn_dev
git lfs install
git lfs pull
