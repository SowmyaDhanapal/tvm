#ifndef METAWARENN_COMMON_H_
#define METAWARENN_COMMON_H_

#define ONNX 0
#define TFLITE 0
#define GLOW 0
#define TVM 1

//ONNXRuntime
#if ONNX
#include "onnx/onnx-ml.pb.h"
#include <numeric>
#endif

//TFLite
#if TFLITE
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <map>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif

//GLOW
#if GLOW
#include "Glow/Graph/Graph.h"
#include "Glow/Graph/Utils.h"
#endif

//TVM
#if TVM
#include <numeric>
#include <regex>
#include "tvm/json/json_node.h"
#include "tvm/json/json_runtime.h"
#endif

#include <boost/serialization/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/streams/bufferstream.hpp>
#include <boost/serialization/vector.hpp>

#if ONNX
using namespace onnx;
#endif
#if GLOW
using namespace glow;
#endif
#if TVM
using namespace tvm::runtime::json;
#endif
#endif //METAWARENN_COMMON_H_
