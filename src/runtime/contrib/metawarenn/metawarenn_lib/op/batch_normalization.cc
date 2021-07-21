#include "batch_normalization.h"

namespace metawarenn {

namespace op {

BatchNormalization::BatchNormalization() { std::cout << "\n In BatchNormalization Constructor!!!"; }

BatchNormalization::BatchNormalization(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "BatchNormalization") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void BatchNormalization::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In BatchNormalization fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
