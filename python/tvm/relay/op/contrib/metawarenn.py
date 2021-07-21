# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument

"""MetaWareNN codegen supported operators."""

import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by MetaWareNN.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by MetaWareNN.
    """

    @tvm.ir.register_op_attr(op_name, "target.metawarenn")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper

_register_external_op_helper("add")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.global_avg_pool2d")
_register_external_op_helper("nn.relu")
_register_external_op_helper("reshape")

def partition_for_metawarenn(mod, params=None):
    """Partition the graph greedily offloading supported operators to MetaWareNN.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    mod : Tuple[Module, Dict[str, Any]]
        A tuple of annotated and partitioned module
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    seq = tvm.transform.Sequential(
    [
        transform.InferType(),
        transform.RemoveUnusedFunctions(),
        transform.FoldConstant(),
        transform.AnnotateTarget("metawarenn"),
        transform.MergeCompilerRegions(),
        transform.PartitionGraph(),
        transform.InferType(),
    ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod
