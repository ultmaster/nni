# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================
import logging

from . import trial


_logger = logging.getLogger(__name__)
_MUTABLE_LAYER_SPACE_PREFIX = "_mutable_layer"


def classic_mode(
        mutable_id,
        mutable_layer_id,
        funcs,
        funcs_args,
        fixed_inputs,
        optional_inputs,
        optional_input_size):
    '''Execute the chosen function and inputs directly.
    In this mode, the trial code is only running the chosen subgraph (i.e., the chosen ops and inputs),
    without touching the full model graph.'''
    if trial._params is None:
        trial.get_next_parameter()
    try:
        mutable_block = trial.get_current_parameter(mutable_id)
        # Great! There is a NAS tuner
        chosen_layer = mutable_block[mutable_layer_id]["chosen_layer"]
        chosen_inputs = mutable_block[mutable_layer_id]["chosen_inputs"]
        real_chosen_inputs = [optional_inputs[input_name]
                              for input_name in chosen_inputs]
    except KeyError:
        # Try to find converted NAS parameters
        params = trial.get_current_parameter()
        expected_prefix = _MUTABLE_LAYER_SPACE_PREFIX + "/" + mutable_id + "/" + mutable_layer_id
        chosen_layer = params[expected_prefix + "/layer_choice"]

        # find how many to choose
        optional_input_size = int(params[expected_prefix + "/optional_input_size"])  # convert from float (uniform) to int
        total_state_size = len(optional_inputs) ** optional_input_size

        # find who to choose, can duplicate
        optional_input_state = int(params[expected_prefix + "/optional_input_chosen_state"] * total_state_size)
        real_chosen_inputs, real_chosen_input_names = [], []
        # make sure dict -> list produce stable result by sorting
        optional_inputs_keys = sorted(optional_inputs.keys())
        optional_inputs_list = [optional_inputs[k] for k in optional_inputs_keys]
        for i in range(optional_input_size):
            real_chosen_inputs.append(optional_inputs_list[optional_input_state % len(optional_inputs)])
            real_chosen_input_names.append(optional_inputs_keys[optional_input_state % len(optional_inputs)])
            optional_input_state //= len(optional_inputs)

        _logger.info("%s_%s: layer: %s, optional inputs: %s" % (mutable_id, mutable_layer_id,
                                                                chosen_layer, real_chosen_input_names))

    layer_out = funcs[chosen_layer](
        [fixed_inputs, real_chosen_inputs], **funcs_args[chosen_layer])

    return layer_out


def enas_mode(
        mutable_id,
        mutable_layer_id,
        funcs,
        funcs_args,
        fixed_inputs,
        optional_inputs,
        optional_input_size,
        tf):
    '''For enas mode, we build the full model graph in trial but only run a subgraph。
    This is implemented by masking inputs and branching ops.
    Specifically, based on the received subgraph (through nni.get_next_parameter),
    it can be known which inputs should be masked and which op should be executed.'''
    name_prefix = "{}_{}".format(mutable_id, mutable_layer_id)
    # store namespace
    if 'name_space' not in globals():
        global name_space
        name_space = dict()
    name_space[mutable_id] = True
    name_space[name_prefix] = dict()
    name_space[name_prefix]['funcs'] = list(funcs)
    name_space[name_prefix]['optional_inputs'] = list(optional_inputs)
    # create tensorflow variables as 1/0 signals used to form subgraph
    if 'tf_variables' not in globals():
        global tf_variables
        tf_variables = dict()
    name_for_optional_inputs = name_prefix + '_optional_inputs'
    name_for_funcs = name_prefix + '_funcs'
    tf_variables[name_prefix] = dict()
    tf_variables[name_prefix]['optional_inputs'] = tf.get_variable(name_for_optional_inputs,
                                                                   [len(
                                                                       optional_inputs)],
                                                                   dtype=tf.bool,
                                                                   trainable=False)
    tf_variables[name_prefix]['funcs'] = tf.get_variable(
        name_for_funcs, [], dtype=tf.int64, trainable=False)

    # get real values using their variable names
    real_optional_inputs_value = [optional_inputs[name]
                                  for name in name_space[name_prefix]['optional_inputs']]
    real_func_value = [funcs[name]
                       for name in name_space[name_prefix]['funcs']]
    real_funcs_args = [funcs_args[name]
                       for name in name_space[name_prefix]['funcs']]
    # build tensorflow graph of geting chosen inputs by masking
    real_chosen_inputs = tf.boolean_mask(
        real_optional_inputs_value, tf_variables[name_prefix]['optional_inputs'])
    # build tensorflow graph of different branches by using tf.case
    branches = dict()
    for func_id in range(len(funcs)):
        func_output = real_func_value[func_id](
            [fixed_inputs, real_chosen_inputs], **real_funcs_args[func_id])
        branches[tf.equal(tf_variables[name_prefix]['funcs'],
                          func_id)] = lambda: func_output
    layer_out = tf.case(branches, exclusive=True,
                        default=lambda: func_output)

    return layer_out


def oneshot_mode(
        mutable_id,
        mutable_layer_id,
        funcs,
        funcs_args,
        fixed_inputs,
        optional_inputs,
        optional_input_size,
        tf):
    '''Similar to enas mode, oneshot mode also builds the full model graph.
    The difference is that oneshot mode does not receive subgraph.
    Instead, it uses dropout to randomly dropout inputs and ops.'''
    # NNI requires to get_next_parameter before report a result. But the parameter will not be used in this mode
    if trial._params is None:
        trial.get_next_parameter()
    optional_inputs = list(optional_inputs.values())
    inputs_num = len(optional_inputs)
    # Calculate dropout rate according to the formular r^(1/k), where r is a hyper-parameter and k is the number of inputs
    if inputs_num > 0:
        rate = 0.01 ** (1 / inputs_num)
        noise_shape = [inputs_num] + [1] * len(optional_inputs[0].get_shape())
        optional_inputs = tf.nn.dropout(
            optional_inputs, rate=rate, noise_shape=noise_shape)
        optional_inputs = [optional_inputs[idx] for idx in range(inputs_num)]
    layer_outs = [func([fixed_inputs, optional_inputs], **funcs_args[func_name])
                  for func_name, func in funcs.items()]
    layer_out = tf.add_n(layer_outs)

    return layer_out


def reload_tensorflow_variables(session, tf=None):
    '''In Enas mode, this function reload every signal varaible created in `enas_mode` function so
    the whole tensorflow graph will be changed into certain subgraph recerived from Tuner.
    ---------------
    session: the tensorflow session created by users
    tf: tensorflow module
    '''
    subgraph_from_tuner = trial.get_next_parameter()
    for mutable_id, mutable_block in subgraph_from_tuner.items():
        if mutable_id not in name_space:
            continue
        for mutable_layer_id, mutable_layer in mutable_block.items():
            name_prefix = "{}_{}".format(mutable_id, mutable_layer_id)
            # extract layer information from the subgraph sampled by tuner
            chosen_layer = name_space[name_prefix]['funcs'].index(
                mutable_layer["chosen_layer"])
            chosen_inputs = [1 if inp in mutable_layer["chosen_inputs"]
                             else 0 for inp in name_space[name_prefix]['optional_inputs']]
            # load these information into pre-defined tensorflow variables
            tf_variables[name_prefix]['funcs'].load(chosen_layer, session)
            tf_variables[name_prefix]['optional_inputs'].load(
                chosen_inputs, session)


def convert_nas_search_space(search_space):
    """
    :param search_space: raw search space
    :return: the new search space, mutable_layers will be converted into choice
    """
    ret = dict()
    for k, v in search_space.items():
        if "_type" not in v:
            # this should not happen
            _logger.warning("There is no _type in one of your search space values with key '%s'"
                            ". Please check your search space" % k)
            ret[k] = v
        elif v["_type"] != "mutable_layer":
            ret[k] = v
        else:
            _logger.info("Converting mutable_layer search space with key '%s'" % k)
            # v["_value"] looks like {'mutable_layer_1': {'layer_choice': ...} ...}
            values = v["_value"]
            for layer_name, layer_data in values.items():
                # there should be at most layer_choice, optional_inputs, optional_input_size in layer_data

                # add "_mutable_layer" as prefix so that they can be recovered later
                layer_key = _MUTABLE_LAYER_SPACE_PREFIX + "/" + k + "/" + layer_name

                if layer_data.get("layer_choice"):  # filter out empty choice and no choice
                    layer_choice = layer_data["layer_choice"]
                else:
                    raise ValueError("No layer choice found in %s" % layer_key)

                if layer_data.get("optional_inputs") and layer_data.get("optional_input_size"):
                    input_size = layer_data["optional_input_size"]
                    if isinstance(input_size, int):
                        input_size = [input_size, input_size]
                    if input_size[0] > input_size[1] or input_size[0] < 0:
                        _logger.error("Might not be able to handle optional_input_size < 0, please double check")
                    input_size[1] += 1
                else:
                    _logger.info("Optional input choices are set to empty by default")
                    input_size = [0, 1]

                ret[layer_key + "/layer_choice"] = {
                    "_type": "choice", "_value": layer_choice
                }
                # there still appears to be some problems in randint, using uniform to convert here
                ret[layer_key + "/optional_input_size"] = {
                    "_type": "uniform", "_value": input_size
                }
                # map all possible options into a real number in [0, 1)
                ret[layer_key + "/optional_input_chosen_state"] = {
                    "_type": "uniform", "_value": [0, 1]
                }
                # seems still need more discussion here, as these choices are not easily interpretable

    return ret
