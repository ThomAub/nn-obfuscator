import copy

import numpy as np
import onnx
import onnxruntime as rt
import pytest
from loguru import logger
from onnx.numpy_helper import to_array

from create_onnx import small_architecture_list
from obfuscator.obfuscate import EncryptNameObfuscator, obfuscate


@pytest.mark.parametrize("architecture", small_architecture_list)
def test_encrypt_name_obfuscator(architecture):
    """Check that named of node changed."""
    logger.debug(f"using architecture: {architecture}")
    user_onnx_model = onnx.load_model(f"tests/data/saved_models/onnx/{architecture}.onnx")
    onnx_model = copy.deepcopy(user_onnx_model)
    noop_obfuscator = EncryptNameObfuscator()
    obfusctated_model = obfuscate(onnx_model=onnx_model, obfuscator=noop_obfuscator)

    # testing the correctness of proto
    obfusctated_nodes = list(obfusctated_model.graph.node)
    obfusctated_initializer = list(obfusctated_model.graph.initializer)
    onnx.save_model(obfusctated_model, f"tests/data/saved_models/onnx/{architecture}_obfuscated.onnx")
    user_nodes = list(user_onnx_model.graph.node)
    user_initializer = list(user_onnx_model.graph.initializer)
    assert len(obfusctated_nodes) == len(user_nodes)
    assert obfusctated_nodes[0].name == f"_encrypt_{user_nodes[0].name}"
    assert obfusctated_initializer[0].name == user_initializer[0].name
    np.testing.assert_almost_equal(to_array(obfusctated_initializer[0]), to_array(user_initializer[0]))

    # testing matching output
    sess = rt.InferenceSession(f"tests/data/saved_models/onnx/{architecture}.onnx")
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    dummy_input = np.random.random(input_shape)
    pred_user_onnx = sess.run(None, {input_name: dummy_input.astype(np.float32)})[0]

    sess = rt.InferenceSession(f"tests/data/saved_models/onnx/{architecture}_obfuscated.onnx")
    input_name = sess.get_inputs()[0].name
    pred_obfuscated_onnx = sess.run(None, {input_name: dummy_input.astype(np.float32)})[0]
    np.testing.assert_almost_equal(pred_user_onnx, pred_obfuscated_onnx)
