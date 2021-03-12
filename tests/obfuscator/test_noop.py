import copy

import numpy as np
import onnx
import pytest
from loguru import logger
from onnx.numpy_helper import to_array

from create_onnx import small_architecture_list
from obfuscator.obfuscate import NoOpObfuscator, obfuscate


@pytest.mark.parametrize("architecture", small_architecture_list)
def test_noop_obfuscator(architecture):
    """Check that nothing changed."""
    logger.debug(f"using architecture: {architecture}")
    user_onnx_model = onnx.load_model(f"tests/data/saved_models/onnx/{architecture}.onnx")
    onnx_model = copy.deepcopy(user_onnx_model)
    noop_obfuscator = NoOpObfuscator()
    obfusctated_model = obfuscate(onnx_model=onnx_model, obfuscator=noop_obfuscator)
    obfusctated_nodes = list(obfusctated_model.graph.node)
    obfusctated_initializer = list(obfusctated_model.graph.initializer)
    user_nodes = list(user_onnx_model.graph.node)
    user_initializer = list(user_onnx_model.graph.initializer)
    assert len(obfusctated_nodes) == len(user_nodes)
    assert obfusctated_nodes[0].name == user_nodes[0].name
    assert obfusctated_initializer[0].name == user_initializer[0].name
    np.testing.assert_almost_equal(to_array(obfusctated_initializer[0]), to_array(user_initializer[0]))
