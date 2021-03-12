import copy
from typing import List

import onnx
import pytest
from loguru import logger
from onnx import helper

from obfuscator.obfuscate import Obfuscator, get_available_obfuscator, insert_node


@pytest.mark.parametrize("architecture", ["simple_conv", "simple_linear", "default"])
def test_insert_node(architecture):
    """Check insertion of node in onnx graph."""
    logger.debug(f"using architecture: {architecture}")
    user_onnx_model = onnx.load_model(f"tests/data/saved_models/onnx/{architecture}.onnx")
    onnx_model = copy.deepcopy(user_onnx_model)
    new_node = helper.make_node("Relu", ["3"], ["4"])
    insert_node(onnx_model.graph.node, 1, new_node)
    assert len(list(user_onnx_model.graph.node)) + 1 == len(list(onnx_model.graph.node))


def test_get_available_obfuscator():
    assert list(get_available_obfuscator()) == ["noop", "encrypt", "elu_on_relu"]
    assert list(get_available_obfuscator(unit_test=False)) == ["noop", "encrypt", "elu_on_relu", "split_conv"]


def test_obfuscator_sorted():
    ofbuscator_map = get_available_obfuscator()
    obfuscator_list: List[Obfuscator] = [value for _, value in ofbuscator_map.items()]
    obfuscator_list = sorted(obfuscator_list, key=lambda obfuscator: obfuscator.priority, reverse=True)
    obfuscator_list = [obfuscator.name for obfuscator in obfuscator_list]
    assert obfuscator_list == ["elu_on_relu", "encrypt", "noop"]

    ofbuscator_map = get_available_obfuscator(unit_test=False)
    obfuscator_list: List[Obfuscator] = [value for key, value in ofbuscator_map.items()]
    obfuscator_list = sorted(obfuscator_list, key=lambda obfuscator: obfuscator.priority, reverse=True)
    obfuscator_list = [obfuscator.name for obfuscator in obfuscator_list]
    assert obfuscator_list == ["split_conv", "elu_on_relu", "encrypt", "noop"]
