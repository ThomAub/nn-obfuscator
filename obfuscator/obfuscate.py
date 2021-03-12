"""Obfuscator Types and Variant."""
import copy
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import onnx
from loguru import logger
from onnx import (
    IR_VERSION,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    helper,
    numpy_helper,
)

DEBUG = os.getenv("DEBUG")

if DEBUG:
    logger.debug(f"Using {IR_VERSION} of ONNX")  # pragma: no cover


class ModificationType(Enum):

    """
    Enum storing the variant of modification to be made on the DAG.

    Variant can be:
        DELETE: Remove current pointed node and and links previous inputs with outputs
        REPLACE: Replace current pointed node and links previous inputs/outpus to this node
        ADD: Insert a node between two nodes and links with previous node outputs and next node inputs
        IDENTITY: Replace current pointed node with a copy with some metadata modify
    """

    DELETE = "delete"
    REPLACE = "replace"
    ADD = "add"
    IDENTITY = "identity"
    REPLACEINIT = "replace"
    ADDINIT = "add_initializer"


@dataclass
class Modification(metaclass=ABCMeta):

    """
    Modification Type.

    modif_type (ModificationType): Enum variant describing the type of modification to be made
    old_node (Optional[NodeProto]): Copy of the node to modify, possibly None
    old_node_index (Optional[int]): Index of the node to modify, possibly None
    new_node (Optional[NodeProto]): Node created and to be inserted in the graph, possibly None
    """

    modif_type: ModificationType
    old_node: Optional[NodeProto]
    old_node_index: Optional[int]
    new_node: Optional[NodeProto]


@dataclass
class IdentityModification(Modification):

    """
    Specification of Modification ReplaceModification. #TODO: update

    modif_type (ModificationType): Enum variant describing the type of modification to be made
    old_node (Optional[NodeProto]): Copy of the node to modify
    old_node_index (Optional[int]): Index of the node to modify
    """

    modif_type: ModificationType = ModificationType.IDENTITY
    old_node: Optional[NodeProto] = None
    old_node_index: Optional[NodeProto] = None
    new_node: Optional[NodeProto] = None


@dataclass
class ReplaceModification(Modification):

    """
    Specification of Modification ReplaceModification.

    modif_type (ModificationType): Enum variant describing the type of modification to be made
    old_node (Optional[NodeProto]): Copy of the node to modify
    old_node_index (Optional[int]): Index of the node to modify
    """

    modif_type: ModificationType
    old_node_index: int
    old_node: NodeProto
    new_node: Optional[NodeProto]


@dataclass
class AddModification(Modification):

    """
    Specification of Modification ReplaceModification. #TODO: update

    modif_type (ModificationType): Enum variant describing the type of modification to be made
    old_node (Optional[NodeProto]): Copy of the node to modify
    old_node_index (Optional[int]): Index of the node to modify
    """

    modif_type: ModificationType
    old_node: Optional[NodeProto]
    old_node_index: int
    new_node: NodeProto


@dataclass
class DeleteModification(Modification):

    """
    Specification of Modification ReplaceModification. #TODO: update

    modif_type (ModificationType): Enum variant describing the type of modification to be made
    old_node (Optional[NodeProto]): Copy of the node to modify
    old_node_index (Optional[int]): Index of the node to modify
    """

    modif_type: ModificationType
    old_node: NodeProto
    old_node_index: int
    new_node: Optional[NodeProto]


@dataclass
class AddInitModification(Modification):

    """
    Specification of Modification ReplaceModification. #TODO: update

    modif_type (ModificationType): Enum variant describing the type of modification to be made
    old_node (Optional[NodeProto]): Copy of the node to modify
    old_node_index (Optional[int]): Index of the node to modify
    """

    modif_type: ModificationType
    old_node: Optional[NodeProto]
    old_node_index: int
    new_node: NodeProto


class Obfuscator(metaclass=ABCMeta):

    """Obfuscator base class."""

    @abstractmethod
    def check_model(self, onnx_model: ModelProto) -> List[Modification]:
        """
        Loop over the node in the graph and check for possible modifications.

        Args:
            onnx_model (ModelProto): onnx model to be obfuscated

        Returns:
            List[Modification]: List of the different modification to be made on the graph

        """
        pass  # pragma: no cover


class NoOpObfuscator(Obfuscator):

    """Obfuscator no op class mostly for tests."""

    def __init__(self):
        self.name: str = "noop"
        self.priority: int = 0

    def check_model(self, onnx_model) -> List[Modification]:
        """
        Loop over the node in the graph and check for possible modifications.

        Args:
            onnx_model (ModelProto): Onnx model to be obfuscated

        Returns:
            List[Modification]: List of the different modification to be made on the graph

        """
        modifications = []
        for node in onnx_model.graph.node:
            if DEBUG:  # pragma: no cover
                logger.debug(self.name)
                logger.debug(node.name)
                logger.debug(node.input)
                logger.debug(node.output)
                logger.debug(node.op_type)
                logger.debug(node.domain)
                logger.debug(node.attribute)
                logger.debug(node.doc_string)
            modifications.append(IdentityModification(ModificationType.IDENTITY, None, None, None))
        return modifications


class EncryptNameObfuscator(Obfuscator):

    """Obfuscator class to encrypt name of the operator."""

    def __init__(self, prefix: str = "_encrypt"):
        self.name: str = "encrypt"
        self.priority: int = 1
        self.prefix: str = prefix

    def check_model(self, onnx_model) -> List[Modification]:
        """
        Loop over the node in the graph and check for possible name encryption.

        Args:
            onnx_model (ModelProto): Onnx model to be obfuscated

        Returns:
            List[Modification]: List of the different node to have their name encrypted

        """
        modifications = []
        for idx, node in enumerate(onnx_model.graph.node):
            new_node = copy.deepcopy(node)
            new_node.name = f"{self.prefix}_{node.name}"
            modifications.append(
                ReplaceModification(ModificationType.REPLACE, old_node=node, old_node_index=idx, new_node=new_node)
            )
        return modifications


class EluOnReluObfuscator(Obfuscator):

    """Obfuscator class to encrypt name of the operator."""

    def __init__(self, prefix: str = "_encrypt"):
        self.name: str = "elu_on_relu"
        self.priority: int = 5
        self.relu_count: int = 0

    def check_model(self, onnx_model) -> List[Modification]:
        """
        Loop over the node in the graph and check for possible name encryption.

        Args:
            onnx_model (ModelProto): Onnx model to be obfuscated

        Returns:
            List[Modification]: List of the different node to have their name encrypted

        """
        modifications = []
        for idx, node in enumerate(onnx_model.graph.node):
            if node.op_type == "Relu":
                connection = [f"_{self.relu_count}"]
                new_node = helper.make_node("Elu", name=f"elu_{self.relu_count}", inputs=node.input, outputs=connection)
                del node.input[:]
                node.input.extend(connection)
                modifications.append(
                    AddModification(
                        ModificationType.ADD, old_node=node, old_node_index=idx - 1 + self.relu_count, new_node=new_node
                    )
                )
                self.relu_count += 1
        return modifications


class SplitConvObfuscator(Obfuscator):

    """Obfuscator class to encrypt name of the operator."""

    def __init__(self, prefix: str = "_encrypt"):
        self.name: str = "split_conv"
        self.priority: int = 8
        self.conv_count: int = 0

    def check_model(self, onnx_model) -> List[Modification]:
        """
        Loop over the node in the graph and check for possible name encryption.

        Args:
            onnx_model (ModelProto): Onnx model to be obfuscated

        Returns:
            List[Modification]: List of the different node to have their name encrypted

        """
        modifications = []
        for idx, node in enumerate(onnx_model.graph.node):
            if node.op_type == "Conv":
                # Keep a untouched copy
                old_node = copy.deepcopy(node)

                # Retrieve the convolution weight and bias to be split into two parallel convolution
                weight = [name for name in node.input if "weight" in name][0]
                bias = [name for name in node.input if "bias" in name][0]
                weights = [ini for ini in onnx_model.graph.initializer if ini.name in weight]
                bias = [ini for ini in onnx_model.graph.initializer if ini.name in bias]
                weights_np_data = numpy_helper.to_array(weights[0])
                bias_np_data = numpy_helper.to_array(bias[0])
                num_filters = weights_np_data.shape[0] // 2
                num_filters_bias = bias_np_data.shape[0] // 2

                # Sanity check
                assert num_filters == num_filters_bias

                # Name of the output of conv input of concat
                connection_0 = [f"_{self.conv_count}_0"]
                connection_1 = [f"_{self.conv_count}_1"]
                concat_input = []

                # First branch of conv
                conv_0 = copy.deepcopy(node)
                del conv_0.input[1:]
                del conv_0.output[:]

                conv_0.name = f"conv__{self.conv_count}_0"
                conv_0.input.extend([f"W0_{self.conv_count}", f"B0_{self.conv_count}"])
                conv_0.output.extend(connection_0)

                ini_w1_shape = list(weights_np_data.shape)
                ini_b1_shape = list(bias_np_data.shape)
                ini_w1_shape[0] = num_filters
                ini_b1_shape[0] = num_filters

                ini_w0 = helper.make_tensor(
                    name=f"W0_{self.conv_count}",
                    data_type=TensorProto.FLOAT,
                    dims=ini_w1_shape,
                    vals=weights_np_data[num_filters:].ravel(),
                )
                ini_b0 = helper.make_tensor(
                    name=f"B0_{self.conv_count}",
                    data_type=TensorProto.FLOAT,
                    dims=ini_b1_shape,
                    vals=bias_np_data[num_filters:].ravel(),
                )
                concat_input.append(f"_{self.conv_count}_0")
                modifications.append(
                    AddModification(ModificationType.ADD, old_node=node, old_node_index=idx, new_node=conv_0)
                )
                modifications.append(
                    AddInitModification(ModificationType.ADDINIT, old_node=None, old_node_index=None, new_node=ini_w0)
                )
                modifications.append(
                    AddInitModification(ModificationType.ADDINIT, old_node=None, old_node_index=None, new_node=ini_b0)
                )
                # Second branch of conv
                conv_1 = copy.deepcopy(node)
                del conv_1.input[1:]
                del conv_1.output[:]
                conv_1.name = f"conv__{self.conv_count}_1"
                conv_1.input.extend([f"W1_{self.conv_count}", f"B1_{self.conv_count}"])
                conv_1.output.extend(connection_1)

                ini_w1 = helper.make_tensor(
                    name=f"W1_{self.conv_count}",
                    data_type=TensorProto.FLOAT,
                    dims=ini_w1_shape,
                    vals=weights_np_data[:num_filters].ravel(),
                )
                ini_b1 = helper.make_tensor(
                    name=f"B1_{self.conv_count}",
                    data_type=TensorProto.FLOAT,
                    dims=ini_b1_shape,
                    vals=bias_np_data[:num_filters].ravel(),
                )
                concat_input.append(f"_{self.conv_count}_1")
                self.conv_count += 1
                modifications.append(
                    AddModification(ModificationType.ADD, old_node=node, old_node_index=idx, new_node=conv_1)
                )
                modifications.append(
                    AddInitModification(ModificationType.ADDINIT, old_node=None, old_node_index=None, new_node=ini_w1)
                )
                modifications.append(
                    AddInitModification(ModificationType.ADDINIT, old_node=None, old_node_index=None, new_node=ini_b1)
                )
                new_concat = helper.make_node("Concat", axis=0, inputs=concat_input, outputs=node.output)
                modifications.append(
                    AddInitModification(ModificationType.ADD, old_node=node, old_node_index=idx, new_node=new_concat)
                )
                modifications.append(
                    AddInitModification(ModificationType.DELETE, old_node=old_node, old_node_index=None, new_node=None)
                )
        return modifications


def get_available_obfuscator(unit_test: bool = True) -> Dict[str, Obfuscator]:
    """
    Return available Obfuscator.

    Args:
        unit_test (bool): Flag to only display Obfuscator available and unit tested

    Returns:
        List[str]: Different Obfuscator's name currently available

    """
    if unit_test:
        return {"noop": NoOpObfuscator(), "encrypt": EncryptNameObfuscator(), "elu_on_relu": EluOnReluObfuscator()}
    else:
        return {
            "noop": NoOpObfuscator(),
            "encrypt": EncryptNameObfuscator(),
            "elu_on_relu": EluOnReluObfuscator(),
            "split_conv": SplitConvObfuscator(),
        }


def insert_node(graph: GraphProto, index: int, node: NodeProto):
    """
    Insert helper function to insert a node at a given index in the link list of the graph.

    Args:
        graph (GraphProto): Onnx graph to be modify
        index (int): Index to insert the new node
        node (NodeProto): The node to be inserted

    """
    graph.extend([graph[-1]])
    for i in reversed(range(index + 1, len(graph) - 1)):
        graph[i].CopyFrom(graph[i - 1])
    graph[index].CopyFrom(node)


def apply_modification(onnx_model: ModelProto, modification: Modification):
    """
    Apply a given modification to the graph.

    Args:
        onnx_model (ModelProto): Onnx model to be modify
        modification (Modification): A modification

    """
    if isinstance(modification, IdentityModification):
        logger.info("Not doing much")

    elif isinstance(modification, ReplaceModification):
        assert modification.old_node_index is not None, "Trying to replace a node without its index"
        insert_node(onnx_model.graph.node, modification.old_node_index + 1, modification.new_node)
        del onnx_model.graph.node[modification.old_node_index]

    elif isinstance(modification, AddModification):
        assert modification.old_node_index is not None, "Trying to add a node without its input index"
        insert_node(onnx_model.graph.node, modification.old_node_index + 1, modification.new_node)

    elif isinstance(modification, DeleteModification):
        for idx, node in enumerate(onnx_model.graph.node):
            if node.name == modification.old_node.name:
                print(node.name)
                del onnx_model.graph.node[idx]

    elif isinstance(modification, AddInitModification):
        onnx_model.graph.initializer.extend([modification.new_node])

    else:
        raise ValueError(f"Trying to use a {modification.modif_type} but not yet supported.")


def obfuscate(onnx_model: ModelProto, obfuscator: Obfuscator) -> ModelProto:
    """
    Obfuscate the model with a given Obfuscator.

    Args:
        onnx_model (ModelProto): onnx_model
        obfuscator (Obfuscator): obfuscator

    Returns:
        ModelProto: New Onnx model obfuscated. Should be hard to read and recognize

    """
    onnx.checker.check_model(onnx_model)
    onnx.helper.strip_doc_string(onnx_model)
    for modification in obfuscator.check_model(onnx_model):
        apply_modification(onnx_model, modification)

    return onnx_model
