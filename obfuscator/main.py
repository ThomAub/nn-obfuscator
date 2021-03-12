import argparse
import os
import pathlib
from typing import List

import onnx
from loguru import logger

from obfuscator.obfuscate import Obfuscator, get_available_obfuscator, obfuscate

DEBUG = os.getenv("DEBUG")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ONNX Obfuscator")
    parser.add_argument(
        "onnx_input",
        type=str,
        help="path to the onnx model to be obfuscated",
    )
    parser.add_argument(
        "onnx_output",
        type=str,
        help="path to the onnx model after the obfuscation",
    )
    parser.add_argument(
        "--obfuscations",
        type=str,
        nargs="+",
        help=f"flag to only select some obfuscation/modification from {get_available_obfuscator()}",
    )
    return parser


def main():

    args_parser = get_args_parser()
    args = args_parser.parse_args()

    # Parsing the different args
    onnx_input_path = pathlib.Path(args.onnx_input)
    onnx_output_path = pathlib.Path(args.onnx_output)
    assert onnx_input_path.suffix == ".onnx", "Only accepting onnx extension as model input"
    assert onnx_output_path.suffix == ".onnx", "Only accepting onnx extension as model output"

    if args.obfuscations is None:
        obfuscations = list(get_available_obfuscator())
    else:
        available_obfuscations = list(get_available_obfuscator(unit_test=False))
        obfuscations = args.obfuscations
        not_supported = set(obfuscations).difference(available_obfuscations)
        if len(not_supported) != 0:
            raise ValueError(
                f"""
        This modifications are not supported: {not_supported}
        Currently available and tested are {list(get_available_obfuscator())}
        and possibly available but not tested are {available_obfuscations}
        """
            )

    # Checking the validity of the onnx model
    logger.info(f"Using the Onnx model from {onnx_input_path.name}")
    onnx.checker.check_model(onnx_input_path.as_posix(), full_check=True)
    onnx_model = onnx.load_model(onnx_input_path.as_posix())

    # Setup of the different obfuscations
    logger.info(f"Obfuscator using {obfuscations}")
    ofbuscator_map = get_available_obfuscator()
    obfuscator_list: List[Obfuscator] = [ofbuscator_map[modif]() for modif in obfuscations]
    obfuscator_list = sorted(obfuscator_list, key=lambda obfuscator: obfuscator.priority, reverse=True)

    for obfuscator in obfuscator_list:
        logger.info(f"Obfuscating with {obfuscator.name}")
        onnx_model = obfuscate(onnx_model=onnx_model, obfuscator=obfuscator)
        # onnx.checker.check_model(onnx_model)

    onnx.save_model(onnx_model, onnx_output_path.as_posix())
    logger.info(f"Saving the obfuscated Onnx model to {onnx_output_path.name}")


if __name__ == "__main__":
    main()
