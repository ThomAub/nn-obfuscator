# Neural Network Obfuscator  <!-- omit in toc -->

The goal is to create a proof of concept for a neural network obfuscator.

Given a neural network trained on a dataset $X$ called $NN_0$ it should be transformed into another neural network called $NN_1$ such that:

- $NN_0$ and $NN_1$ have almost the same accuracy on $X$
- $NN_1$ topology is different and more complex to understand than $NN_0$
- $NN_0$ parameters should not be easy to retrieve when looking at $NN_1$ parameters and topology
- $NN_1$ inference speed can be up to x500 longer than $NN_0$

## Table of contents  <!-- omit in toc -->

- [Disclaimer](#disclaimer)
- [Compiling the model](#compiling-the-model)
- [Architecture of the Obfuscator](#architecture-of-the-obfuscator)
- [Feature Tracking](#feature-tracking)
- [Install & Setup](#install--setup)
  - [Local setup with pyenv and poetry](#local-setup-with-pyenv-and-poetry)
  - [Docker setup](#docker-setup)
- [Usage](#usage)

## Disclaimer

This is a toy project. Mostly to play with ONNX and onnx-runtime.

Inspired by [onnx-optimizer](https://github.com/onnx/optimizer) and [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)

An obfuscator use-case is to hide and protect one's intellectual property.
In my humble opinion this technique is not really effective to protect one's model.
If this model is made available to an end user or customer then it should be safe to say this model has a good enough accuracy.
The user then can just make inferences using on his dataset and use the predictions as labels for his dataset.
This is a similar process to knowledge distillation, a technique often used to compress neural network.

Also it's a toy project, privacy regarding user data and protection of a company intellectual property are real challenges for the next years that could enable greater use-case for machine learning algorithms.

Changing the topology of the model can be done in either the framework used for training like Pytorch or Tensorflow but one should make sure that all operation are supported by the framework and ONNX.

## Compiling the model

One of the first idea that can come to mind is to statically compile the model with the execution engine or compile the model to another format that can be run by an engine.
This process makes the model, to my knowledge, impossible to be clearly visible by the user. The tool that first come to mind is [tensorRT](https://developer.nvidia.com/tensorrt). After a transformation from onnx to tensorRT, [onnx-tensorRT](https://github.com/onnx/onnx-tensorrt), engine format. The model can only be run by their runtime. To my knowledge there is no existing tool that would make it easy to see the topology and the weights of the model. Other framework could allow this hiding by compiling paradigm. [TVM](https://github.com/apache/tvm) from OctoML is one of the other compiler for Neural Network and can be a good choice also for performance. Moreover some new tool like [Tract](https://github.com/sonos/tract) from sonos could be another candidate.
Still this is an idea and not really answering to the challenge.

## Architecture of the Obfuscator

The goal during the week would be to have a simple CLI that can take as input any ONNX model and convert it to another ONNX model.
There would be a main function [obfuscate](obuscator/obfuscate.py) that would apply some possible changes that a first module called `Obfuscator` would have identify as possible changes.

The Obfuscator has a simple api.

```python
class Obfuscator(ABC):

    def check_model(self, onnx_model: ModelProto) -> List[Modification]:
        pass
```

It takes a ONNX graph as input an see by stepping over all the operator if it can be modify.

Modifications can be of multiple type and resume in this Enum.

```python
class ModificationType(Enum):
    DELETE = "delete"
    REPLACE = "replace"
    ADD = "add"
    IDENTITY = "identity"
    REPLACEINIT = "replace"
    ADDINIT = "add_initializer"
```

Each modification is link to a node from the unprocess, or currently process graph and add or swap some node.

```python
@dataclass
class Modification:
    modif_type: ModificationType
    old_node: Optional[NodeProto]
    old_node_index: Optional[int]
    new_node: Optional[NodeProto]
```

To note: This approach doesn't require to rewrite the complete graph. It's quite simple but it might not be the best to compose more intricate `Obfuscator`

## Feature Tracking

- [X] NoOp Obfuscator that run over all node and just do some print

- [X] EncryptName Obfuscator to be able to change name of operator

- [ ] Add a operator that doesn't really affect chain of maths operator
  - [ ] Add some identity operator, + 0, * 1
  - [ ] Add then subtrack with random weights
  - [X] Add a Elu before a relu
  - [ ] Add a maxpooling with a kernel size similar to input channels
  - [ ] Add some back to back transpose

- [ ] Replace standard Convolution with equivalent operator
  - [X] Split a conv into two parallel convolution
  - [ ] Replace a conv with equivalent linear layer i.e fully connected

- [ ] Encryption of the different weights
  - [ ] Quantize to f16 or int8 then add a secret key like a big prime number
An Neural Network Obfuscator based on ONNX

## Install & Setup

### Local setup with pyenv and poetry

1. Install Python 3.x using [pyenv/pyenv: Simple Python version management](https://github.com/pyenv/pyenv#installation)

    <details>
    <summary>Install on Mac OS X</summary>
    ```sh
    #eg on Mac Os X
    brew update
    brew install pyenv
    pyenv install 3.8.2 # or any other ^3.8
    ```
    </details>

    <details>
    <summary>Install on Ubuntu Desktop</summary>
    ```bash
    git clone https://github.com/pyenv/pyenv ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc # or >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc # or >> ~/.bashrc
    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.zshrc # or ~/.bash_profile
    source ~/.zshrc # or >> ~/.bashrc
    pyenv install 3.8.2 # or any other ^3.8
    ```
    </details>

    <details>
    <summary>Install on Windows</summary>
    Follow you prefered installation from [pyenv-win](https://github.com/pyenv-win/pyenv-win#get-pyenv-win)
    ```powershell
    pyenv install 3.8.2 # or any other ^3.8
    ```
    </details>

2. Install [Poetry](https://poetry.eustace.io/) for dependency management

    ```sh
    curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
    source $HOME/.poetry/env
    poetry --version
    ```

3. Install dependencies in virtual environement

```sh
poetry config virtualenvs.in-project true # Can be set globally
poetry shell # initialize `.venv/` virtualenv directory
poetry install
# Also make sure to install libomp.dylib for onnx-runtime
brew install libomp
```

### Docker setup

You can use the script `dev.dockerfile.build.sh` in `.devcontainer/` to build an image with all the dependencies.
Then you can use the script `dev.dockerfile.run.sh` in `.devcontainer/` the other one to run and attach to a container.

Another way can be to use the built-in feature of [remote container](https://code.visualstudio.com/docs/remote/containers) from vscode

## Usage

```sh
poetry run obfuscator path_to_private_model.onnx path_to_obfuscated_destination.onnx
```

```sh
poetry run obfuscator --help
```

```sh
poetry run python create_onnx.py
poetry run pytest tests
```