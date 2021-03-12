# This is the base image of the project.
FROM nvidia/cuda:10.2-devel-ubuntu18.04
SHELL ["/bin/bash", "-c"]

# Install here the system dependencies for building c++ or cuda extension of the project
# RUN export DEBIAN_FRONTEND=noninteractive \
# && apt-get update -qq \
# && apt-get -y install --no-install-recommends \
# build-essential \
# cuda-compiler-10-1 \
# cuda-toolkit-10-1

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -qq \
    && apt-get -y install --no-install-recommends \
    git unzip curl zip libgomp1\
    libbz2-dev libffi-dev liblzma-dev libreadline-dev libsqlite3-dev libssl-dev \
    libboost-all-dev libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    python3.8 python3.8-dev python3.8-venv python3-distutils python3-pip \
    && apt-get remove python2.7-minimal -y \
    # # Switch to python 3.7 by default
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && unset DEBIAN_FRONTEND

# Some ENV
ENV PIP_NO_CACHE_DIR=off
ENV POETRY_VERSION=1.1.2

# Install poetry
RUN python -m pip install pip --upgrade
RUN python -m pip install setuptools --upgrade
RUN python -m pip install "poetry==$POETRY_VERSION"

# Create the directory for your project dependencies
COPY pyproject.toml .

# Install the dependencies
RUN poetry config virtualenvs.in-project true \
    && poetry install

CMD /bin/bash