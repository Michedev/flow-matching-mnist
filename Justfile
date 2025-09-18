PYTHON_VERSION := "3.13"
ENV_NAME := "mnist"

alias p := python
setup-venv:
    conda create -n {{ENV_NAME}} python={{PYTHON_VERSION}}
    conda run -n {{ENV_NAME}} pip install -r requirements.txt

python *ARGS:
    conda run -n {{ENV_NAME}} python {{ARGS}}

