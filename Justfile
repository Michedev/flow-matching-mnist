PYTHON_VERSION := "3.13"
ENV_NAME := "mnist-flow-matching"

alias p := python
alias setup := setup-env
setup-env:
    conda create -n {{ENV_NAME}} python={{PYTHON_VERSION}}
    conda run -n {{ENV_NAME}} pip install -r requirements.txt

[confirm]
remove-env:
    conda env remove -n {{ENV_NAME}}


python *ARGS:
    conda run --live-stream -n {{ENV_NAME}} python {{ARGS}}

