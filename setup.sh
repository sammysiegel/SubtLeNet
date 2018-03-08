#!/bin/bash

# global
export SUBTLENET="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${PYTHONPATH}:${SUBTLENET}/python/ # use this for dev purposes

# config
export SUBTLENET_FIGSDIR="/home/snarayan/public_html/figs/deep/v1"
mkdir -p $SUBTLENET_FIGSDIR

export PATH="${SUBTLENET}/bin":$PATH
