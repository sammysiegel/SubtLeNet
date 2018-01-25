#!/bin/bash

# global
export SUBTLENET="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${PYTHONPATH}:${SUBTLENET}/python/

# config
#export SUBTLENET_DATA="/fastscratch/snarayan/pandaarrays/v1/"
#mkdir -p $SUBTLENET_DATA
export SUBTLENET_FIGSDIR="/home/snarayan/public_html/figs/deep/v1"
mkdir -p $SUBTLENET_FIGSDIR

alias npy="mypy ${SUBTLENET}/bin/npy"
