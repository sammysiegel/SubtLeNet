#!/bin/bash

# global
export BADNET="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${PYTHONPATH}:${BADNET}/python/

# config
#export BADNET_DATA="/fastscratch/snarayan/pandaarrays/v1/"
#mkdir -p $BADNET_DATA
export BADNET_FIGSDIR="/home/snarayan/public_html/figs/deep/v1"
mkdir -p $BADNET_FIGSDIR

alias npy='mypy /home/snarayan/home000/BAdNet/pf/adv/npy'
