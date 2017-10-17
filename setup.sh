#!/bin/bash

# global
export BADNET="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${PYTHONPATH}:${BADNET}/python/

# config
export BADNET_DATA="/home/snarayan/hscratch/baconarrays/v8_repro/"
export BADNET_FIGSDIR="/home/snarayan/public_html/figs/badnet/prong_defs/"
mkdir -p $BADNET_FIGSDIR
