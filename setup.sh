#!/bin/bash

# global
export SUBTLENET="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${PYTHONPATH}:${SUBTLENET}/python/ # use this for dev purposes

# config
export SUBTLENET_FIGSDIR="/uscms_data/d3/jkrupa/subjetNN/CMSSW_10_2_11/src/SubtLeNet/train/smh/plots/"
mkdir -p $SUBTLENET_FIGSDIR

export PATH="${SUBTLENET}/bin":$PATH
