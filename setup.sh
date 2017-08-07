#!/bin/bash

# global
export BADNET="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${PYTHONPATH}:${BADNET}/python/

# config
export BADNET_DATA="/home/snarayan/hscratch/baconarrays/v8_repro/"
