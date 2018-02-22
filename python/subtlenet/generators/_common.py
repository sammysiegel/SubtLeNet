import numpy as np
from .. import config
from ..backend import obj, smear
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
from keras.utils import np_utils
