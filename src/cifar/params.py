# -*- coding: utf-8 -*-
#
# params.py: Implements IO functions for pickling network parameters.
#
import cPickle as pickle
import os

import lasagne as nn

__all__ = [
    'read_model_data',
    'write_model_data',
]

PARAM_EXTENSION = 'params'


def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, PARAM_EXTENSION))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    nn.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = nn.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(filename, 'w') as f:
        pickle.dump(data, f)
