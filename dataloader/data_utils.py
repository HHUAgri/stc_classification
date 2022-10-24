# -*- coding: utf-8 -*-

"""
Functions for time-series dataset

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import numpy as np


def write_numpy_array(numpy_array, target_path):

    parent_dir = os.path.dirname(target_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    np.save(target_path, numpy_array)
    return target_path


def load_numpy_array(array_path):
    return np.load(array_path)
