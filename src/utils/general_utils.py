import numpy as np
import networkx as nx
import pickle

import os
from os.path import exists

def digitize_equalize(values, bins=10):
    '''
    Given values, create an equalized histogram and then digitize them into bins
    Args:
        values: np.array - with diverge values.
        bins: int - number of wanted bins.

    Returns:
        values_digitized: np.array - with the same length as values, were now values_digitized[i] is the bin number
                                     of values[i] after equalization and digitization.
    '''
    # equalize values histogram
    values_equalized = equalize(values)

    # digitize values to bins
    values_digitized = digitize(values_equalized, bins=bins)

    return values_digitized


def digitize(values, bins=10):
    '''
    Given values, digitize them into bins.
    Args:
        values: np.array - with diverge values.
        bins: int - number of wanted bins.

    Returns:
        values_digitized: np.array - with the same length as values, were now values_digitized[i] is the bin number
                                     of values[i] after digitization.
    '''
    values_digitized = np.digitize(values, bins=np.histogram(values, bins=bins)[1])
    values_digitized[values_digitized == 0] = 1
    values_digitized[values_digitized > bins] = bins
    return values_digitized


def equalize(values):
    '''
    Given values, equalized them.
    Args:
        values: np.array - with diverge values.

    Returns:
        values_equalized: np.array - with the same length as values, were now values_equalized[i] is the equalized
                                     value of values[i] after histogram equalization.
    '''
    histogram, bins = np.histogram(values, 100000)
    cdf = histogram.cumsum()  # cumulative distribution function
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])  # normalize

    # use linear interpolation of cdf to find new pixel values
    values_equalized = np.interp(values, bins[:-1], cdf)

    return values_equalized


def get_nx_function(name):
    '''
    return the networkx function given its name.
    Args:
        name: str - the netwrokx function name

    Returns:
        networkx - function
    '''
    return eval(f'nx.{name}')


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', filename, ':', e)
        raise
    return data

def sub_dict(dict,keys):
    return {k: v for k, v in dict.items() if k in keys}

def get_if_path_exist(path):
    if exists(path):
        i = 2
        while exists(path + f' ({i})'):
            i += 1
        path = path + f' ({i})'
    os.mkdir(path)
    return path