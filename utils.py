import os, re
from typing import List, Dict
from ast import literal_eval
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder = './data/experiment'
filename_fields = ['vehicle', 'trajectory', 'method', 'condition']

def save_data(Data : List[dict], folder : str, fields=['t', 'p', 'p_d', 'v', 'v_d', 'q', 'R', 'w', 'T_sp', 'q_sp', 'hover_throttle', 'fa', 'pwm']):
    ''' Save {Data} to individual csv files in {folder}, serializing (2+)d ndarrays as lists '''
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print('Created data folder ' + folder)
    for data in Data:
        if 'fa' in fields and 'fa' not in data:
            data['fa'] = data['fa_num_Tsp']

        df = pd.DataFrame()

        missing_fields = []
        for field in fields:
            try:
                df[field] = data[field].tolist()
            except KeyError as err:
                missing_fields.append(field)
        if len(missing_fields) > 0:
            print('missing fields ', ', '.join(missing_fields))

        filename = '_'.join(data[field] for field in filename_fields)
        df.to_csv(f"{folder}/{filename}.csv")

def load_data(folder : str, expnames = None) -> List[dict]:
    ''' Loads csv files from {folder} and return as list of dictionaries of ndarrays '''
    Data = []

    if expnames is None:
        filenames = os.listdir(folder)
    elif isinstance(expnames, str): # if expnames is a string treat it as a regex expression
        filenames = []
        for filename in os.listdir(folder):
            if re.search(expnames, filename) is not None:
                filenames.append(filename)
    elif isinstance(expnames, list):
        filenames = (expname + '.csv' for expname in expnames)
    else:
        raise NotImplementedError()
    for filename in filenames:
        # Ingore not csv files, assume csv files are in the right format
        if not filename.endswith('.csv'):
            continue

        # Load the csv using a pandas.DataFrame
        df = pd.read_csv(folder + '/' + filename)

        # Lists are loaded as strings by default, convert them back to lists
        for field in df.columns[1:]:
            if isinstance(df[field][0], str):
                df[field] = df[field].apply(literal_eval)

        # Copy all the data to a dictionary, and make things np.ndarrays
        Data.append({})
        for field in df.columns[1:]:
            Data[-1][field] = np.array(df[field].tolist(), dtype=float)

        # Add in some metadata from the filename
        namesplit = filename.split('.')[0].split('_')
        for i, field in enumerate(filename_fields):
            Data[-1][field] = namesplit[i]
        # Data[-1]['method'] = namesplit[0]
        # Data[-1]['condition'] = namesplit[1]

    return Data


SubDataset = namedtuple('SubDataset', 'X Y C meta')
feature_len = {}

def format_data(RawData: List[Dict['str', np.ndarray]], features: 'list[str]' = ['v', 'q', 'pwm'], output: str = 'fa', hover_pwm_ratio = 1.):
    ''' Returns a list of SubDataset's collated from RawData.

        RawData: list of dictionaries with keys of type str. For keys corresponding to data fields, the value should be type np.ndarray. 
        features: fields to collate into the SubDataset.X element
        output: field to copy into the SubDataset.Y element
        hover_pwm_ratio: (average pwm at hover for testing data drone) / (average pwm at hover for training data drone)
         '''
    Data = []
    for i, data in enumerate(RawData):
        # Create input array
        X = []
        for feature in features:
            if feature == 'pwm':
                X.append(data[feature] / 1000 * hover_pwm_ratio)
            else:
                X.append(data[feature])
            feature_len[feature] = len(data[feature][0])
        X = np.hstack(X)

        # Create label array
        Y = data[output]

        # Pseudo-label for cross-entropy
        C = i

        # Save to dataset
        Data.append(SubDataset(X, Y, C, {'method': data['method'], 'condition': data['condition'], 't': data['t']}))

    return Data

def plot_subdataset(data, features, title_prefix=''):
    fig, axs = plt.subplots(1, len(features)+1, figsize=(10,4))
    idx = 0
    for feature, ax in zip(features, axs):
        for j in range(feature_len[feature]):
            ax.plot(data.meta['t'], data.X[:, idx], label = f"{feature}_{j}")
            idx += 1
        ax.legend()
        ax.set_xlabel('time [s]')
    ax = axs[-1]
    ax.plot(data.meta['t'], data.Y)
    ax.legend(('fa_x', 'fa_y', 'fa_z'))
    ax.set_xlabel('time [s]')
    fig.suptitle(f"{title_prefix} {data.meta['condition']}: c={data.C}")
    fig.tight_layout()