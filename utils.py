import os
from typing import List
from ast import literal_eval
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder = './data/experiment'

def save_data(Data : List[dict], folder : str, fields=['t', 'p', 'p_d', 'v', 'v_d', 'q', 'R', 'w', 'T_sp', 'q_sp', 'hover_throttle', 'fa', 'pwm']):
    ''' Save {Data} to individual csv files in {folder}, serializing (2+)d ndarrays as lists '''
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print('Created data folder ' + folder)
    for data in Data:
        if 'fa' in fields and 'fa' not in data:
            data['fa'] = data['fa_num_Tsp']

        df = pd.DataFrame()

        for field in fields:
            df[field] = data[field].tolist()
        df.to_csv(f"{folder}/{data['method']}_{data['condition']}.csv")

def load_data(folder : str, expnames : List[str] = None) -> List[dict]:
    ''' Loads csv files from {folder} and return as list of dictionaries of ndarrays '''
    Data = []

    if expnames is None:
        filenames = os.listdir(folder)
    else:
        filenames = (expname + '.csv' for expname in expnames)
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
        Data[-1]['method'] = namesplit[0]
        Data[-1]['condition'] = namesplit[1]

    return Data


SubDataset = namedtuple('SubDataset', 'X Y C meta')
feature_len = {}

def format_data(RawData, features: 'list[str]' = ['v', 'q', 'pwm'], output: str = 'fa'):
    Data = []
    for i, data in enumerate(RawData):
        # Create input array
        X = []
        for feature in features:
            if feature == 'pwm':
                X.append(data[feature] / 1000)
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