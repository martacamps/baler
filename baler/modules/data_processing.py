import json
import torch
import uproot
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import modules.models as models


def import_config(config_path):
    with open(config_path, encoding='utf-8') as json_config:
        config = json.load(json_config)
    return config


def save_model(model, model_path):
    return torch.save(model.state_dict(), model_path)


def initialise_model(config):
    model_name = config["model_name"]
    ModelObject = getattr(models, model_name)
    return ModelObject


def load_model(ModelObject, model_path, n_features, z_dim):
    model = ModelObject(n_features, z_dim)

    # Loading the state_dict into the model
    model.load_state_dict(torch.load(str(model_path)), strict=False)
    return model


def Type_clearing(TTree):
    typenames = TTree.typenames()
    column_type = []
    column_names = []

    # In order to remove non integers or -floats in the TTree,
    # we separate the values and keys
    for keys in typenames:
        column_type.append(typenames[keys])
        column_names.append(keys)

    # Checks each value of the typename values to see if it isn't an int or
    # float, and then removes it
    for i in range(len(column_type)):
        if column_type[i] != 'float[]' and column_type[i] != 'int32_t[]':
            #print("Index ",i," was of type ",Typename_list_values[i]," and was deleted from the file")
            del column_names[i]

    # Returns list of column names to use in load_data function
    return column_names


def numpy_to_df(array):
    df = pd.DataFrame(array, columns=CLEARED_COLUMN_NAMES)
    return df


def load_data(data_path, config):
    path = Path(data_path)
    file_extension = path.suffix

    if file_extension in [".csv"]:
        data_file = pd.read_csv(data_path, low_memory=False)
    elif file_extension in [".root"]:
        root_file = uproot.open(data_path)
        tree = root_file[config["Branch"]][config["Collection"]][config["Objects"]]
        names = Type_clearing(tree)
        data_file = tree.arrays(names, library="pd")
    elif file_extension in [".pickle"]:
        data_file = pd.read_pickle(data_path)
    elif file_extension in [".hdf", ".h4", ".hdf4", ".he2", ".h5", ".hdf5", ".he5"]:

        import h5py
        hdf = h5py.File(data_path, "r")
        print(list(hdf.keys()))
        
        data_file = pd.read_hdf(data_path)
    else:
        raise Exception(f"Unsupported file type: {file_extension}")

    return data_file


def clean_data(df, config):
    df = df.drop(columns=config["dropped_variables"])
    df = df.dropna()
    global CLEARED_COLUMN_NAMES
    CLEARED_COLUMN_NAMES = list(df)
    return df


def find_minmax(data):
    data = np.array(data)
    data = list(data)
    true_max_list = np.apply_along_axis(np.max, axis=0, arr=data)
    true_min_list = np.apply_along_axis(np.min, axis=0, arr=data)

    feature_range_list = true_max_list - true_min_list

    normalization_features = \
        pd.DataFrame({'True min': true_min_list,
                      'Feature Range': feature_range_list})
    return normalization_features


def normalize(data, config):
    data = np.array(data)
    if config["custom_norm"]:
        pass
    elif config["custom_norm"]:
        true_min = np.min(data)
        true_max = np.max(data)
        feature_range = true_max - true_min
        data = [((i - true_min)/feature_range) for i in data]
        data = np.array(data)
    return data


def split(df, test_size, random_state):
    return train_test_split(df, test_size=test_size, random_state=random_state)


def renormalize_std(data, true_min, feature_range):
    data = list(data)
    data = [((i * feature_range) + true_min) for i in data]
    data = np.array(data)
    return data


def renormalize_func(norm_data, min_list, range_list):
    norm_data = np.array(norm_data)    
    renormalized = [renormalize_std(norm_data, min_list[ii], range_list[ii]) for ii in range(len(min_list))]
    renormalized_full = [(renormalized[ii][:, ii]) for ii in range(len(renormalized))]
    renormalized_full = np.array(renormalized_full).T
    #renormalized_full = pd.DataFrame(renormalized_full,columns=config["CLEARED_COLUMN_NAMES"])

    return renormalized_full


def get_columns(df):
    return list(df.columns)
