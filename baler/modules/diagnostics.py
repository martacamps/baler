import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.colors
import copy
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# class MidpointNormalize(matplotlib.colors.Normalize):
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False) -> None:
#         self.midpoint = midpoint
#         matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

#     def __call__(self, value, clip= None):
#         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(value, x, y))


def dict_to_square_matrix(input_dict: dict) -> np.array:
    max_number_of_nodes = 0
    number_of_layers = len(input_dict)
    for kk in input_dict:
        if len(input_dict[kk]) > max_number_of_nodes:
            max_number_of_nodes = len(input_dict[kk])
    square_matrix = np.empty((number_of_layers, max_number_of_nodes))
    counter = 0
    for kk in input_dict:
        layer = np.array(input_dict[kk])
        if len(layer) == max_number_of_nodes:
            square_matrix[counter] = layer
        else:
            layer = np.append(layer, np.zeros(max_number_of_nodes - len(layer)) + np.nan)
            square_matrix[counter] = layer
        counter += 1
    return square_matrix

def get_mean_node_activations(input_dict: dict) -> dict:
    output_dict = {}
    for kk in input_dict:
        output_dict_layer = []
        for node in input_dict[kk].T:
            output_dict_layer.append(torch.mean(node).item())
        output_dict[kk] = output_dict_layer
    return output_dict


def plot(data: np.array, output_path: str) -> None:
    midpoint_value = abs(np.min(data))/(abs(np.max(data)) + abs(np.min(data)))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["navy", "white", "firebrick"])
    # norm = MidpointNormalize(midpoint=0)

    NAP = plt.imshow(
        data.T,
        cmap=cmap,
        interpolation='nearest',
        aspect='auto',
        origin='lower',
        # norm=norm
    )
    plt.colorbar(NAP)
    plt.title('NAP diagram')
    plt.savefig(output_path + "diagnostics.pdf")
    


def diag(output_path, input_path, config) -> None:
    with open(input_path, "rb") as handle:
        input = pickle.load(handle)
    data = dict_to_square_matrix(get_mean_node_activations(input))
    plot(data, output_path)
  
        