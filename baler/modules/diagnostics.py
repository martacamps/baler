import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
import copy
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def get_nodes_numbers(input_dict: dict) -> np.array:
    nodes_numbers = np.array([0])
    for kk in input_dict:
        nodes_numbers = np.append(nodes_numbers, len(input_dict[kk].T))
    return nodes_numbers

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


def plot(data: np.array, nodes_numbers: np.array, output_path: str) -> None:
    fig, ax = plt.subplots()
    NAP = ax.imshow(
        data.T,
        cmap='RdBu_r',
        interpolation='nearest',
        aspect='auto',
        origin='lower',
        norm=matplotlib.colors.CenteredNorm()
    )
    colorbar = plt.colorbar(NAP)
    colorbar.set_label("Activation")
    ax.set_title("Neural Activation Pattern")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Number of nodes")
    xtick_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xtick_loc))
    ax.set_xticklabels(['','en1', 'en2', 'en3', 'de1', 'de2', 'de3', ''])
    ax.set_yticks(nodes_numbers)
    ax.figure.savefig(output_path + "diagnostics.pdf")
    


def diag(output_path, input_path, config) -> None:
    with open(input_path, "rb") as handle:
        input = pickle.load(handle)
    nodes_numbers = get_nodes_numbers(input)
    data = dict_to_square_matrix(get_mean_node_activations(input))
    plot(data, nodes_numbers, output_path)
  
        