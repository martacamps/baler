import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def diag(input_path):
    with open(input_path, "rb") as handle:
        input = pickle.load(handle)

    print(input)