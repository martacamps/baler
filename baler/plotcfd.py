from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


pickle_file = "../data/cfd/cfd.pickle"

decompressed_cfd = "../projects/ConvAEDes/decompressed_output/decompressed.pickle"


def pickle_to_df(file):
    # From pickle to df:
    with open(file, "rb") as handle:
        data = pickle.load(handle)

        return data


data = pickle_to_df(pickle_file)
data = pd.DataFrame(data).iloc[51:,50:100]
data = data.astype(dtype="float32")

data_decompressed = pickle_to_df(decompressed_cfd)
data_decompressed = pd.DataFrame(data_decompressed.reshape((50,50)))
data_decompressed = data_decompressed.astype(dtype="float32")

diff = data_decompressed - data

fig, axs = plt.subplots(3, sharex=True, sharey=True)
axs[0].set_title("Original", fontsize=11)
im1 = axs[0].imshow(data, cmap="CMRmap", interpolation="nearest")
plt.ylim(0, 50)
divider_1 = make_axes_locatable(axs[0])
cax_1 = divider_1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax_1, shrink=0.1)

axs[1].set_title("Decompressed", fontsize=11)
im2 = axs[1].imshow(data_decompressed, cmap="CMRmap", interpolation="nearest")
plt.ylim(0, 50)
divider_2 = make_axes_locatable(axs[1])
cax_2 = divider_2.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax_2, shrink=0.1)

axs[2].set_title("Decompressed - Original", fontsize=11)
im3 = axs[2].imshow(diff, cmap="cool_r", interpolation="nearest")
plt.ylim(0, 50)
divider_3 = make_axes_locatable(axs[2])
cax_3 = divider_3.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax_3, shrink=0.1)
fig.savefig('../projects/ConvAEDes/decompressed_output/CFD_plot_500EP_new.png')
exit()
