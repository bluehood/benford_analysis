#!/usr/bin/python3
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy.special
from matplotlib import gridspec
import matplotlib.patches as mpatches
from matplotlib import ticker


def roundup(x):
    return(int(math.ceil(x / 10.0)) * 10)


def plot_heat_map(data, m):
    # increase font size
    plt.rcParams.update({'font.size': 13.5})

    data = np.round(data, decimals=1)
    print(data)

    #Setup x and y axis arrays

    if m[0:2] == '12':
        y_axis = np.arange(1, 11, 1)
        x_axis = np.arange(0, 10, 1)
    elif m[0:2] == '23':
        y_axis = np.arange(0, 11, 1)
        x_axis = np.arange(0, 11, 1)
    
    max = np.absolute(np.amax(data))
    min = np.absolute(np.amin(data))
    
    limit = round(np.maximum(max, min),1)
    
    test = [-limit, limit, limit, limit, limit, limit, limit, limit, limit, limit]
    array = np.asarray(test)
   
    values_to_plot = np.vstack((data, array))


    fig, ax = plt.subplots()

    im, cbar = heatmap(limit, values_to_plot, y_axis , x_axis, m, ax=ax,
                   cmap="coolwarm", cbarlabel="Deviation / 1000")
    texts = annotate_heatmap(im, valfmt="{x}")

    fig.tight_layout()
    print('[Debug] Saving Plot as {}'.format(sys.argv[3]))
    plt.savefig('{}'.format(sys.argv[3]), bbox_inches='tight')
    return(0)




# Credit for the next two functions https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html. These have been mildly edited 
# to suit the needs of the program. 

def heatmap(limit, data, row_labels, col_labels, mode, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # ticks = [0,5,10,15,20,25]
    ticks = [0,2,4,6,8,10]
    # number_of_ticks = 5
    # for x in range(1, number_of_ticks + 1):
    #     ticks.append(round(limit/number_of_ticks * x,1))
        
    for x in range(1, len(ticks)):
        ticks.append(-ticks[x])
    
    # print(ticks)
    # exit()


    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, ticks=ticks)
    cbar.ax.set_ylim(-limit, limit) 

    #cbar.clim(-60, 60)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    if mode[0:2] == '12':
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
    elif mode[0:2] == '23':
        ax.set_xticks(np.arange(data.shape[0]))
        ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    # print(col_labels)
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # ! Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # ! Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # ! Turn spines off and create white grid.
    #for edge, spine in ax.spines.items():
        #spine.set_visible(False)

    if mode[0:2] == '12':
        plt.xlabel("Second Digit")
        plt.ylabel("First Digit")

        ax.set_xticks(np.arange(data.shape[1]+1), minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1), minor=True)
        ax.set_xlim(row_labels[0] - 1.5, row_labels[-1] - 0.5)
        ax.set_ylim(col_labels[-1] - 0.5, col_labels[0] - 0.5)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
    elif mode[0:2] == '23':
        plt.xlabel("Third Digit")
        plt.ylabel("Second Digit")

        ax.set_xticks(np.arange(data.shape[1]+1), minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1), minor=True)
        ax.set_xlim(row_labels[0] - 0.5, row_labels[-1] - 0.5)
        ax.set_ylim(col_labels[-1] - 0.5, col_labels[0] - 0.5)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    # plt.xlabel("Second Digit")
    # plt.ylabel("First Digit")

    # ax.set_xticks(np.arange(data.shape[1]+1), minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1), minor=True)
    # ax.set_xlim(row_labels[0] - 1.5, row_labels[-1] - 0.5)
    # ax.set_ylim(col_labels[-1] - 0.5, col_labels[0] - 0.5)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0] - 1):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

    


def main():
    array_from_file = np.loadtxt(sys.argv[1], dtype=float)
    # print(array_from_file)

    mode = sys.argv[2]
    plot_heat_map(array_from_file, mode)


if __name__ == '__main__':
    main()

