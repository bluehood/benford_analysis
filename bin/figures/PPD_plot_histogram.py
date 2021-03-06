#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.stats as stats
import pylab as pl
import matplotlib.patches as mpatches

def usage():
    print(f'Plot histogram of a list of data provided in a local file. This program is designed specifically for price paid data.')
    print(f"Usage: {sys.argv[0]} <datafile> <plot_savefile>")
    return(0)

def standard_form(foo, round_too):
    
    foo_mag = math.floor(np.log10(foo))
    foo_report = r'{} \times 10^{}'.format(round(foo / 10**foo_mag, round_too), foo_mag)
    return(foo_report)

def main(open_file, save_file):
    usage()
    # import data
    data_to_plot = []
    f = open(open_file, "r")
    for x in f:
        x = x.replace('\n','')
        if x != '' and int(x) < 0.5 * 10 ** 6 :
        # if x != '' and int(x) < 0.5 * 10 ** 6 :
        #     if int(x) in [125000, 250000]:
        #         x = int(x) - 1

            data_to_plot.append(int(x))
    
    # Sort data and determine fit
    data_to_plot = sorted(data_to_plot)
    # fit = stats.norm.pdf(data_to_plot, np.mean(data_to_plot), np.std(data_to_plot))

    plt.rcParams.update({'font.size': 15})
    
    # Calcualte pdf and plot
    fig = plt.figure()
    ax = plt.subplot(111)
    # pl.plot(data_to_plot, fit,'--', linewidth=1.5, markersize=0.1, label="Normal Distribution")
    pl.xlabel(r'Price Paid / £ $\times 10^3$', size=15)
    pl.ylabel(r"Normalised Occurence $(10^{-6})$", size=15)
    pl.xlim(0, 500000)


    # Calculate Bins we define the binwidth. Define colour of the bins
    binwidth = 10000
    # bins_one = np.arange(0, 120000 + binwidth, binwidth)
    # threshold_one = np.array([125000, 130000])
    # total = np.concatenate((bins_one, threshold_one), axis=None)
    # bins_two = np.arange(140000, 240000 + binwidth, binwidth)
    # threshold_two = np.array([245000, 250000, 255000])
    # total = np.concatenate((total, bins_two, threshold_two), axis=None)
    # bins_three = np.arange(260000, max(data_to_plot) + binwidth, binwidth)
    # total = np.concatenate((total, bins_three), axis=None)
    # define_bins = total

    bins_one = np.arange(1, 120001 + binwidth, binwidth)
    threshold_one = np.array([125001, 130001])
    total = np.concatenate((bins_one, threshold_one), axis=None)
    bins_two = np.arange(140001, 240001 + binwidth, binwidth)
    threshold_two = np.array([245001, 250001, 255001])
    total = np.concatenate((total, bins_two, threshold_two), axis=None)
    bins_three = np.arange(260001, max(data_to_plot) + binwidth + 1, binwidth)
    total = np.concatenate((total, bins_three), axis=None)
    define_bins = total
    
    

    # Plot Histogram
    N, bins, patches = ax.hist(data_to_plot, bins = define_bins, density=True, color = "skyblue", ec="black", lw=1, label='PPD', alpha=0.7)
    
    print(N)

    # Set tick values
    ticks = np.arange(0, max(data_to_plot) + 50000, 50000)
    labels = []
    for x in ticks:
        labels.append(int(x/1000))
    pl.xticks(ticks, labels) 

    # remove y axis label units
    # ax.axes.yaxis.label.set_visible(False)

    for i in range(0, len(bins_one) - 1):
        patches[i].set_facecolor('skyblue')
    for i in range(len(bins_one) - 1, len(bins_one) + len(threshold_one) - 1):
        patches[i].set_facecolor('red')
    for i in range(len(bins_one) + len(threshold_one) - 1, len(bins_one) + len(threshold_one) + len(bins_two) - 1):
        patches[i].set_facecolor('skyblue')
    for i in range(len(bins_one) + len(threshold_one) + len(bins_two) - 1, len(bins_one) + len(threshold_one) + len(bins_two) + len(threshold_two) - 1):
        patches[i].set_facecolor('red')

    # standard form for legend
    mean = standard_form(np.mean(data_to_plot), 2)
    st_dev = standard_form(np.std(data_to_plot), 2)
    dataset_size = standard_form(len(data_to_plot), 1)
        
    # Format legend
    patch = []
    handles, labels = ax.get_legend_handles_labels()
    patch.append(mpatches.Patch(color='red', ec='black' , label="PPD Thresholds"))
    patch.append(mpatches.Patch(color='none', label=r"$\bar{x}" + r" = {:.2f}$".format(mean)))
    patch.append(mpatches.Patch(color='none', label=r"$\sigma = {:.2f}$".format(st_dev)))
    patch.append(mpatches.Patch(color='none', label=r"$N = {}$".format(dataset_size)))
    
    for x in patch:
        handles.append(x)
    
    plt.legend(handles = handles, loc='best')
    pl.savefig(f"{save_file}", bbox_inches='tight')
    pl.show()

    return(0)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])