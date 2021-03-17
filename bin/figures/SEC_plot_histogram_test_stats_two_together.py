#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pylab as pl
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

def usage():
    print(f'Plot two lists of test statistics in one histogram to compare distributions. The columns in each file must be formated as:\n <chi_squared>, <d*>, <A squared>.')
    print(f'{sys.argv[0]} <filename list one> <filename list two> <directory for produced plot> <mode> <number of bins> <include plot inset (true or false)>\n')
    print(f'<mode>: \n 0 - Chi-Squared\n 1 - d*\n 2 - A Squared\n')

    return(0)

def import_test_stats(filename):
    chi_squared_local = []
    d_star_local = []
    a_squared_local = []

    # import data
    f = open(filename, "r")

    # Split data into three arrays corresponding to the three test statistics. 
    for x in f:
        try:
            x = x.replace('\n','')
            chi_squared_local.append(float(x.split(',')[0]))
            d_star_local.append(float(x.split(',')[2]))
            a_squared_local.append(float(x.split(',')[1]))
        except:
            print(x)

    return(chi_squared_local, d_star_local, a_squared_local)


def main(original_data_file, modified_data_file, save_file):
    chi_squared_original = []
    d_star_original = []
    a_squared_original = []
    chi_squared_modified = []
    d_star_modified = []
    a_squared_modified = []
    data_to_plot_original = []
    data_to_plot_modified = []

    # import original data 
    chi_squared_original, d_star_original, a_squared_original = import_test_stats(original_data_file)

    # import modified data
    chi_squared_modified, d_star_modified, a_squared_modified = import_test_stats(modified_data_file)
 
    j = int(sys.argv[4])
    
    # First Digit Histograms
    if j == 0:
        data_to_plot_original = chi_squared_original
        data_to_plot_modified = chi_squared_modified
        x_label = r'$\chi^{2}$'
        save_file = save_file + '/chi_squared.png'
        # set binwidth
        
        
    elif j == 1:
        data_to_plot_original = d_star_original
        data_to_plot_modified = d_star_modified
        x_label = r'$d^{*}$'
        save_file = save_file + '/d_star.png'
        
        
        
    elif j == 2:
        data_to_plot_original = a_squared_original
        data_to_plot_modified = a_squared_modified
        x_label = r'$A^2$'
        save_file = save_file + '/a_squared.png'
        

    # # Second Digit histograms
    # if j == 20:
    #     data_to_plot = chi_squared
    #     x_label = r'$\chi^{2}$'
    #     save_file = save_file + '/chi_squared.png'
    #     # set binwidth
        
        
    # elif j == 21:
    #     data_to_plot = d_star
    #     x_label = r'$d^{*}$'
    #     save_file = save_file + '/d_star.png'
        
        
        
    # elif j == 22:
    #     data_to_plot = a_squared
    #     x_label = r'$A^2$'
    #     save_file = save_file + '/a_squared.png'

    
    
    # Sort data and determine fit
    data_to_plot_original = sorted(data_to_plot_original)
    data_to_plot_modified = sorted(data_to_plot_modified)
    
    # fit = stats.norm.pdf(data_to_plot, np.mean(data_to_plot), np.std(data_to_plot))

    # Calculate bin width. - 10 bins each time. Take the minimum of original and moodified data 
    number_of_bins = int(sys.argv[5])
    individual_bin_width = min((np.ceil(max(data_to_plot_original)) - np.floor(min(data_to_plot_original))) / number_of_bins,  np.ceil(max(data_to_plot_modified)) - np.floor(min(data_to_plot_modified)) / number_of_bins)
    bin_width = np.arange( min(np.floor(min(data_to_plot_original)), np.floor(min(data_to_plot_modified))), max(np.ceil(max(data_to_plot_original)), np.ceil(max(data_to_plot_modified))) + individual_bin_width, individual_bin_width)
    # print(bin_width)
    
    
    # Plot on two datasets in a histogram on top of one another
    plt.rcParams.update({'font.size': 12.5})
    fig = plt.figure()
    ax = plt.subplot(111)
    # pl.plot(data_to_plot, fit,'--', linewidth=1.5, markersize=0.1, label="Normal Distribution")
    pl.xlabel(x_label, size=12.5)
    pl.ylabel("Normalised Occurence", size=12.5)
    pl.xlim(np.floor(min(data_to_plot_original[0], data_to_plot_modified[0])), np.ceil(max(data_to_plot_original[-1], data_to_plot_modified[-1])))


    # Calculate Bins we define the binwidth. Define colour of the bins
    # binwidth = 10000
    # bins_one = np.arange(0, 120000 + binwidth, binwidth)
    # threshold_one = np.array([125000, 130000])
    # total = np.concatenate((bins_one, threshold_one), axis=None)
    # bins_two = np.arange(140000, 240000 + binwidth, binwidth)
    # threshold_two = np.array([245000, 250000, 255000])
    # total = np.concatenate((total, bins_two, threshold_two), axis=None)
    # bins_three = np.arange(260000, max(data_to_plot) + binwidth, binwidth)
    # total = np.concatenate((total, bins_three), axis=None)
    # define_bins = total

    # bins_one = np.arange(1, 120001 + binwidth, binwidth)
    # threshold_one = np.array([125001, 130001])
    # total = np.concatenate((bins_one, threshold_one), axis=None)
    # bins_two = np.arange(140001, 240001 + binwidth, binwidth)
    # threshold_two = np.array([245001, 250001, 255001])
    # total = np.concatenate((total, bins_two, threshold_two), axis=None)
    # bins_three = np.arange(260001, max(data_to_plot) + binwidth + 1, binwidth)
    # total = np.concatenate((total, bins_three), axis=None)
    # define_bins = total

    
    

    # Plot Histogram
    # N, bins, patches = ax.hist(data_to_plot, bins = define_bins, density=True, color = "skyblue", ec="black", lw=1, label='PPD', alpha=0.7)
    N, bins, patches = ax.hist(data_to_plot_original, bins = bin_width, density=True, color = "skyblue", ec="black", lw=1, label=f'Total Dataset', alpha=0.5)
    # N, bins, patches = ax.hist(data_to_plot, density=True, color = "skyblue", ec="black", lw=1, label=x_label, alpha=0.7)

    N_2, bins_2, patches_2 = ax.hist(data_to_plot_modified, bins = bin_width, density=True, color = "indianred", ec="black", lw=1, label=f'Consolidated Financial Data', alpha=0.4)


    

    # Format legend    
    patch = []
    handles_altered = []
    handles, labels = ax.get_legend_handles_labels()

    if j == 0:
        patch.append(mpatches.Patch(color='none', label=r"$\bar{\chi}^2_{\nu, T}$" + " = {:.1f}, $\sigma_T$ = {:.1f}, $N_T$ = {}".format(np.mean(data_to_plot_original), np.std(data_to_plot_original), len(data_to_plot_original))))
    elif j == 1:
        patch.append(mpatches.Patch(color='none', label=r"$\bar{d}^{*}_{T}$" + " = {:.1f}, $\sigma_T$ = {:.1f}, $N_T$ = {}".format(np.mean(data_to_plot_original), np.std(data_to_plot_original), len(data_to_plot_original))))
    elif j == 2:
        patch.append(mpatches.Patch(color='none', label=r"$\bar{A}^{2}_T$" + " = {:.1f}, $\sigma_T$ = {:.1f}, $N_T$ = {}".format(np.mean(data_to_plot_original), np.std(data_to_plot_original), len(data_to_plot_original))))
    
    # patch.append(mpatches.Patch(color='none', label=r"N = {}".format(len(data_to_plot_original))))

    handles_altered.append(handles[0])
    for x in patch:
        handles_altered.append(x)

    patch = []
    if j == 0:
        patch.append(mpatches.Patch(color='none', label=r"$\bar{\chi}^2_{\nu, C}$" + " = {:.1f}, $\sigma_C$ = {:.1f}, $N_C$ = {}".format(np.mean(data_to_plot_modified), np.std(data_to_plot_modified), len(data_to_plot_modified))))
    elif j == 1:
        patch.append(mpatches.Patch(color='none', label=r"$\bar{d}^{*}_C$" + " = {:.1f}, $\sigma_C$ = {:.1f}, $N_C$ = {}".format(np.mean(data_to_plot_modified), np.std(data_to_plot_modified), len(data_to_plot_modified))))
    elif j == 2:
        patch.append(mpatches.Patch(color='none', label=r"$\bar{A}^{2}_C$" + " = {:.1f}, $\sigma_C$ = {:.1f}, $N_C$ = {}".format(np.mean(data_to_plot_modified), np.std(data_to_plot_modified), len(data_to_plot_modified))))
    
    #  patch.append(mpatches.Patch(color='none', label=r"N = {}".format(len(data_to_plot_modified))))
    
    handles_altered.append(handles[1])
    for x in patch:
        handles_altered.append(x)
    
    plt.legend(handles = handles_altered, loc='upper right')
    plt.tight_layout()

     # Set y_ticks
    locs, labels = plt.yticks()
    y_interval = (max(locs) - min(locs)) / 5
    pl.yticks(np.arange(min(locs), max(locs) + y_interval, y_interval).round(2))
    pl.ylim(min(locs), 0.25)
    

    # Inset if applicable
    if sys.argv[6] == 'true':
        # Define inset data
        maximum = max(data_to_plot_original[-1], data_to_plot_modified[-1]) / 5
        data_to_plot_original_inset = []
        data_to_plot_modified_inset = []

        x = 0
        try:
            while data_to_plot_original[x] <= maximum:
                data_to_plot_original_inset.append(data_to_plot_original[x])
                x += 1
        except:
            data_to_plot_original_inset = data_to_plot_original

        try:
            x = 0
            while data_to_plot_modified[x] <= maximum:
                data_to_plot_modified_inset.append(data_to_plot_modified[x])
                x += 1
        except:
            data_to_plot_modified_inset = data_to_plot_modified

        print(data_to_plot_modified, data_to_plot_original)

        # Set position and size of inset 
        ax2 = plt.axes([0,0,1,1])
        # d*
        # ip = InsetPosition(ax, [0.52,0.17,0.43,0.43])
        ip = InsetPosition(ax, [0.45,0.1,0.5,0.5])
        ax2.set_axes_locator(ip)
        

        # Mark the region corresponding to the inset axes on ax1 and draw lines
        # in grey linking the two axes.
        # mark_inset(ax, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

        # Calculate bin width.. Take the minimum of original and moodified data 
        number_of_bins = 30
        individual_bin_width = min((np.ceil(max(data_to_plot_original_inset)) - np.floor(min(data_to_plot_original_inset))) / number_of_bins,  np.ceil(max(data_to_plot_modified_inset)) - np.floor(min(data_to_plot_modified_inset)) / number_of_bins)
        bin_width = np.arange( min(np.floor(min(data_to_plot_original)), np.floor(min(data_to_plot_modified))), max(np.ceil(max(data_to_plot_original)), np.ceil(max(data_to_plot_modified))) + individual_bin_width, individual_bin_width)
        # print(bin_width)

        #Plot data below a maximum
        
        ax2.hist(data_to_plot_original, bins = bin_width, density=True, color = "skyblue", ec="black", lw=1, label=f'Total Dataset', alpha=0.5)
        ax2.hist(data_to_plot_modified, bins = bin_width, density=True, color = "indianred", ec="black", lw=1, label=f'Modified Dataset', alpha=0.4)
        
        plt.xlim(0, maximum)
    # N, bins, patches = ax.hist(data_to_plot, density=True, color = "skyblue", ec="black", lw=1, label=x_label, alpha=0.7)

    #  ax.hist(data_to_plot_modified, bins = bin_width, density=True, color = "indianred", ec="black", lw=1, label=f'Consolidated Financial Data', alpha=0.4)





    # plt.show()
    # exit()
    pl.savefig(f"{save_file}", bbox_inches='tight' )
    # pl.savefig(f'{save_file}')


    return(0)


if __name__ == '__main__':
    usage()
    main(sys.argv[1], sys.argv[2], sys.argv[3])