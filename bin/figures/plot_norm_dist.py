import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.patches as mpatches


def main(file_to_add, savelocation):
    # setup variables 
    # import d* values and add to figure
    # import and sanitise data
    entries = []
    f = open(file_to_add, "r")
        
    for x in f:
        x = x.replace('\n', '')
        try:
            if "." in x:
                entries.append(float(str(x)))
            else:
                entries.append(int(str(x)))
        except:
            if " " in x or "£" in x or x == '':
                continue
            try:
                entries.append(int(str(x).replace(" ", "")))
            except:
                continue
            
    # Sort data and plot using pylab
    entries = sorted(entries)
    fit = stats.norm.pdf(entries, np.mean(entries), np.std(entries))

    # Update font size
    plt.rcParams.update({'font.size': 11})
    
    # Calcualte pdf and plot
    fig = plt.figure()
    ax = plt.subplot(111)
    pl.plot(entries, fit,'-o', linewidth=2, markersize=5, label="Normal Distribution")
    pl.xlabel(r'$d^*$', size=12)
    pl.ylabel("Normalised Probability", size=12)

    # Calculate Bins we define the binwidth here
    binwidth = 0.05 
    define_bins = np.arange(min(entries), max(entries) + binwidth, binwidth)
    
    # Plot Histogram
    pl.hist(entries, bins = define_bins, density=True, color = "skyblue", ec="black", lw=1, label=r'$d^*$ value')

    # Format legend
    patch = []
    handles, labels = ax.get_legend_handles_labels()
    patch.append(mpatches.Patch(color='none', label=r"$\bar{x}$ = " + "{:.3f}".format(np.mean(entries))))
    patch.append(mpatches.Patch(color='none', label=r"$\sigma$ = {:.3f}".format(np.std(entries))))
    patch.append(mpatches.Patch(color='none', label=r"N= {}".format(len(entries))))
    
    for x in patch:
        handles.append(x)
    
    plt.legend(handles = handles, loc='best')

    # Save result to sys.argv[2]
    pl.savefig(f"{savelocation}", bbox_inches='tight' ) 

if __name__ == '__main__':
    # try:
    main(sys.argv[1], sys.argv[2])
    # except:
    #     print(f"Usage: {sys.argv[0]} <datafile> <savefile>")
    #     exit()