import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab as pl
import math

def usage():
    print(f'Display scatter plot of a list of data in a file.')
    print(f'{sys.argv[0]} <datafile>\n')
    return(0)

def main(filename):
    # output_file = ""
    
 # Import and sanitise data
    f = open(filename, "r")
    seperate_data_sets = False
    x = []
    y = []

    for element in f:
        element = element.replace('\n', '')
        if element == '':
            seperate_data_sets = True
            continue

        if seperate_data_sets == False:
            if "." in element:
                x.append(float(element))
            else:
                print("False")
                x.append(int(element))

            continue

        elif seperate_data_sets == True:
            if "." in element:
                print("True")
                y.append(float(element))
            else:
                y.append(int(element))

            continue
    f.close()

    # Set x to a log scale
    x_log = np.log10(x)
    
    # yerrors
    yerror = []
    for j in range(0, len(x)):
        yerror.append(1 / math.sqrt(x[j]))

    # Plot graph
    pl.errorbar(x_log, y, yerr=yerror, label="Observed d* values", color='black', marker='x', fmt='x', capsize=3, elinewidth=1, zorder=1)
    pl.show()
    return(0)


if __name__ == '__main__':
    usage()
    main(sys.argv[1])
    