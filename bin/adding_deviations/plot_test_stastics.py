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

def input_numbers(input_filename): 
    # Input data from argv[1] into input_data (newline delimited)
    try:
        open_file = open(input_filename, "r")
        input_data = open_file.readlines()  
        open_file.close()  
    except:
        print("[Fatal Error] Failed to read data from input file. Exiting.")
        exit()  

    print("[Debug] Input Data recieved from", input_filename)                       
    
    
    for x in range(0, len(input_data)):
        # if input_data != ['noise']:
        input_data[x] = input_data[x].split(',')[1:]
    
    for x in range(0, len(input_data)):
        try:
            if input_data[x][0][-1] == '.':
                input_data[x][0] = float(input_data[x][0][0:-1])
            input_data[x][0] = float(input_data[x][0])
            input_data[x][1] = float(input_data[x][1])
            input_data[x][2] = float(input_data[x][2])
            input_data[x][3] = float(input_data[x][3].replace('\n', ''))
        except:
            pass
    
    # Remove all null entries from the input data
    input_data = [x for x in input_data if x != []]
    

    return(input_data)


def main(base_set):
    # import data from file
    data = input_numbers(base_set)

    # Define data
    x_axis = [x[0] for x in data]
    X_2 = [x[1] for x in data]
    A_2 = [x[2] for x in data]
    d_star = [x[3] for x in data]


    # Define font size
    plt.rcParams.update({'font.size': 12})

    # Plot data
    plt.figure(figsize=(8,6))
    plt.plot(x_axis, X_2, 'r', linewidth=1, label=r"$\chi_v^2$")
    plt.plot(x_axis, A_2, 'b', linewidth=1, label=r"$A^2$")
    plt.plot(x_axis, d_star, 'g', linewidth=1, label=r"$d^*$")

    # Plot significance values at .05 value
    #     if A_squared >= 2.492 and A_squared < 3.88:

    plt.axhline(y=1.330, linewidth=0.75, color='g', linestyle='--')
    plt.axhline(y=2.492, linewidth=0.75, color='b', linestyle='--')

    plt.legend(loc='best')

    # Formatting of graph
    plt.xlim(0, x_axis[-1])
    plt.ylim(0, max(max(X_2), max(d_star)))
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])