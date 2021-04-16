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

def usage():
    print(f'Plot the variation of test statistics over different values of sigma (parameter controlling the strength of deviations introduced into Benford sets).\n')
    print(f'{sys.argv[0]} <List of comma delimited sigma values and test statistics> <Normalise boolean> <mode> <Plot save location>\n')
    print(f'<List of comma delimited test statistics> - filepath of computed test statistics to plot')
    print(f'<Normalise boolean> - true or false. If true all test statistics are normalised with respect to their .05 percent significance values.')
    print(f'<mode> - the mode of operation - first digit (1) or second digit (2).')
    print(f'<Plot save location> - filepath to save resultant plot. Must be a .png file.')
    return(0)

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


def main(base_set, normalise, mode):
    # import data from file
    data = input_numbers(base_set)
    

    # Define data
    x_axis = [str(float(x[0]) / 100) for x in data]
    # x_axis = [str(float(x[0])) for x in data]
    # print(x_axis)
    X_2 = [x[1] for x in data]
    A_2 = [x[2] for x in data]
    d_star = [x[3] for x in data]

    if normalise == 'true':
        if mode == '1':
            for x in range(0, len(X_2)):
                X_2[x] = X_2[x] / 1.938
                A_2[x] = A_2[x] / 2.84
                d_star[x] = d_star[x] / 1.35
        elif mode == '2':
            for x in range(0, len(X_2)):
                X_2[x] = X_2[x] / 1.938
                A_2[x] = A_2[x] / 2.61
                d_star[x] = d_star[x] / 1.32


    # Define font size
    plt.rcParams.update({'font.size': 15})

    # Plot data
    plt.figure(figsize=(8,6))
    # plt.plot(x_axis, X_2, 'r', linewidth=1, label=r"$\chi_v^2$")
    plt.plot(x_axis, A_2, 'b', linewidth=1, label=r"$A^2$")
    plt.plot(x_axis, d_star, 'r', linewidth=1, label=r"$d^*$")

    # Plot significance values at .05 value
    #     if A_squared >= 2.492 and A_squared < 3.88:

    if normalise != 'true':
        # plt.axhline(y=1.938, linewidth=0.75, color='r', linestyle='--')
        if mode == '1':
            plt.axhline(y=1.35, linewidth=0.75, color='r', linestyle='--')
            plt.axhline(y=2.84, linewidth=0.75, color='b', linestyle='--')
        elif mode == '2':
            plt.axhline(y=1.32, linewidth=0.75, color='r', linestyle='--')
            plt.axhline(y=2.61, linewidth=0.75, color='b', linestyle='--')

    elif normalise == 'true':
        plt.axhline(y=1, linewidth=0.75, color='black', linestyle='--')
        
        if mode == '1':
            plt.axhline(y=1.23, linewidth=0.75, color='r', linestyle='--')
            plt.axhline(y=1.61, linewidth=0.75, color='b', linestyle='--')
        elif mode == '2':
            plt.axhline(y=1.13, linewidth=0.75, color='r', linestyle='--')
            plt.axhline(y=1.52, linewidth=0.75, color='b', linestyle='--')

    plt.legend(loc='best')

    # Formatting of graph
    plt.xlim(0, 50)
    plt.ylim(0, 8)


    # set xticks
    # locs, labels = plt.xticks()
    # print(locs)
    # plt.xticks(np.arange(0,55,10))
    plt.xticks(np.arange(0,55,10))

    plt.xlabel(r"$\sigma$", fontsize=16)
    plt.ylabel("Normalised Test Statistics", fontsize=16)
    plt.savefig('{}'.format(sys.argv[4]), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as e:
        print(e)
        usage()
        exit()