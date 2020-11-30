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
from subprocess import call


def usage():
    print(sys.argv[0], "file_to_test", "mode", "figure_filename", "lower_limit", "upper_limit")
    print("\nModes:\n 1    Second Digit Finite Range")
    return(0)



### --------------------------------------- Import Data --------------------------------------------- ###

def input_data_from_file(input_filename): 
    # Input data from argv[1] into input_data (newline delimited)
    try:
        open_file = open(input_filename, "r")
        input_data = open_file.readlines()  
        open_file.close()  
    except:
        print("[Fatal Error] Failed to read data from input file. Exiting.")
        usage()
        exit()  

    print("[Debug] Input Data recieved from ", input_filename)                       

    # Remove all null entries from the input data
    input_data = ' '.join(input_data).split()   

    #Remove first entry of data if this entry contains a label for a column.
    try:
        int(input_data[0])
    except:
        del input_data[0]

    #Remove all null values and leading zeros. Add trailing zero to any entry of length 1. Save this result in input_data_sanitised
    print("[Debug] Sanitising Input Data")

    try:
        input_data.remove('0')
        input_data.remove('0.')
    except:
        pass

    input_data_sanitised = input_data

    for x in range(0, len(input_data)):
        input_data_sanitised[x] = '{:.3f}'.format(float(input_data[x]))
        # while input_data_sanitised[x][0] == "0":
        #         input_data_sanitised[x] = input_data_sanitised[x][1:]
        #         if input_data_sanitised[x] == '':
        #             break
        # if len(input_data_sanitised[x]) == 1:
        #     input_data_sanitised[x] = input_data_sanitised[x] + "0"

    # Remove all null entries from input_data_sanitised
    input_data_sanitised = ' '.join(input_data_sanitised).split()
    print("[Debug] Input Data Sanitised Successfully")  
    return(input_data_sanitised) 




### --------------------------------------- SECOND DIGIT TEST --------------------------------------------- ###



#Output the results of first_digit_test to file argv[2]
def output_second_digit_test(digit_occurance, benford_occurance, z_stat):
    #Round output figures to 3.d.p. Output relevant entries as percentages.
    digit_frequency = []
    benford_frequency = []
    for x in range(0,len(digit_occurance)):
        digit_frequency.append(str(int(digit_occurance[x])))
        benford_frequency.append(str(int(benford_occurance[x])))
        z_stat[x] = '{:.3f}'.format(z_stat[x])

    #Identify significant deviations based on Z statistic. 
    for x in range(0, len(z_stat)):
        if float(z_stat[x]) >= 1.96 and float(z_stat[x]) < 2.576:
            z_stat[x] = z_stat[x] + " *"
        elif float(z_stat[x]) >= 2.576:
            z_stat[x] = z_stat[x] + " **"
    
    print("Digit        Expected Distribution Occurance        Observed Distribution Occurance        Z-Statistic")
    print("-----------------------------------------------------------------------------------------------")

    #write results to the file with table formats. 
    for x in range(0,len(z_stat)):
        line = ""
        line = str(x + 1) + '&' + " " * (8 + len("Digit") - len(str(x + 1)) - 1)
        line += benford_frequency[x] + '&' + " " * (8 + len("Expected Distribution Occurance") - len(benford_frequency[x]) - 1)
        line += '{:.0f}'.format(float(digit_frequency[x])) + '&' + " " * (8 + len("Observed Distribution Occurance") - len('{:.0f}'.format(float(digit_frequency[x]))) - 1)
        line += z_stat[x] + "\\\\"
        print(line)

    #Calaculate the sum of Benford Distribution - ensure it equals 100!
    print("")
    expected_sum = 0

    for x in range(0, len(digit_frequency)):
        expected_sum += float(benford_frequency[x])

    line = " " * (8 + len("Digit")) + '{:.1f}'.format(expected_sum) + "\n"
    print(line)
    return(0)


def second_digit_analysis(set_to_test):
    occurence = [0] * 10

    for x in range(0, len(set_to_test)):
        try:
            occurence[int(set_to_test[x][1])] += 1
        except:
            pass
    
    return(occurence)

def second_digit_test(input_data):
    #Calculate the frequency of each character {0,1,2,...,9} in the second digit. 
    print("[Debug] Calculating first digit frequency")
    digit_frequency = [0] * 10
    first_digit = 0
    for x in input_data:
        try:
            first_digit = int(x[1])
        except:
            #account for single digit values
            continue
        digit_frequency[first_digit] += 1

    #Convert frequncies to percentage expressed as a decimal. 
    print("[Debug] Converting to percentages")
    digit_frequency_percent = [0] * 10

    for x in range(0, len(digit_frequency)):
        digit_frequency_percent[x] = float(digit_frequency[x] / len(input_data))

    #Calcuate perfect Benford distribution.
    print("[Debug] Computing ideal Benford frequency")
    benford_frequency_percent = benford_distribution(3, 0)
    
    #Compute Benford distribution for data of length equal to dataset
    benford_frequency = []

    for x in benford_frequency_percent:
        benford_frequency.append(round(x * len(input_data)))

    #Compute Z statistic for this data:
    print("[Debug] Computing Z statistic")
    z_stat = []
    for x in range(0, len(digit_frequency)):
        z_stat.append(compute_z_statistic(digit_frequency_percent[x], benford_frequency_percent[x], len(input_data)))

    #Compute von-mises statistics
    von_mises_stat = compute_von_mises(benford_frequency, digit_frequency, benford_frequency_percent, len(input_data))

    #Compute d* statistic
    d_star_stat = compute_dstar(digit_frequency_percent, benford_frequency_percent, len(input_data))

    return(digit_frequency, benford_frequency, digit_frequency_percent, benford_frequency_percent, z_stat, von_mises_stat, d_star_stat)




### --------------------------------------- Generate Synthetic Benford Set --------------------------------------------- ###


def generate_benford_set_from_c_program(lower, upper, size_set):
    call(["generate_benford", "/tmp/generate_benford_output.txt", str(size_set), str(lower), str(upper)])
    return(0)

def import_process_benford_set(size_set):
    filename = "/tmp/generate_benford_output.txt"
    benford_set_raw = []
    benford_set_counts = [0] * 10
    # open the file for reading
    filehandle = open(filename, 'r')
    i = 0
    while True:
        # read a single line
        line = filehandle.readline()
        benford_set_raw.append(line)

        if i % 10000 == 0:
            observed_counts = second_digit_analysis(benford_set_raw)
            for x in range(0, len(benford_set_counts)):
                benford_set_counts[x] += observed_counts[x]

            benford_set_raw = []
        
        if i == size_set:
            observed_counts = second_digit_analysis(benford_set_raw)
            for x in range(0, len(benford_set_counts)):
                benford_set_counts[x] += observed_counts[x]

            break
        
        i += 1
   
    
    filehandle.close()

    print(benford_set_counts)
    total = 0
    for x in benford_set_counts:
        total += x
    print("total", total)
    
    return(0)


### --------------------------------------- Cut data in specific range. --------------------------------------------- ###

def cut_data_range(lowerbound, upperbound, dataset):
    #Compute data in finite range
    dataset_refined = []
    for x in range(0, len(dataset)):
        if float(dataset[x]) >= lowerbound and float(dataset[x]) <= upperbound:
            dataset_refined.append(dataset[x])

    # #Calculate the frequency of each digit
    # digit_frequency_observed = [0] * 9

    # for x in dataset_refined:
    #     digit_frequency_observed[int(str(x)[0]) - 1] += 1

    
    return(dataset_refined)



### --------------------------------------- Main --------------------------------------------- ###


def main(mode):
    # Import test data
    filename = sys.argv[1]
    test_data = input_data_from_file(filename)
    print(f"[Debug] Imported test set from {filename}")
    print(f"[Debug] Reduce to range specified by the user.")
    
    # Select data in specific range 
    test_data = cut_data_range(float(sys.argv[4]), float(sys.argv[5]), test_data)

    print("[Debug] Calculating lower and upper magnitude of the test set")
    # Calculate min and max of test_data
    float_test_data = sorted(test_data, key=float)
    try:
        float_test_data.remove('0.0000')
    except:
        pass
    
    # normalise wrt to c program output
    test_data_normalised = [int(i.replace('.','')) for i in float_test_data]
    
    # calculate lower/upper limit of normalised set
    lowerlimit = int(test_data_normalised[0])
    upperlimit = float(test_data_normalised[-1])

    print(f"{lowerlimit} {upperlimit}")

    #convert to standard form
    lower = '{:e}'.format(lowerlimit)
    upper = '{:e}'.format(upperlimit)

    #calculate lower magnitude and upper magnitude
    lower_mag = int(str(lower).split('e')[1])
    upper_mag = int(str(upper).split('e')[1])

    upper_mag += 1
    lower_mag += 1

    # generate synthetic Benford set
    size = 100000
    print("[Debug] Generating Benford set. This could take a while zzz")
    generate_benford_set_from_c_program(lower_mag, upper_mag, size)
    import_process_benford_set(size)



    return(0)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        usage()
        exit()
    else:
        main(int(sys.argv[2]))