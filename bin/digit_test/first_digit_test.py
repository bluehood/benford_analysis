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


def finite_range_benford_distribution(mode, data):
    #Finite range calculation
    if mode == -1:
        P = []
        #First digit finite range analysis
        digit_observed, size = refine_first_digit_finite_range(int(lowerlimit), int(upperlimit), data)
        #convert to standard form
        lower = '{:e}'.format(lowerlimit)
        upper = '{:e}'.format(upperlimit)

        #calculate a,b,alpha,beta
        LowerLimit = str(lower).split('e')
        UpperLimit = str(upper).split('e')

        a = float(LowerLimit[0])
        alpha = int(LowerLimit[1])
        b = float(UpperLimit[0])
        beta = int(UpperLimit[1])
        ret = 0

        for D in range(1, 10):
            #compute lambda_a
            if D > int(str(a)[0]):
                lambda_a = math.log10(1 + 1/D)
            elif D == int(str(a)[0]):
                lambda_a = math.log10((1 + D) / a)
            elif D < int(str(a)[0]):
                lambda_a = 0

            #compute lamdba_b
            if D > int(str(b)[0]):
                lambda_b = 0
            elif D == int(str(b)[0]):
                lambda_b = math.log10(b/D)
            elif D < int(str(b)[0]):
                lambda_b = math.log10(1 + 1/D)

            #compute lambda_c
            lambda_c = (beta - alpha) + math.log10(b/a)

            #Compute P_D
            P_D = 1/lambda_c * ((beta - alpha -1) * math.log10(1 + 1/D) + lambda_a + lambda_b)

            if ret == 0:
                P.append(round(P_D * size))
            elif ret == 1:
                P.append(P_D)

        
        return(P, digit_observed, size)


#First Digit Finite range Benford's law
def first_digit_benford_finite_range(input_data):
    #print("[Debug] Calculating first digit frequency")
    #Calcuate perfect Benford distribution.
    #print("[Debug] Computing ideal Benford frequency")
    
    benford_frequency, digit_frequency, dataset_size = finite_range_benford_distribution(-1, input_data)

    #Compute Benford distribution for data of length equal to dataset
    benford_frequency_percent = []
    digit_frequency_percent = []


    for x in benford_frequency:
        benford_frequency_percent.append(float(x / dataset_size))

    #Compute digit frequency percent 
    for x in digit_frequency:
        digit_frequency_percent.append(float(x / dataset_size))

    #Compute Z statistic for this data:

    #Compute von-mises statistics
    von_mises_stat = compute_von_mises(benford_frequency, digit_frequency, benford_frequency_percent, dataset_size)

    #Compute d* statistic
    d_star_stat = compute_dstar(digit_frequency_percent, benford_frequency_percent, dataset_size)

    return(digit_frequency, benford_frequency, digit_frequency_percent, benford_frequency_percent, von_mises_stat, d_star_stat, dataset_size)



def refine_first_digit_finite_range(lowerbound, upperbound, dataset):
    #Compute data in finite range
    dataset_refined = []
    for x in range(0, len(dataset)):
        if int(dataset[x]) >= lowerbound and int(dataset[x]) <= upperbound:
            dataset_refined.append(int(dataset[x]))

    #Calculate the frequency of each digit
    digit_frequency_observed = [0] * 9

    for x in dataset_refined:
        digit_frequency_observed[int(str(x)[0]) - 1] += 1

    
    return(digit_frequency_observed, len(dataset_refined))


#Output the results of first_digit_test to file argv[2]
def output_first_digit_test(digit_occurance, benford_occurance, d_star_test, ad_test, size, savefilename):
    #Round output figures to 3.d.p. Output relevant entries as percentages.
    for x in range(0,len(digit_occurance)):
        digit_occurance[x] = str(int(digit_occurance[x]))
        benford_occurance[x] = str(int(benford_occurance[x]))

    line = ""
    line += "N = {}".format(int(size))

    #write results to the file with table formats. 
    for x in range(0,len(digit_occurance)):
        line += ",{}".format(str(int(digit_occurance[x]) - int(benford_occurance[x])))

    line += ",{},{}".format(str(d_star_test), str(ad_test))

    #Calaculate the sum of Benford Distribution - ensure it equals 100!
    print(line)
    return(0)


def compute_von_mises(expected_list, observed_list, benford_probability, size):
    #Compute cdf for expected and observed outcomes NOT normalised
    observed_cdf = []
    for x in range(0, len(observed_list)):
        r = 0
        for y in range(0, x + 1):
            r += observed_list[y] 
        observed_cdf.append(r)

    expected_cdf = []
    for x in range(0, len(expected_list)):
        r = 0
        for y in range(0, x + 1):
            r += expected_list[y]
        expected_cdf.append(r)

    #Compute Z_j, Z_bar and H_j
    Z = []
    for x in range(0, len(observed_cdf)):
        Z.append(observed_cdf[x] - expected_cdf[x])

    Z_bar = 0
    for x in range(0, len(Z)):
        Z_bar += Z[x] * benford_probability[x]

    H = []
    for x in range(0, len(expected_cdf)):
        H.append(expected_cdf[x] / size)

    #Compute W^2
    summation = 0
    for j in range(0, len(Z)):
        summation += (Z[j] ** 2) * benford_probability[j]

    W_squared = (1/size) * summation

    #Compute U^2
    summation = 0
    for j in range(0, len(Z)):
        summation += ((Z[j] - Z_bar) ** 2) * benford_probability[j]

    U_squared = (1/size) * summation

    #Compute A^2
    summation = 0
    for j in range(0, len(Z) - 1):
        #print(H[j])
        if (H[j] * (1 - H[j])) == 0:
            continue
            
        else:
            summation += ((Z[j] ** 2) * benford_probability[j]) / (H[j] * (1 - H[j]))

    A_squared = (1/size) * summation


    #Catogrise signifcance levels
    #W^2
    if W_squared >= 0.461 and W_squared < 0.743:
        W_squared = str('{:.3f}'.format(W_squared)) + " *"
    elif W_squared >= 0.743:
        W_squared = str('{:.3f}'.format(W_squared)) + " **"
    else:
        W_squared = str('{:.3f}'.format(W_squared))

    #U^2
    if U_squared >= 0.187 and U_squared < 0.268:
        U_squared = str('{:.3f}'.format(U_squared)) + " *"
    elif U_squared >= 0.268:
        U_squared = str('{:.3f}'.format(U_squared)) + " **"
    else:
        U_squared = str('{:.3f}'.format(U_squared))

    #A^2
    if A_squared >= 2.492 and A_squared < 3.88:
        A_squared = str('{:.3f}'.format(A_squared)) + "\enspace(*)"
    elif A_squared >= 3.88:
        A_squared = str('{:.3f}'.format(A_squared)) + "\enspace(**)"
    else:
        A_squared = str('{:.3f}'.format(A_squared))



    #Format return parameter
    return_value = ["W^2={},".format(W_squared), "U^2={},".format(U_squared), "A^2={}".format(A_squared)]

    return(return_value)
    
#compute d* statistic 
def compute_dstar(p, b, size):
    #compute maximum value of d
    d_max = 0
    for x in range(0, len(b)):
        if x != 8:
            d_max += b[x] ** 2
        else:
            d_max += (p[x] - b[x]) ** 2

    d_max  = math.sqrt(d_max)

    #compute d*
    d_star = 0
    for x in range(0, len(b)):
        d_star += (p[x] - b[x]) ** 2

    d_star_morrow = d_star
    d_star = math.sqrt(d_star)
    d_star_norm = d_star / d_max

    #Morrow's d* statistic
    d_star_morrow = size * d_star_morrow
    d_star_morrow = math.sqrt(d_star_morrow)

    #Compute confidence levels for Morrow's d* test
    if d_star_morrow >= 1.330 and d_star_morrow < 1.596:
        d_star_morrow = str(format('{:.3f}'.format(d_star_morrow))) + "\enspace(*)"
    elif d_star_morrow >= 1.596:
        d_star_morrow = str(format('{:.3f}'.format(d_star_morrow))) + "\enspace(**)"
    else:
        d_star_morrow = str(format('{:.3f}'.format(d_star_morrow)))

    d_test = []
    d_test.append(str(d_star_norm))
    d_test.append(r'd*={}'.format(str(d_star_morrow)))
    d_test.append(" (Morrow)")

    return(d_test)
    
def input_numbers(input_filename): 
    # Input data from argv[1] into input_data (newline delimited)
    try:
        open_file = open(input_filename, "r")
        input_data = open_file.readlines()  
        open_file.close()  
    except:
        print("[Fatal Error] Failed to read data from input file. Exiting.")
        #usage()
        exit()  

    #print("[Debug] Input Data recieved from ", input_filename)                       

    # Remove all null entries from the input data
    input_data = ' '.join(input_data).split()   

    #Remove first entry of data if this entry contains a label for a column.
    try:
        int(input_data[0])
    except:
        del input_data[0]

    #Remove all null values and leading zeros. Save this result in input_data_sanitised
    #print("[Debug] Sanitising Input Data")

    try:
        input_data.remove('0')
        input_data.remove('0.')
    except:
        pass

    input_data_sanitised = input_data

    for x in range(0,len(input_data) - 1):
        input_data_sanitised[x] = input_data[x].replace('.','')
        while input_data_sanitised[x][0] == "0":
                input_data_sanitised[x] = input_data_sanitised[x][1:]
                if input_data_sanitised[x] == '':
                    break

    # Remove all null entries from input_data_sanitised
    input_data_sanitised = ' '.join(input_data_sanitised).split()
    #print("[Debug] Input Data Sanitised Successfully")  
    return(input_data_sanitised) 


def main():
    global lowerlimit
    global upperlimit

    try:
        #input lower limit
        lowerlimit = int(sys.argv[2])
        #input upper limit
        upperlimit = int(sys.argv[3])
    except:
        print("Please type an integer as input.")
        exit()

    #Check that the limits make sense
    if lowerlimit >= upperlimit:
        print("Please ensure that the lower limit is less than the upper limit.")
        exit()

    filename = sys.argv[4]
    datafile = sys.argv[1]

    data = input_numbers(datafile)
    data_raw, benford_raw, data_percent, benford_percent, von_mises_statistic, d_star_statistic, data_size = first_digit_benford_finite_range(data)

    output_first_digit_test(data_raw, benford_raw, d_star_statistic[1], von_mises_statistic[2], data_size, filename)
    return(0)

if __name__ == '__main__':
    main()
