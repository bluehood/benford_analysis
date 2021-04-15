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




### --------------------------------------- RANDOM --------------------------------------------- ###




def roundup(x):
    return(int(math.ceil(x / 10.0)) * 10)


# print the usage for the program.
def usage():
    print(f'Analyse data and verify conformity with the Benford Distribution. Output is printed to the screen and contains the Chi sqauared, A squared and d* test statistics for a given Benford digit test. Commandline argument required are to the text file containing raw data to analyse and the mode of analysis (see below).\n\n    {sys.argv[0]} <filename> <mode (numeric)> \n\nModes:\n f1   First Digit Finite Range \n 1    First Digit\n 12   First-Second Digit\n 2    Second Digit\n 23   Second-Third Digit test\n')
    return(0)






### --------------------------------------- IMPORT DATA --------------------------------------------- ###







def input_numbers(input_filename): 
    # Input data from argv[1] into input_data (newline delimited)
    try:
        open_file = open(input_filename, "r")
        input_data = open_file.readlines()  
        open_file.close()  
    except:
        print("[Fatal Error] Failed to read data from input file. Exiting.")
        usage()
        exit()  

    # print("[Debug] Input Data recieved from ", input_filename)                       

    # Remove all null entries from the input data
    input_data = ' '.join(input_data).split()   

    #Remove first entry of data if this entry contains a label for a column.
    try:
        int(input_data[0])
    except:
        del input_data[0]

    #Remove all null values and leading zeros. Add trailing zero to any entry of length 1. Save this result in input_data_sanitised
    # print("[Debug] Sanitising Input Data")

    try:
        input_data.remove('0')
        input_data.remove('0.')
    except:
        pass

    input_data_sanitised = input_data

    for x in range(0, len(input_data)):
        input_data_sanitised[x] = input_data[x].replace('.','')
        while input_data_sanitised[x][0] == "0":
                input_data_sanitised[x] = input_data_sanitised[x][1:]
                if input_data_sanitised[x] == '':
                    break
        

    # Remove all null entries from input_data_sanitised
    input_data_sanitised = ' '.join(input_data_sanitised).split()
    # print("[Debug] Input Data Sanitised Successfully")  
    return(input_data_sanitised) 






### --------------------------------------- DIGIT TEST --------------------------------------------- ###



#Perform the digit test on the data.
def digit_test(input_data, mode):
    #Calculate the frequency of each character set in indexes specified.
    if mode in ['1']:
        # print("[Debug] Calculating first digit frequency")
        digit_frequency = [0] * 9
        offset = 1
    elif mode in ['2']:
        # print("[Debug] Calculating second digit frequency")
        digit_frequency = [0] * 10
        offset = 0
    elif mode in ['12', '12h', '12hn']:
        # print("[Debug] Calculating second digit frequency")
        digit_frequency = [0] * 90
        offset = 10
    elif mode in ['23', '23h', '23hn']:
        # print("[Debug] Calculating second digit frequency")
        digit_frequency = [0] * 100
        offset = 0
    
    digits = 0
    for x in input_data:
        try:
            digits = int(x[int(mode[0]) - 1:int(mode[1 % len(mode)])])
            digit_frequency[digits - offset] += 1
        except:
            pass

    #Convert frequncies to percentage expressed as a decimal. 
    # print("[Debug] Converting to percentages")

    digit_frequency_percent = []
    for x in range(0, len(digit_frequency)):
        digit_frequency_percent.append(float(digit_frequency[x] / len(input_data)))

    #Calcuate perfect Benford distribution.
    # print("[Debug] Computing ideal Benford frequency")
    benford_frequency_percent = benford_distribution(mode, 0)
    
    #Compute Benford distribution for data of length equal to dataset
    benford_frequency = []
    for x in benford_frequency_percent:
        benford_frequency.append(round(x * len(input_data)))

    
    #Compute von-mises statistics
    von_mises_stat = compute_von_mises(benford_frequency, digit_frequency, benford_frequency_percent, len(input_data), mode)

    #Compute d* statistic
    d_star_stat = compute_dstar(digit_frequency_percent, benford_frequency_percent, len(input_data), mode)

    # Compute X_2 statistic 
    x_squared = compute_chi_squared_statistic(benford_frequency, digit_frequency, len(digit_frequency) - 1)

    return(x_squared, von_mises_stat, d_star_stat, len(input_data))





### --------------------------------------- BENDFORD DISTRIBUTIONS--------------------------------------------- ###






#Calculate the Ideal Benford distribution
def benford_distribution(mode, size):

    if mode == '1':
        benford_frequency_first_digit = []
        for x in range(1,10):
            benford_frequency_first_digit.append(math.log10(1 + 1/x))

        return(benford_frequency_first_digit)

    elif mode in ['12', '12h', '12hn']:
        benford_frequency_first_second_digit = []
        for x in range(1,10):
            for y in range(0,10):
                benford_frequency_first_second_digit.append(math.log10(x + (y+1)/10) - math.log10(x + y/10))

        return(benford_frequency_first_second_digit)

    elif mode == '2':
        benford_frequency_first_second_digit = []
        benford_frequency_second_digit = [0] * 10
        for x in range(1,10):
            for y in range(0,10):
                benford_frequency_first_second_digit.append(math.log10(x + (y+1)/10) - math.log10(x + y/10))

        for x in range(0, len(benford_frequency_first_second_digit)):
            # print(int(str(x + 10)[1]))
            benford_frequency_second_digit[int(str(x + 10)[1])] += benford_frequency_first_second_digit[x]

        return(benford_frequency_second_digit)
    
    elif mode in ['23', '23h', '23hn']:
        benford_frequency_first_second_third_digit = []
        benford_frequency_second_third_digit = [0] * 100
        for x in range(100,1000):
            benford_frequency_first_second_third_digit.append(math.log10(1 + 1/x))

        for x in range(0, len(benford_frequency_first_second_third_digit)):
            # print(int(str(x + 100)[1:3]))
            benford_frequency_second_third_digit[int(str(x + 100)[1:3])] += benford_frequency_first_second_third_digit[x]
        
        print(benford_frequency_second_third_digit)

        return(benford_frequency_second_third_digit)


#Calculate finite range Benford distributions
def finite_range_benford_distribution(mode, data):
    #Finite range calculation
    if mode == -1:
        P = []
        # try:
        #     #input lower limit
        #     lowerlimit = int(input("Lower Limit: "))
        #     #input upper limit
        #     upperlimit = int(input("Upper Limit: "))
        # except:
        #     print("Please type an integer as input.")
        #     exit()

        # #Check that the limits make sense
        # if lowerlimit >= upperlimit:
        #     print("Please ensure that the lower limit is less than the upper limit.")
        #     exit()

        # Calculate upperlimit
        data_float = [ float(x) for x in data ]
        lowerlimit = min(data_float)
        upperlimit = max(data_float)
        print(lowerlimit, upperlimit)

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





### --------------------------------------- FINITE RANGE --------------------------------------------- ##





#First Digit Finite range Benford's law
def first_digit_benford_finite_range(input_data, mode):
    # print("[Debug] Calculating first digit frequency")
    #Calcuate perfect Benford distribution.
    # print("[Debug] Computing ideal Benford frequency")
    
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
    # print("[Debug] Computing Z statistic")
    z_stat = []
    for x in range(0, len(digit_frequency)):
        if int(digit_frequency[x]) != 0:
            z_stat.append(compute_z_statistic(digit_frequency_percent[x], benford_frequency_percent[x], dataset_size))
        else:
            z_stat.append(0)

    #Compute von-mises statistics
    von_mises_stat = compute_von_mises(benford_frequency, digit_frequency, benford_frequency_percent, dataset_size, mode)

    #Compute d* statistic
    d_star_stat = compute_dstar(digit_frequency_percent, benford_frequency_percent, dataset_size, mode)

    return(digit_frequency, benford_frequency, digit_frequency_percent, benford_frequency_percent, z_stat, von_mises_stat, d_star_stat, dataset_size)



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






### --------------------------------------- GOF TESTS --------------------------------------------- ###






# Normlasied residuals 
def compute_normalised_residuals(observed, expected):
    # Compute errors for expected distribution
    yerror = []
    for x in range(0, len(observed)):
        yerror.append(math.sqrt(expected[x]))

    # Calculate Normalised residuals
    difference = []
    y_colours = []
    for x in range(0, len(yerror)):
        if yerror[x] == 0:
            yerror[x] = 0.01
        difference.append((observed[x] - expected[x]) / yerror[x])
        if abs(difference[x]) > 1:
            y_colours.append('firebrick')
        else:
            y_colours.append('green')
    return(difference, y_colours, yerror)

#Calculate Z statistic
def compute_z_statistic(p, p_zero, N):
    numerator = 0
    denominator = 0
    
    numerator = abs(p - p_zero) - (1 / (2*N))
    denominator = math.sqrt((p_zero * (1 - p_zero)) / N)
    try:
        return(float(numerator / denominator))
    except:
        return(0)

#Calculate X^2 statistic
def compute_chi_squared_statistic(expected_list, actual_list, norm):
    #Multiply by the total to achieve and approximation of the actual and expected observations
    chi = 0
    for x in range(0, len(expected_list)):
        entry = (float(expected_list[x]) - float(actual_list[x])) ** 2
        entry = entry / (expected_list[x])
        chi += entry

    return_value = np.around(chi/norm, 3)

    return(return_value)

#Calculate KS statistic
def compute_ks_statistic(expected_list, actual_list, size):
    '''https://towardsdatascience.com/when-to-use-the-kolmogorov-smirnov-test-dd0b2c8a8f61
    http://people.cs.pitt.edu/~lipschultz/cs1538/prob-table_KS.pdf
    Critical Values: https://www.jstor.org/stable/pdf/2284444.pdf?refreqid=excelsior%3A75be83116d56691079b7426f31618ba3
    '''
    #Note expected_list are percentages at this point
    
    #Compute the cdf for the observed frequency 
    actual_cdf = []
    for x in range(0, len(actual_list)):
        r = 0
        for y in range(0, x + 1):
            r += actual_list[y] / size
        actual_cdf.append(r)

    #Compute expected cdf
    expected_cdf = []
    for x in range(0, len(expected_list)):
        r = 0
        for y in range(0, x + 1):
            r += expected_list[y]
        expected_cdf.append(r)

    #Compute differences in cdf's
    cdf_diff_abs = []

    for x in range(0, len(actual_cdf)):
        difference = abs(actual_cdf[x] - expected_cdf[x])
        cdf_diff_abs.append(difference)

    cdf_diff_abs.sort()

    return(cdf_diff_abs[-1])

#Calculate CM statistics
def compute_von_mises(expected_list, observed_list, benford_probability, size, mode):
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
    if mode == '1':
        p_values = [2.84, 4.56]
    elif mode == '2':       
        p_values = [2.61, 3.97]
    else:
        p_values = [2.61, 3.97]

    # print(f'HERE: {p_values}')

    if A_squared >= p_values[0] and A_squared < p_values[1]:
        A_squared = str('{:.3f}'.format(A_squared)) + "\enspace(*)"
    elif A_squared >= p_values[1]:
        A_squared = str('{:.3f}'.format(A_squared)) + "\enspace(**)"
    else:
        A_squared = str('{:.3f}'.format(A_squared))



    #Format return parameter
    return_value = ["W^2={},".format(W_squared), "U^2={},".format(U_squared), "A^2={}".format(A_squared)]

    return(return_value)
    
#compute d* statistic 
def compute_dstar(p, b, size, mode):
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

    if mode == '1':
        p_values = [1.35, 1.66]
    elif mode == '2':      
        p_values = [1.32, 1.49]
    else:
        p_values = [1.32, 1.49]

    #Compute confidence levels for Morrow's d* test
    if d_star_morrow >= p_values[0] and d_star_morrow < p_values[1]:
        d_star_morrow = str(format('{:.3f}'.format(d_star_morrow))) + "\enspace(*)"
    elif d_star_morrow >= p_values[1]:
        d_star_morrow = str(format('{:.3f}'.format(d_star_morrow))) + "\enspace(**)"
    else:
        d_star_morrow = str(format('{:.3f}'.format(d_star_morrow)))

    d_test = []
    d_test.append(str(d_star_norm))
    d_test.append(r'd*={}'.format(str(d_star_morrow)))
    d_test.append(" (Morrow)")

    return(d_test)
    





### --------------------------------------- Main() --------------------------------------------- ###


def main(mode):
    #Process mode of analysis
    try:
        str(mode)
    except:
        print("[Fatal Error] Cannot process mode", str(mode), ". Please enter a valid mode of analysis.")
        usage()
        exit()

    #Import data from argv[1]
    filename = sys.argv[1]
    data = input_numbers(filename)
    # print("[Debug] Starting First Digit Analysis")

    if mode in ['1', '12', '12h', '12hn', '23', '23hn', '2']:
        X_sqaured, von_mises_statistic, d_star_statistic, data_size = digit_test(data, mode)
    
    elif mode == 'f1':
        X_sqaured, von_mises_statistic, d_star_statistic, data_size = first_digit_benford_finite_range(data, mode)
    

    # # Test statistic Output
    # print("Cramer-von Mises test: {} {} {}".format(von_mises_statistic[0],von_mises_statistic[1],von_mises_statistic[2]))
    # print("d* test: {}, {}{}".format(d_star_statistic[0], d_star_statistic[1], d_star_statistic[2]))
    # #Legend significance levels
    # print("\n * significant at the .05 level\n** significant at the .01 level\n")
    
    # print("[Debug] Output complete. Exiting.")

    # print(f"{filename.split('/')[-2]},{filename.split('/')[-1][5:9].replace('-', '.')},{X_sqaured},{von_mises_statistic[2]},{d_star_statistic[1]}")
    print(f"{X_sqaured},{von_mises_statistic[2]},{d_star_statistic[1]}")

    exit()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        usage()
        exit()
    else:
        main(sys.argv[2])
    
