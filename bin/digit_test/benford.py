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
    print("Analyse data and verify conformity with the Benford Distribution. The output includes several goodness-of-fit tests for the data and a plot of the data is saved to file. Commandline argument required are to the text file containing raw data to analyse, the mode of analysis (see below) and the location to save plotted data.\n\n    python3 benford.py <filename> <mode (numeric)> <plot_filename>\n\nModes:\n f1   First Digit Finite Range \n 1    First Digit\n 12   First-Second Digit\n 12h  First-Second Digit with heatmap\n 12hn Normalised Residual First-Second Digit with heatmap\n 2    Second Digit\n 23h  Second-Third Digit with heatmap\n 23hn Normalised Residual Second-Third Digit with heatmap\n")
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

    print("[Debug] Input Data recieved from ", input_filename)                       

    # Remove all null entries from the input data
    input_data = ' '.join(input_data).split()   

    #Remove first entry of data if this entry contains a label for a column.
    try:
        float(input_data[0])
    except:
        del input_data[0]

    #Remove all null values and leading zeros. Add trailing zero to any entry of length 1. Save this result in input_data_sanitised
    print("[Debug] Sanitising Input Data")

    input_data_floats = [float(x) for x in input_data]
    lowerbound = min(input_data_floats)
    upperbound = max(input_data_floats)
    print(f'{lowerbound} {upperbound}')

    try:
        input_data.remove('0')
        input_data.remove('0.')
    except:
        pass

    input_data_sanitised = input_data

    for x in range(0, len(input_data)):
        input_data_sanitised[x] = input_data[x].replace('.','').replace('-','')
        while input_data_sanitised[x][0] == "0":
                input_data_sanitised[x] = input_data_sanitised[x][1:]
                if input_data_sanitised[x] == '':
                    break
        if len(input_data_sanitised[x]) == 1:
            input_data_sanitised[x] = input_data_sanitised[x]

    # Remove all null entries from input_data_sanitised
    input_data_sanitised = ' '.join(input_data_sanitised).split()
    print("[Debug] Input Data Sanitised Successfully")  
    return(input_data_sanitised, lowerbound, upperbound) 






### --------------------------------------- DIGIT TEST --------------------------------------------- ###




#Output the results of first_digit_test to file argv[2]
def output_digit_test(digit_occurance, benford_occurance, z_stat, mode):
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
    if mode[0:2] == '12':
        offset = 10
    elif mode[0:2] in ['23'] or mode in ['2']:
        offset = 0
    else:
        offset = 1

    for x in range(0,len(z_stat)):
        line = ""
        line = str(x + offset) + '&' + " " * (8 + len("Digit") - len(str(x + offset)) - 1)
        line += benford_frequency[x] + '&' + " " * (8 + len("Expected Distribution Occurance") - len(benford_frequency[x]) - 1)
        line += '{:.0f}'.format(float(digit_frequency[x])) + '&' + " " * (8 + len("Observed Distribution Occurance") - len('{:.0f}'.format(float(digit_frequency[x]))) - 1)
        line += z_stat[x] + "\\\\"
        print(line)

    #Calaculate the sum of Benford Distribution 
    print("")
    expected_sum = 0

    for x in range(0, len(digit_frequency)):
        expected_sum += float(benford_frequency[x])

    line = " " * (8 + len("Digit")) + '{:.1f}'.format(expected_sum) + "\n"
    print(line)
    return(0)



#Perform the digit test on the data.
def digit_test(input_data, mode):
    #Calculate the frequency of each character set in indexes specified.
    if mode in ['1']:
        print("[Debug] Calculating first digit frequency")
        digit_frequency = [0] * 9
        offset = 1
    elif mode in ['2']:
        print("[Debug] Calculating second digit frequency")
        digit_frequency = [0] * 10
        offset = 0
    elif mode in ['12', '12h', '12hn']:
        print("[Debug] Calculating second digit frequency")
        digit_frequency = [0] * 90
        offset = 10
    elif mode in ['23', '23h', '23hn']:
        print("[Debug] Calculating second digit frequency")
        digit_frequency = [0] * 100
        offset = 0
    
    digits = 0
    for x in input_data:
        try:
            digits = int(x[int(mode[0]) - 1:int(mode[1 % len(mode)])])
            digit_frequency[digits - offset] += 1
        except:
            print(f'[Debug] Ignoring {x}')
            pass


    #Convert frequncies to percentage expressed as a decimal. 
    print("[Debug] Converting to percentages")

    digit_frequency_percent = []
    for x in range(0, len(digit_frequency)):
        digit_frequency_percent.append(float(digit_frequency[x] / len(input_data)))

    #Calcuate perfect Benford distribution.
    print("[Debug] Computing ideal Benford frequency")
    benford_frequency_percent = benford_distribution(mode, 0)
    
    #Compute Benford distribution for data of length equal to dataset
    benford_frequency = []
    for x in benford_frequency_percent:
        benford_frequency.append(x * len(input_data))

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
def finite_range_benford_distribution(data, mode, lb, ub):
    #Finite range calculation
    P = []
    # Calculate upperlimit
    data_float = [ float(x) for x in data ]
    lowerlimit = lb
    upperlimit = ub
    print(lowerlimit, upperlimit)

    # refine range 
    digit_observed, size = refine_finite_range(int(lowerlimit), int(upperlimit), data, mode)
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

    if mode == 'f1':
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

            
            P.append(P_D * size)
            

        
        return(P, digit_observed, size)

    elif mode == 'f2':
        a_1 = int(str(a)[0])
        a_2 = int(str(a).replace('.','')[1])
        b_1 = int(str(b)[0])
        b_2 = int(str(b).replace('.','')[1])

        for d_2 in range(0, 10):
            # compute lambda_a
            lambda_a = 0

            if d_2 > a_2:
                for d_1 in range(a_1, 10):
                    lambda_a += np.log10(1 + 1/(10*d_1 + d_2))

            elif d_2 < a_2:
                for d_1 in range(a_1 + 1, 10):
                    lambda_a += np.log10(1 + 1/(10*d_1 + d_2))

            elif d_2 == a_2:
                lambda_a += np.log10((a_1 + (d_2 + 1) * 10 ** (-1) ) / a)
                for d_1 in range(a_1 + 1, 10):
                    lambda_a += np.log10(1 + 1/(10*d_1 + d_2))

            # compute lambda_b
            lambda_b = 0

            if d_2 > b_2:
                for d_1 in range(1, b_1):
                    lambda_b += np.log10(1 + 1/(10*d_1 + d_2))

            elif d_2 < b_2:
                for d_1 in range(1, b_1 + 1):
                    lambda_b += np.log10(1 + 1/(10*d_1 + d_2))

            elif d_2 == b_2:
                lambda_b += np.log10(b / (b_1 + 10**(-1) * d_2))
                for d_1 in range(1, b_1):
                    lambda_b += np.log10(1 + 1/(10*d_1 + d_2))



            #compute lambda_c
            lambda_c = (beta - alpha) + math.log10(b/a)

            #compute lambda_d (sum in main expression)
            lambda_d = 0

            
            for d_1 in range(1, 10):
                 lambda_d += np.log10(1 + 1/(10*d_1 + d_2))

            #Compute P_D
            P_D = 1/lambda_c * ((beta - alpha -1) * lambda_d + lambda_a + lambda_b)
            
            # Append to distribution
            P.append(round(P_D * size, 4))

        print(P)
        return(P, digit_observed, size)
            





### --------------------------------------- FINITE RANGE --------------------------------------------- ##





#First Digit Finite range Benford's law
def benford_finite_range(input_data, mode, lb, ub):
    print("[Debug] Calculating first digit frequency")
    #Calcuate perfect Benford distribution.
    print("[Debug] Computing ideal Benford frequency")
    
    benford_frequency, digit_frequency, dataset_size = finite_range_benford_distribution(input_data, mode, lb, ub)

    #Compute Benford distribution for data of length equal to dataset
    benford_frequency_percent = []
    digit_frequency_percent = []


    for x in benford_frequency:
        benford_frequency_percent.append(float(x / dataset_size))

    #Compute digit frequency percent 
    for x in digit_frequency:
        digit_frequency_percent.append(float(x / dataset_size))

    #Compute Z statistic for this data:
    print("[Debug] Computing Z statistic")
    z_stat = []
    for x in range(0, len(digit_frequency)):
        if int(digit_frequency[x]) != 0:
            z_stat.append(compute_z_statistic(digit_frequency_percent[x], benford_frequency_percent[x], dataset_size))
        else:
            z_stat.append(0)

    #Compute von-mises statistics
    von_mises_stat = compute_von_mises(benford_frequency, digit_frequency, benford_frequency_percent, dataset_size)

    #Compute d* statistic
    d_star_stat = compute_dstar(digit_frequency_percent, benford_frequency_percent, dataset_size)

    return(digit_frequency, benford_frequency, digit_frequency_percent, benford_frequency_percent, z_stat, von_mises_stat, d_star_stat, dataset_size)



def refine_finite_range(lowerbound, upperbound, dataset, mode):
    #Compute data in finite range
    dataset_refined = []
    for x in range(0, len(dataset)):
        if int(dataset[x]) >= lowerbound and int(dataset[x]) <= upperbound:
            dataset_refined.append(int(dataset[x]))

    #Calculate the frequency of each digit
    if mode == 'f1':
        digit_frequency_observed = [0] * 9

        for x in dataset_refined:
            try:
                digit_frequency_observed[int(str(x)[0]) - 1] += 1
            except:
                pass

    elif mode == 'f2':
        digit_frequency_observed = [0] * 10

        for x in dataset_refined:
            try:
                digit_frequency_observed[int(str(x)[1])] += 1
            except:
                continue
    
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

        #edge case for finite range second digit law
        if yerror[x] == 0.01 and observed[x] == 1 and expected[x] == 0:
            difference.append(0)
        else:
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
    try:
        denominator = math.sqrt((p_zero * (1 - p_zero)) / N)
    except:
        denominator = 0
    try:
        return(float(numerator / denominator))
    except:
        return(0)

#Calculate X^2 statistic
def compute_chi_squared_statistic(expected_list, actual_list):
    #Multiply by the total to achieve and approximation of the actual and expected observations
    chi = 0
    for x in range(0, len(expected_list)):
        entry = (float(expected_list[x]) - float(actual_list[x])) ** 2
        entry = entry / (expected_list[x])
        chi += entry

    return_value = [chi, chi/8]

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

    #A^2 - define p_values from https://www.jstor.org/stable/pdf/3315828.pdf?refreqid=excelsior%3A5c388242fbac9f8040c6fae714dc9e86
    if len(benford_probability) == 9:
        p_values = [(2.392 + 2.367)/2, (3.78 + 3.72)/2]
    elif len(benford_probability) == 10:
        p_values = [2.392, 3.78]
    else:
        p_values = [2.492, 3.88]

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
    





### --------------------------------------- PLOT DATA --------------------------------------------- ###







def plot_bar_chart(bins, frequency, benford_freq, dataset_size, von_mises, dstar, mode):
    #increase font size
    plt.rcParams.update({'font.size': 13.5})
    # plt.rcParams['text.usetex'] = True

    difference, y_colours, yerror = compute_normalised_residuals(frequency, benford_freq)

    width = 0.7

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    ax0 = plt.subplot(gs[0])

    if mode in ['1', 'f1']:
        ax0.errorbar(bins, benford_freq, yerr=yerror, label="Expected Occurrence", color='black', marker='x', fmt='x', capsize=3, elinewidth=1, zorder=1)
    elif mode == '12':
        ax0.errorbar(bins, benford_freq, yerr=yerror, label="Expected Occurrence", color='black', marker='.', fmt='x', capsize=2, elinewidth=1, zorder=1)
    elif mode in ['2','f2']:
        ax0.errorbar(bins, benford_freq, yerr=yerror, label="Expected Occurrence", color='black', marker='x', fmt='x', capsize=3, elinewidth=1, zorder=1)
    

    ax0.bar(bins, frequency, width, color='grey', label="Observed Occurrence", zorder=-1)

    #Format tick labels in scientific notation
    
    yticks = ax0.get_yticks()
    # print(yticks)

    #Find first non-zero integer
    lower_index = 0
    for x in range(0, len(yticks)):
        if yticks[x] > 0:
            lower_index = x
            lower_index_value = yticks[x]
            break

    # Only format in scientfic notation if the lowest tick is greater than 1000
    # print(lower_index)
    if lower_index_value >=1000:
        # Compute difference in magnitude 
        mag_diff = math.floor(np.log10(yticks[-1])) - math.floor(np.log10(math.floor(yticks[lower_index])))
        
        # Compute magnitude to report on ylabel
        mean_mag_diff = math.floor(np.log10(yticks[lower_index])) + mag_diff
        report_mag_diff = f'10^{mean_mag_diff}'
        # print(report_mag_diff)
        
        # Divide x ticks by this magnitude to obtain fractional part
        yticks_fractional = yticks / (10**mean_mag_diff)
        # print(yticks_fractional)
        
        # Set fractional ticks to tick labels
        plt.yticks(yticks[lower_index - 1:-1], yticks_fractional[lower_index - 1:-1])

    else: 
        report_mag_diff = ""

    # Format N in scientific notation of N>10000
    if dataset_size >= 10000:
        N_mag = math.floor(np.log10(dataset_size))
        
        # Compute magnitude to report on ylabel
        # N_mean_mag_diff = math.floor(np.log10(yticks[lower_index])) + N_mag_diff
        # print(f'{round(dataset_size / 10**N_mag, 1) } * 10** {N_mag}')
        dataset_size_report = r'{} \times 10^{}'.format(round(dataset_size / 10**N_mag, 1),N_mag)

    else:
        dataset_size_report = str(dataset_size)

    # Format axis labels
    plt.xlabel("Digit Value")
    if report_mag_diff != "":
        plt.ylabel(r"Digit Occurrence (${}$)".format(report_mag_diff), fontsize=15)
    else:
        plt.ylabel(r"Digit Occurrence", fontsize=15)
    plt.xticks(bins, "")

    if mode == '12':
        plt.xlim(9,100)

    
    if mode not in ['2', 'f2']:#Format Legend
        patch = []
        handles, labels = ax0.get_legend_handles_labels()
        patch.append(mpatches.Patch(color='green', label=r'$|\sigma|$ < 1'))
        patch.append(mpatches.Patch(color='firebrick', label=r'$|\sigma|$ > 1'))
        patch.append(mpatches.Patch(color='none', label=r'${}$'.format(von_mises)))
        patch.append(mpatches.Patch(color='none', label=r'${}$'.format(dstar)))
        patch.append(mpatches.Patch(color='none', label=r'$N = {}$'.format(dataset_size_report)))
    
    else:
        patch = []
        handles, labels = ax0.get_legend_handles_labels()
        patch.append(mpatches.Patch(color='green', label=r'$|\sigma|$ < 1,    ${}$'.format(dstar)))
        patch.append(mpatches.Patch(color='firebrick', label=r'$|\sigma|$ > 1,    $N={}$'.format(dataset_size_report)))
        #patch.append(mpatches.Patch(color='white', label=r'${}$'.format(von_mises)))
        # patch.append(mpatches.Patch(color='none', label=r'${}$, $N={}$'.format(dstar, str(dataset_size))))
        #patch.append(mpatches.Patch(color='white', label=r'$N = {}$'.format(str(dataset_size))))

    for x in patch:
        handles.append(x)

    plt.legend(handles=handles, loc='best')
    
    plt.subplots_adjust(hspace=0)

    #Second (smaller) subplot
    #Calculate y ticks
    y_range = 0
    for x in difference: 
        if abs(x) >= y_range:
            y_range = math.ceil(abs(x))

    
    #Begin plotting
    ax1 = plt.subplot(gs[1])
    ax1.bar(bins, difference, 0.70, color=y_colours)

    if mode in ['1', 'f1']:
        bins_ticks = []
        for x in bins:
            bins_ticks.append(x + 1)

    else:
        bins_ticks = bins

    plt.xticks(bins, bins_ticks)
    

    #format spacing on graph

    if y_range > 2 and y_range <= 3:
        plt.yticks((-y_range + 1, 0, y_range - 1))
    elif y_range > 3 and y_range <= 6:
        plt.yticks((-y_range + 1, 0, y_range - 1))
        #y_range = y_range + 1
    elif y_range > 6 and y_range <= 10:
        plt.yticks((-y_range + 3, 0, y_range - 3))
        #y_range = y_range + 2
    elif y_range > 10:
        plt.yticks((-int(y_range/2), 0, int(y_range/2)))

    ax1.set_ylim([-y_range,y_range])

    if mode in ['1', 'f1']:
        plt.xlabel("First Digit Value", fontsize=15)
        plt.ylabel("Normalised Residual", fontsize=15)
        plt.ylim(-y_range - 0.75, y_range + 0.75) 

    elif mode == '12':
        plt.xlabel("First Two Digit Values", fontsize=15)
        plt.ylabel("Normalised Residual", fontsize=15)
        plt.ylim(-y_range - 0.75, y_range + 0.75) 
        plt.xticks(np.arange(10, 100, 5))
        plt.xlim(9,100)

    elif mode in ['2','f2']:
        plt.xlabel("Second Digit Value", fontsize=15)
        plt.ylabel("Normalised Residual", fontsize=15)
        plt.ylim(-y_range - 0.75, y_range + 0.75) 


    #format graph in general
    plt.axhline(linewidth=0.5, color='black')
    plt.axhline(y=1, linewidth=0.75, color='black', linestyle='--')
    plt.axhline(y=-1, linewidth=0.75, color='black', linestyle='--')
    #plt.legend(handles=legend_elements, loc='best')
    fig.align_ylabels()
    print('[Debug] Saving Plot as {}'.format(sys.argv[3]))
    plt.savefig('{}'.format(sys.argv[3]), bbox_inches='tight')
    return(0)


def plot_heat_map(frequency, benford_freq, m):
    # increase font size
    plt.rcParams.update({'font.size': 13.5})

    #Setup x and y axis arrays

    if m[0:2] == '12':
        y_axis = np.arange(1, 11, 1)
        x_axis = np.arange(0, 10, 1)
    elif m[0:2] == '23':
        y_axis = np.arange(0, 11, 1)
        x_axis = np.arange(0, 11, 1)
        print(len(benford_freq), len(frequency))

    values_to_plot = []
    row = []
    
    # 12 heatmap test non-normalised
    if m in ['12h', '23h']:
        # print(benford_freq)
        #Compute absolute difference as numpy array to plot

        if m[0:2] == '12':
            lower_x = 1
            indent = 10
        elif m[0:2] == '23':
            lower_x = 0
            indent = 0


        for x in range(lower_x,10):
            for y in range(0,10):
                # print(str(x)+str(y))
                # print(frequency[int(str(x) + str(y)) - 10] - benford_freq[int(str(x) + str(y)) - 10])
                row.append(round(frequency[int(str(x) + str(y)) - indent] - benford_freq[int(str(x) + str(y)) - indent], 1))

            array = np.asarray(row)

            if x == lower_x:
                values_to_plot = array
            else:
                values_to_plot = np.vstack((values_to_plot, array))
            row = []
        limit = roundup(max(abs(np.amin(values_to_plot)), np.amax(values_to_plot)))
        



    # 12 and 23 normalised heatmap
    elif m == '23hn' or m == '12hn':
        for i in range(0, len(frequency)):
            frequency[i] = int(round(frequency[i], 0))

        print(frequency)
        row = []
        for x in range(0, len(frequency) - 9, 10):
            row.append(frequency[x:x + 10])
            array = np.asarray(row)
            if x == 0:
                values_to_plot = array
            else:
                values_to_plot = np.vstack((values_to_plot, array))
            row = []
        
        limit = math.ceil(max(abs(np.amin(values_to_plot)), np.amax(values_to_plot)))
    
    print(values_to_plot)
    
    test = [-limit, limit, limit, limit, limit, limit, limit, limit, limit, limit]
    array = np.asarray(test)
   
    values_to_plot = np.vstack((values_to_plot, array))


    fig, ax = plt.subplots()

    im, cbar = heatmap(limit, values_to_plot, y_axis , x_axis, m, ax=ax,
                   cmap="coolwarm", cbarlabel="Deviation")
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

    if mode in ['12h', '23h']:
        cbarlabel = 'Deviation'
        ticks = []
        lim = roundup(limit)
        base = 1 - math.floor(np.log10(lim))

        for x in range(0, limit + 1, int(round(lim/3, base))):
            ticks.append(x)

        for i in range(1, len(ticks)):
            ticks.append(-ticks[i])
   
    elif mode == '23hn' or mode == '12hn':
        cbarlabel = 'Normalised Deviation'
        ticks = []
        lim = roundup(limit)
        base = 1 - math.floor(np.log10(lim))

        for x in range(0, lim + 1, int(round(lim/3, base))):
            ticks.append(x)

        
        for i in range(0, len(ticks)):
            ticks.append(-ticks[i])
        # ticks.append(0)

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

    


### --------------------------------------- Main() --------------------------------------------- ###

#main function of the program. 
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
    data, lowerbound, upperbound = input_numbers(filename)
    print("[Debug] Starting First Digit Analysis")

    if mode in ['1', '12', '12h', '12hn', '23', '23h', '23hn', '2']:
        data_raw, benford_raw, data_percent, benford_percent, z_statistic, von_mises_statistic, d_star_statistic = digit_test(data, mode)
        
        bins_to_plot = []

        if mode == '1':
            bins_to_plot = np.arange(9)
        elif mode == '2':
            bins_to_plot = np.arange(0, 10, 1)
        elif mode in ['12', '12h', '12hn']:
            bins_to_plot = np.arange(10, 100, 1)
        elif mode in ['23', '23hn', '23h']:
            for x in range(0, 100):
                bins_to_plot.append(x)
    
    elif mode == 'f1':
        data_raw, benford_raw, data_percent, benford_percent, z_statistic, von_mises_statistic, d_star_statistic, data_size = benford_finite_range(data, mode, lowerbound, upperbound)
        bins_to_plot = np.arange(9)
    
    elif mode == 'f2':
        data_raw, benford_raw, data_percent, benford_percent, z_statistic, von_mises_statistic, d_star_statistic, data_size = benford_finite_range(data, mode, lowerbound, upperbound)
        bins_to_plot = np.arange(10)

    
    # print(data_raw)
    data_size = 0
    for x in data_raw:
        data_size += x

    #Output results
    print("[Debug] Analysis complete. Outputing results.")
    print("\n\n###--- Analysis for", filename, "---###\n")

    # Process the mode of analysis e.g. first digit analysis = 1
    output_digit_test(data_raw, benford_raw, z_statistic, mode)
    
    # Test statistic Output
    print("Cramer-von Mises test: {} {} {}".format(von_mises_statistic[0],von_mises_statistic[1],von_mises_statistic[2]))
    print("d* test: {}, {}{}".format(d_star_statistic[0], d_star_statistic[1], d_star_statistic[2]))
    #Legend significance levels
    print("\n * significant at the .05 level\n** significant at the .01 level\n")

    # Create plots of the data
    print("[Debug] Generating Plot of the data.")
    if mode in ['1','f1','f2','2','12']:
        plot_bar_chart(bins_to_plot, data_raw, benford_raw, data_size, von_mises_statistic[2], d_star_statistic[1], mode)
    
    elif mode in ['12h', '12hn', '23h', '23hn']:
        if 'hn' in mode:
            norm_residuals, null, null = compute_normalised_residuals(data_raw, benford_raw)
            plot_heat_map(norm_residuals, benford_percent, mode)
        else:
            plot_heat_map(data_raw, benford_raw, mode)
    
    print("[Debug] Output complete. Exiting.")
    exit()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage()
        exit()
    else:
        main(sys.argv[2])
    
