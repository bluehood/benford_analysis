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




### ---------------------------------- Random --------------------------------------------- ###




def roundup(x):
    return(int(math.ceil(x / 10.0)) * 10)


# print the usage for the program.
def usage():
    print("Analyse data and verify conformity with the Benford Distribution. The output includes several goodness-of-fit tests for the data. Commandline argument required are to the text file containing raw data to analyse, the mode of analysis (see below) and the location to save plotted data.\n\n    python3 benford.py <filename> <mode (numeric)> <plot_filename>\n\nModes:\n -1 - First Digit Finite Range \n 0 - First Digit\n 1 - First-Second Digit\n 2 - First-Second Digit with heatmap\n 3 - Second Digit")
    return(0)






### ---------------------------------- IMPORT DATA --------------------------------------------- ###







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
        input_data_sanitised[x] = input_data[x].replace('.','')
        while input_data_sanitised[x][0] == "0":
                input_data_sanitised[x] = input_data_sanitised[x][1:]
                if input_data_sanitised[x] == '':
                    break
        if len(input_data_sanitised[x]) == 1:
            input_data_sanitised[x] = input_data_sanitised[x] + "0"

    # Remove all null entries from input_data_sanitised
    input_data_sanitised = ' '.join(input_data_sanitised).split()
    print("[Debug] Input Data Sanitised Successfully")  
    return(input_data_sanitised) 








### --------------------------------------- FIRST DIGIT TEST --------------------------------------------- ###







#Output the results of first_digit_test to file argv[2]
def output_first_digit_test(digit_occurance, benford_occurance, z_stat):
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

    #Calaculate the sum of Benford Distribution 
    print("")
    expected_sum = 0

    for x in range(0, len(digit_frequency)):
        expected_sum += float(benford_frequency[x])

    line = " " * (8 + len("Digit")) + '{:.1f}'.format(expected_sum) + "\n"
    print(line)
    return(0)



#Perform the first digit test on the data.
def first_digit_test(input_data):
    #Calculate the frequency of each character {1,2,...,9} in the first digit. 
    print("[Debug] Calculating first digit frequency")
    digit_frequency = [0] * 9
    first_digit = 0
    for x in input_data:
        first_digit = int(x[0])
        digit_frequency[first_digit - 1] += 1

    #Convert frequncies to percentage expressed as a decimal. 
    print("[Debug] Converting to percentages")
    digit_frequency_percent = [0] * 9

    for x in range(0, len(digit_frequency)):
        digit_frequency_percent[x] = float(digit_frequency[x] / len(input_data))

    #Calcuate perfect Benford distribution.
    print("[Debug] Computing ideal Benford frequency")
    benford_frequency_percent = benford_distribution(0, 0)
    
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








### --------------------------------------- FIRST-SECOND DIGIT TEST --------------------------------------------- ###







#Output the results of first_digit_test to file argv[2]
def output_first_second_digit_test(digit_occurance, benford_occurance, z_stat):
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
        line = str(x + 10) + '&' + " " * (8 + len("Digit") - len(str(x + 10)) - 1)
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

def first_second_digit_test(input_data):
    # Calculate the frequency of {10,11,12,...,98,99} in the first two digits. 
    print("[Debug] Calculating first-second digit frequency")
    digit_occurence = [0] * 90
    digit_percent = [0] * 90
    first_two_digits = 0
    
    for x in input_data:
        try:
            first_two_digits = int(x[0:2])
            digit_occurence[first_two_digits - 10] += 1

        except:
            pass


    #Convert frequncies to percentage expressed as a decimal. 
    print("[Debug] Converting to percentages")

    for x in range(0, len(digit_percent)):
        digit_percent[x] = float(digit_occurence[x] / len(input_data))

    #Calcuate perfect Benford distribution.
    print("[Debug] Computing ideal Benford frequency")
    benford_percent = benford_distribution(1, 0)

    benford_occurence = []

    for x in benford_percent:
        benford_occurence.append(round(x * len(input_data)))

    #Compute z statistic for this data:
    print("[Debug] Computing Z statistic")
    z_stat = []
    for x in range(0, len(digit_percent)):
        z_stat.append(compute_z_statistic(digit_percent[x], benford_percent[x], len(input_data)))


    #Compute von-mises statistics
    von_mises_stat = compute_von_mises(benford_occurence, digit_percent, benford_percent, len(input_data))

    #Compute d* statistic
    d_star_stat = compute_dstar(digit_percent, benford_percent, len(input_data))

    return(digit_occurence, benford_occurence, digit_percent, benford_percent, z_stat, von_mises_stat, d_star_stat)







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









### ---------------------------------- BENDFORD DISTRIBUTIONS--------------------------------------------- ###






#Calculate the Ideal Benford distribution
def benford_distribution(mode, size):

    if mode == 0:
        benford_frequency_first_digit = []
        for x in range(1,10):
            benford_frequency_first_digit.append(math.log10(1 + 1/x))

        return(benford_frequency_first_digit)

    elif mode == 1 or mode == 2:
        benford_frequency_first_second_digit = []
        for x in range(1,10):
            for y in range(0,10):
                benford_frequency_first_second_digit.append(math.log10(x + (y+1)/10) - math.log10(x + y/10))

        return(benford_frequency_first_second_digit)

    elif mode == 3:
        benford_frequency_first_second_digit = []
        benford_frequency_second_digit = [0] * 10
        for x in range(1,10):
            for y in range(0,10):
                benford_frequency_first_second_digit.append(math.log10(x + (y+1)/10) - math.log10(x + y/10))

        for x in range(0, len(benford_frequency_first_second_digit)):
            # print(int(str(x + 10)[1]))
            benford_frequency_second_digit[int(str(x + 10)[1])] += benford_frequency_first_second_digit[x]

        return(benford_frequency_second_digit)


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





### ---------------------------------- FINITE RANGE --------------------------------------------- ##





#First Digit Finite range Benford's law
def first_digit_benford_finite_range(input_data):
    print("[Debug] Calculating first digit frequency")
    #Calcuate perfect Benford distribution.
    print("[Debug] Computing ideal Benford frequency")
    
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


#Empricical second digit finite range
def second_digit_finite_range():
    return(0)





### ---------------------------------- GOF TESTS --------------------------------------------- ###






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
    





### ---------------------------------- PLOT DATA --------------------------------------------- ###







def plot_bar_chart(bins, frequency, benford_freq, dataset_size, von_mises, dstar, mode):
    #increase font size
    plt.rcParams.update({'font.size': 12})

    #Compute errors
    yerror = []
    for x in range(0, len(frequency)):
        yerror.append(math.sqrt(benford_freq[x]))

    #normalised residuals and colours
    difference = []
    y_colours = []
    for x in range(0, len(yerror)):
        if yerror[x] == 0:
            yerror[x] =1
        difference.append((frequency[x] - benford_freq[x]) / yerror[x])
        if abs(difference[x]) > 1:
            y_colours.append('firebrick')
        else:
            y_colours.append('green')

    #Output as histogram
    #First (main) subplot

    if mode in [0,-1]:
        ind = np.arange(9)
    elif mode == 1:
        ind = np.arange(10, 100, 1)
    elif mode == 3:
        ind = np.arange(0, 10, 1)

    width = 0.7


    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    ax0 = plt.subplot(gs[0])

    if mode in [0, -1]:
        ax0.errorbar(bins, benford_freq, yerr=yerror, label="Expected Occurrence", color='black', marker='x', fmt='x', capsize=3, elinewidth=1, zorder=1)
    elif mode == 1:
        ax0.errorbar(bins, benford_freq, yerr=yerror, label="Expected Occurrence", color='black', marker='.', fmt='x', capsize=2, elinewidth=1, zorder=1)
    elif mode == 3:
        ax0.errorbar(bins, benford_freq, yerr=yerror, label="Expected Occurrence", color='black', marker='x', fmt='x', capsize=3, elinewidth=1, zorder=1)
    

    ax0.bar(ind, frequency, width, color='grey', label="Observed Occurrence", zorder=-1)

    plt.xlabel("Digit Value")
    plt.ylabel("Observed Occurence")
    plt.xticks(ind, "")


    if mode == 1:
        #plt.xticks(np.arange(10, 100, 5))
        plt.xlim(9,100)

    
    
    if mode != 3:#Format Legend
        patch = []
        handles, labels = ax0.get_legend_handles_labels()
        patch.append(mpatches.Patch(color='green', label=r'$|\sigma|$ < 1'))
        patch.append(mpatches.Patch(color='firebrick', label=r'$|\sigma|$ > 1'))
        patch.append(mpatches.Patch(color='none', label=r'${}$'.format(von_mises)))
        patch.append(mpatches.Patch(color='none', label=r'${}$'.format(dstar)))
        patch.append(mpatches.Patch(color='none', label=r'$N = {}$'.format(str(dataset_size))))
    
    else:
        patch = []
        handles, labels = ax0.get_legend_handles_labels()
        patch.append(mpatches.Patch(color='green', label=r'$|\sigma|$ < 1'))
        patch.append(mpatches.Patch(color='firebrick', label=r'$|\sigma|$ > 1'))
        #patch.append(mpatches.Patch(color='white', label=r'${}$'.format(von_mises)))
        patch.append(mpatches.Patch(color='none', label=r'${}$, $N={}$'.format(dstar, str(dataset_size))))
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
    ax1.bar(ind, difference, 0.70, color=y_colours)

    plt.xticks(ind, bins)
    

    #format spacing on graph

    if y_range > 2 and y_range <= 3:
        plt.yticks((-y_range + 1, 0, y_range - 1))
    elif y_range > 3 and y_range <= 6:
        plt.yticks((-y_range + 1, 0, y_range - 1))
        #y_range = y_range + 1
    elif y_range > 6:
        plt.yticks((-y_range + 3, 0, y_range - 3))
        #y_range = y_range + 2
    ax1.set_ylim([-y_range,y_range])

    if mode in [0, -1]:
        plt.xlabel("First Digit Value")
        plt.ylabel("Normalised Residual")
        plt.ylim(-y_range - 0.75, y_range + 0.75) 

    elif mode == 1:
        plt.xlabel("First Two Digit Values")
        plt.ylabel("Normalised Residual")
        plt.ylim(-y_range - 0.75, y_range + 0.75) 
        plt.xticks(np.arange(10, 100, 5))
        plt.xlim(9,100)

    elif mode == 3:
        plt.xlabel("Second Digit Value")
        plt.ylabel("Normalised Residual")
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


def plot_heat_map(frequency, benford_freq):
    #increase font size
    plt.rcParams.update({'font.size': 12})

    #Setup x and y axis arrays
    y_axis = np.arange(1, 11, 1)
    x_axis = np.arange(0, 10, 1)

    values_to_plot = []
    row = []

    #Compute absolute difference as numpy array to plot
    for x in range(1,10):
        for y in range(0,10):
            row.append(int(frequency[int(str(x) + str(y)) - 10] - benford_freq[int(str(x) + str(y)) - 10]))

        array = np.asarray(row)

        if x == 1:
            values_to_plot = array
        else:
            values_to_plot = np.vstack((values_to_plot, array))
        row = []

    limit = roundup(max(abs(np.amin(values_to_plot)), np.amax(values_to_plot)))
    
    test = [-limit, limit, limit, limit, limit, limit, limit, limit, limit, limit]
    array = np.asarray(test)
   
    values_to_plot = np.vstack((values_to_plot, array))


    fig, ax = plt.subplots()

    im, cbar = heatmap(limit, values_to_plot, y_axis , x_axis, ax=ax,
                   cmap="coolwarm", cbarlabel="Deviation")
    texts = annotate_heatmap(im, valfmt="{x}")

    fig.tight_layout()
    print('[Debug] Saving Plot as {}'.format(sys.argv[3]))
    plt.savefig('{}'.format(sys.argv[3]), bbox_inches='tight')
    return(0)






# Credit for the next two functions https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html. These have been mildly edited 
# to suit the needs of the program. 

def heatmap(limit, data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
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
    ticks = []
    for x in range(-limit, limit + 10, 10):
        ticks.append(x)

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, ticks=ticks)
    cbar.ax.set_ylim(-limit, limit) 

    #cbar.clim(-60, 60)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
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

    plt.xlabel("Second Digit")
    plt.ylabel("First Digit")

    ax.set_xticks(np.arange(data.shape[1]+1), minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1), minor=True)
    ax.set_xlim(row_labels[0] - 1.5, row_labels[-1] - 0.5)
    ax.set_ylim(col_labels[-1] - 0.5, col_labels[0] - 0.5)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

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

    


### ---------------------------------- Main() --------------------------------------------- ###

#main function of the program. 
def main(mode):
    #Process mode of analysis
    try:
        int(mode)
    except:
        print("[Fatal Error] Cannot process mode", str(mode), ". Please enter a valid mode of analysis.")
        usage()
        exit()

    if mode < -1 or mode > 3:
        print("[Fatal Error] Cannot process mode", str(mode), ". Please enter a valid mode of analysis.")
        usage()
        exit()


    #Import data from argv[1]
    filename = sys.argv[1]
    data = input_numbers(filename)
    print("[Debug] Starting First Digit Analysis")

    if mode == 0:
        data_raw, benford_raw, data_percent, benford_percent, z_statistic, von_mises_statistic, d_star_statistic = first_digit_test(data)
        bins_to_plot = ['1','2','3','4','5','6','7','8','9']
    elif mode == 1 or mode == 2:
        data_raw, benford_raw, data_percent, benford_percent, z_statistic, von_mises_statistic, d_star_statistic = first_second_digit_test(data)
        bins_to_plot = []
        for x in range(10, 100):
            bins_to_plot.append(x)
    elif mode == 3:
        data_raw, benford_raw, data_percent, benford_percent, z_statistic, von_mises_statistic, d_star_statistic = second_digit_test(data)
        bins_to_plot = []
        for x in range(0, 10):
            bins_to_plot.append(x)
    elif mode == -1:
        data_raw, benford_raw, data_percent, benford_percent, z_statistic, von_mises_statistic, d_star_statistic, data_size = first_digit_benford_finite_range(data)
        bins_to_plot = ['1','2','3','4','5','6','7','8','9']

    #Output results
    print("[Debug] Analysis complete. Outputing results.")
    print("\n\n###--- Analysis for", filename, "---###\n")

    if mode == 0:
        output_first_digit_test(data_raw, benford_raw, z_statistic)
    elif mode == 1:
        output_first_second_digit_test(data_raw, benford_raw, z_statistic)
    elif mode == 2:
        output_first_second_digit_test(data_raw, benford_raw, z_statistic)
        plot_heat_map(data_raw, benford_raw)
        exit()
    elif mode == 3:
        output_second_digit_test(data_raw, benford_raw, z_statistic)
    elif mode == -1:
        output_first_digit_test(data_raw, benford_raw, z_statistic)

    if mode >= 0:
        data_size = len(data)



    print("Cramer-von Mises test: {} {} {}".format(von_mises_statistic[0],von_mises_statistic[1],von_mises_statistic[2]))
    print("d* test: {}, {}{}".format(d_star_statistic[0], d_star_statistic[1], d_star_statistic[2]))
    #Legend significance levels
    print("\n * significant at the .05 level\n** significant at the .01 level\n")

    #Output plot
    print("[Debug] Generating Plot of the data.")

    #if mode in [0,1,2,3]:
    plot_bar_chart(bins_to_plot, data_raw, benford_raw, data_size, von_mises_statistic[2], d_star_statistic[1], mode)

    #elif mode in [-1]:


    print("[Debug] Output complete. Exiting.")
    exit()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage()
        exit()
    else:
        main(int(sys.argv[2]))
    
