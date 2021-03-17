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
from sigfig import round
import random

def usage():
    print(f'Calculate test statistics for Benford Sets generated using geometric series in a randomly selected finite range. There is also the option to plot these test statistics agasint the range considered. (Note the lowerbound is fixed and the upperbound randomly selected.)\n')
    print(f'To process Benford Set and compute test statistics: {sys.argv[0]} <mode>')
    print(f'mode: <digit_test><test_statistic> where digit_test is in [1,2] and test_statistic is in [X_2, d, A].\n')
    print(f'To plot test statistics: {sys.argv[0]} plot <datafile> <savefile (.png)>\n')
    return(0)

# ------ Generate and Import Bendford set 

# Generate benford set using Python geometric series
def generate_benford_set_from_py_program():
    call(["generate_benford_geometric.py", "/tmp/generate_benford_output.txt", str(1000000), str(10**2), str(10**6)])
    return(0)

def import_process_benford_set():
    # Location of saved benford set 
    filename = "/tmp/generate_benford_output.txt"
    # Open contents of filename and write to the list lines
    with open(filename) as f:
        lines = f.read().splitlines()
    return(lines)

def export_results(export_filename, sec_report_extracted_data):
    # Write to file
    with open(export_filename, 'w') as f:
        for item in sec_report_extracted_data:
            f.write("%s\n" % item)

    return(0)



# ------ Ordinary Benford tests



# Benford distribution
def benford_distribution(mode, size):
    # print("Here1")
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
        # print("Here2")
        benford_frequency_first_second_digit = []
        benford_frequency_second_digit = [0] * 10
        for x in range(10,100):
            benford_frequency_first_second_digit.append(np.log10(1 + 1/x))

        for x in range(0, len(benford_frequency_first_second_digit)):
            benford_frequency_second_digit[int(str(x + 10)[1])] += benford_frequency_first_second_digit[x]

        # print(benford_frequency_first_second_digit)
        # print(benford_frequency_second_digit)

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

#Perform the digit test on the data.
def digit_test(input_data, mode, test_statistic):
    # print("Hello")
    #Calculate the frequency of each character set in indexes specified.
    if mode in ['2']:
        # print("[Debug] Calculating second digit frequency")
        digit_frequency = [0] * 10
        offset = 0

    elif mode in ['1']:
        # print("[Debug] Calculating first digit frequency")
        digit_frequency = [0] * 9
        offset = 1
    
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
        benford_frequency.append(x * len(input_data))

    if test_statistic == 'X_2':
        X_squared = compute_chi_squared_statistic(benford_frequency, digit_frequency)
        return(X_squared)
    elif test_statistic == 'd':
        d_star_stat = compute_dstar(digit_frequency_percent, benford_frequency_percent, len(input_data))
        return(d_star_stat)
    elif test_statistic == 'A':
        von_mises_stat = compute_von_mises(benford_frequency, digit_frequency, benford_frequency_percent, len(input_data))
        return(von_mises_stat)

# ------ Finite range Benford tests

#Finite range Benford's law
def benford_finite_range(input_data, mode, test_statistic):
    # print("[Debug] Calculating first digit frequency")
    #Calcuate perfect Benford distribution.
    # print("[Debug] Computing ideal Benford frequency")
    
    benford_frequency, digit_frequency, dataset_size = finite_range_benford_distribution(input_data, mode)

    #Compute Benford distribution for data of length equal to dataset
    benford_frequency_percent = []
    digit_frequency_percent = []

    for x in benford_frequency:
        benford_frequency_percent.append(float(x / dataset_size))

    #Compute digit frequency percent 
    for x in digit_frequency:
        digit_frequency_percent.append(float(x / dataset_size))

    if test_statistic == 'X_2':
        X_squared = compute_chi_squared_statistic(benford_frequency, digit_frequency)
        return(X_squared)
    elif test_statistic == 'd':
        d_star_stat = compute_dstar(digit_frequency_percent, benford_frequency_percent, len(input_data))
        return(d_star_stat)
    elif test_statistic == 'A':
        von_mises_stat = compute_von_mises(benford_frequency, digit_frequency, benford_frequency_percent, len(input_data))
        return(von_mises_stat)

#Calculate finite range Benford distributions
def finite_range_benford_distribution(data, mode):
    #Finite range calculation
    P = []
    size = len(data)
    data_float = [ float(x) for x in data ]
    lowerlimit = min(data_float)
    upperlimit = max(data_float)

    #Calculate the frequency of each digit
    if mode == 'f1':
        digit_frequency_observed = [0] * 9

        for x in data_float:
            digit_frequency_observed[int(str(x)[0]) - 1] += 1

    elif mode == 'f2':
        digit_frequency_observed = [0] * 10

        for x in data_float:
            digit_frequency_observed[int(str(x)[1])] += 1


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
            

        
        return(P, digit_frequency_observed, size)

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
            P_D = 1/lambda_c * ((beta - alpha - 1) * lambda_d + lambda_a + lambda_b)
            
            # Append to distribution
            P.append(P_D * size)

        return(P, digit_frequency_observed, size)



# ----- Test stats
#Calculate X^2 statistic
def compute_chi_squared_statistic(expected_list, actual_list):
    #Multiply by the total to achieve an approximation of the actual and expected observations
    chi = 0
    for x in range(0, len(expected_list)):
        entry = (float(expected_list[x]) - float(actual_list[x])) ** 2
        entry = entry / (expected_list[x])
        chi += entry

    return_value = chi

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

    return(d_star_morrow)

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

    H = []
    for x in range(0, len(expected_cdf)):
        H.append(expected_cdf[x] / size)

    #Compute A^2
    summation = 0
    for j in range(0, len(Z) - 1):
        
        if (H[j] * (1 - H[j])) == 0:
            continue
            
        else:
            summation += ((Z[j] ** 2) * benford_probability[j]) / (H[j] * (1 - H[j]))

    A_squared = (1/size) * summation

    return(A_squared)
    

def plot_result(x_axis, y_axis1, y_axis2, mode, test):
    # mean of test statistic. Round to two significant figures
    X2_mean = round(float(sum(y_axis2) / len(y_axis2)), 2)

    # calculate magnitude of test statistic mean
    test_statistic_magnitude = math.floor(np.log10(X2_mean))

    # Remove leading zeros from test statistic. 
    X2_leading_digits = ''
    for x in range(0, len(str(X2_mean))):
        if str(X2_mean)[x] != '0' and str(X2_mean)[x] != '.':
            X2_leading_digits = str(X2_mean)[x - 1] + '.' + str(X2_mean)[x:]

    # format test statistic mean for display
    test_statistic_format = f'${str(X2_leading_digits)} \\times 10^' + '{' +  f'{test_statistic_magnitude}' + '}$'

    # Font size
    plt.rcParams.update({'font.size': 13})

    # Define figure
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    # Calculate zoomed in portion of data 
    x_axis_zoomed = []
    y_axis1_zoomed = []
    y_axis2_zoomed = []
    
    # format zoomed subplot points 
    for x in range(0, len(x_axis)):
        if x_axis[x] >= 4 and x_axis[x] <= 6:
            x_axis_zoomed.append(x_axis[x])
            y_axis1_zoomed.append(y_axis1[x])
            y_axis2_zoomed.append(y_axis2[x])

    # Seperate between testing cases
    if test == True:
        plt.xlim(1,2)
        ticks_labels = []
        for x in range(10, 21):
            ticks_labels.append(x/10)
        plt.xticks(ticks_labels, ticks_labels)

        # vertical grey lines
        for value in range(10, 21):
            plt.axvline(x=value / 10, linewidth=0.75, color='grey', linestyle='--')

        # Horizontal line at x2_mean
        plt.axhline(y=X2_mean, linewidth=1, color='black')

    else:
        # xlimit
        plt.xlim(0.8, 10.2)

        # define x ticks
        ticks_labels = [1,2,3,4,5,6,7,8,9,10]
        plt.xticks(ticks_labels, ticks_labels)
        
        # vertical grey lines
        for value in range(1, 10 + 1):
            plt.axvline(x=value, linewidth=0.75, color='grey', linestyle='--')

        # Horizontal line at x2_mean
        plt.axhline(y=X2_mean, linewidth=1, color='black')
    
    # Plot data

    ax.scatter(x_axis, y_axis1, label="Benford's Law", color='royalblue', s=3, zorder=1)
    ax.scatter(x_axis, y_axis2, label="FR Benford's Law", color='firebrick', s=3, zorder=1)


    # Draw lines from main figure to zoomed in subfigure
    if mode == 'd':
        x_values_one = [4, 6.2]
        y_values_one = [0, 5.5]

        x_values_two = [6, 9.5]
        y_values_two = [0, 5.4]

        plt.plot(x_values_one, y_values_one, linewidth=0.75, color='grey')
        plt.plot(x_values_two, y_values_two, linewidth=0.75, color='grey')

    elif mode == 'X_2':
        x_values_one = [4, 5.6]
        y_values_one = [0, 480]

        x_values_two = [6, 9.45]
        y_values_two = [0, 460]

        plt.plot(x_values_one, y_values_one, linewidth=0.75, color='grey')
        plt.plot(x_values_two, y_values_two, linewidth=0.75, color='grey')

    elif mode == 'A':
        x_values_one = [4, 5.5]
        y_values_one = [0, 190]

        x_values_two = [6, 9.45]
        y_values_two = [0, 190]

        plt.plot(x_values_one, y_values_one, linewidth=0.75, color='grey')
        plt.plot(x_values_two, y_values_two, linewidth=0.75, color='grey')

    #Format tick labels in scientific notation
    # yticks = ax.get_yticks()
    # print(yticks)

    # #Find first non-zero integer
    # lower_index = 0
    # for x in range(0, len(yticks)):
    #     if yticks[x] > 0:
    #         lower_index = x
    #         break
    
    # # Compute difference in magnitude 
    # mag_diff = math.floor(np.log10(yticks[-1])) - math.floor(np.log10(math.floor(yticks[lower_index])))
    
    # # Compute magnitude to report on ylabel
    # mean_mag_diff = math.floor(np.log10(yticks[lower_index])) + mag_diff
    # report_mag_diff = f'10^{mean_mag_diff}'
    # # print(report_mag_diff)
    
    # # Divide x ticks by this magnitude to obtain fractional part
    # yticks_fractional = yticks / (10**mean_mag_diff)
    # # print(yticks_fractional)
    
    # # Set fractional ticks to tick labels
    # plt.yticks(yticks[lower_index - 1:-1], yticks_fractional[lower_index - 1:-1])

    # if mode == 'd':
    #     # label axis for d star
    #     plt.xlabel("b coefficent")
    #     plt.ylabel(r"$d^*$ (${}$)".format(report_mag_diff))

    # elif mode == 'X_2':
    #     # label axis for chi squared
    #     plt.xlabel("b coefficent")
    #     plt.ylabel(r"$\chi^2$ (${}$)".format(report_mag_diff))

    # elif mode == 'A':
    #     # label axis for A squared
    #     plt.xlabel("b coefficent")
    #     plt.ylabel(r"$A^2$ (${}$)".format(report_mag_diff))


    if mode == 'd':
        # label axis for d star
        plt.xlabel("b coefficent")
        plt.ylabel(r"$d^*$")

    elif mode == 'X_2':
        # label axis for chi squared
        plt.xlabel("b coefficent")
        plt.ylabel(r"$\chi^2$")

    elif mode == 'A':
        # label axis for A squared
        plt.xlabel("b coefficent")
        plt.ylabel(r"$A^2$")
    # Define Legend 
    patch = []
    # handles, labels = ax.get_legend_handles_labels()
    patch.append(mpatches.Patch(color='royalblue', label='Benford\'s Law'))
    patch.append(mpatches.Patch(color='firebrick', label='FR Benford\'s Law'))

    if mode == 'd':
        patch.append(mpatches.Patch(color='Black', label=r'Mean $d^*$ FR = {}'.format(test_statistic_format)))
    elif mode == 'X_2':
        patch.append(mpatches.Patch(color='Black', label=r'Mean $\chi^2$ FR = {}'.format(test_statistic_format)))
    elif mode == 'A':
        patch.append(mpatches.Patch(color='Black', label=r'Mean $A^2$ FR = {}'.format(test_statistic_format)))



     

    plt.legend(handles=patch, loc='best')

    # location of zoomed portion
    if test != True:
        if mode == 'd':
            sub_axes = plt.axes([.55,.4,.3,.3])
        elif mode == 'X_2':
            sub_axes = plt.axes([.5,.3,.35,.35])
        elif mode == 'A':
            sub_axes = plt.axes([.5,.3,.35,.35])
    
        # plot zoomed portion 
        sub_axes.scatter(x_axis_zoomed, y_axis1_zoomed, color='royalblue', s=3)
        sub_axes.scatter(x_axis_zoomed, y_axis2_zoomed, color='firebrick', s=3)
        # Horizontal line at x2_mean
        plt.axhline(y=X2_mean, linewidth=1, color='black')
        
        # Horziontal lines on subplots
        for value in range(4, 7):
            sub_axes.axvline(x=value, linewidth=0.75, color='grey', linestyle='--')

    plt.savefig('{}'.format(sys.argv[3]), bbox_inches='tight')
    plt.show()
    
    return(0)



def main(mode):
    
    if mode[0] in ['1', '2']:
        # Generate Benford set 
        generate_benford_set_from_py_program()
        synthetic_benford_set = import_process_benford_set()

        # Setup variables
        b_values = []
        test_statistic = []
        test_statistic_finite_range = []

        for x in range(0, 1000):
            # Randomly select b variable in [1, 10)
            b = random.uniform(1, 10)
            b_values.append(b)
            
            
            # Split benford set up to b * 10 ** 5
            for y in range(len(synthetic_benford_set) - 1, 0, -1):
                if float(synthetic_benford_set[y]) < b * 10**5:
                    split_benford_set = synthetic_benford_set[0:y]
                    break
            
            if x >= 0 and x < 10:
                print("Here")
                export_results(f'/tmp/generate_benford_output_{str(x)}.txt', split_benford_set)
                call(["benford.py", f'/tmp/generate_benford_output_{str(x)}.txt', '1',f'/tmp/fr_benford_output_{str(x)}.png'])
            
            continue
            
            if mode[1:] == 'X_2':
                # Compute X^2 statistic using ordinary second digit test
                test_statistic.append(digit_test(split_benford_set, mode[0], 'X_2'))
                
                # Compute X^2 statistic using finite range digit test
                test_statistic_finite_range.append(benford_finite_range(split_benford_set, f'f{mode[0]}', 'X_2'))

            elif mode[1:] == 'd':
                # Compute d star statistic using ordinary second digit test
                test_statistic.append(digit_test(split_benford_set, mode[0], 'd'))
                
                # Compute d star statistic using finite range digit test
                test_statistic_finite_range.append(benford_finite_range(split_benford_set, f'f{mode[0]}', 'd'))

            elif mode[1:] == 'A':
                # Compute A^2 statistic using ordinary second digit test
                test_statistic.append(digit_test(split_benford_set, mode[0], 'A'))
                
                # Compute A^2 star statistic using finite range digit test
                test_statistic_finite_range.append(benford_finite_range(split_benford_set, f'f{mode[0]}', 'A'))

            
        
        for y in range(0, len(b_values)):
            print(b_values[y], test_statistic[y], test_statistic_finite_range[y])

        # Append to file
        with open(sys.argv[2], 'a') as f:
            for x in range(0, len(b_values)):
                f.write(f"{b_values[x]},{test_statistic[x]},{test_statistic_finite_range[x]}\n")

    elif mode[0] == 'plot':

        filename = sys.argv[2]
        # Open contents of filename and write to the list lines
        with open(filename) as f:
            lines = f.read().splitlines()
        
        X = []
        Y1 = []
        Y2 = []

        for entry in lines:
            
            X.append(float(entry.split(',')[0]))
            Y1.append(float(entry.split(',')[1]))
            Y2.append(float(entry.split(',')[2]))


        plot_result(X, Y1, Y2, mode[1:], False)

    return(0)


if __name__ == '__main__':
    usage()
    main(sys.argv[1])