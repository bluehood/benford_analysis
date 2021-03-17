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


def usage():
    print(f'Synthetic Finite Range Benford Law Test.\n')
    print(sys.argv[0], "<file_to_test>", "<mode>", "<figure_filename>", "<lower_limit>", "<upper_limit>")
    print("\nModes:\n 2    Second Digit Finite Range")
    print(" 3    Third Digit Finite Range")
    return(0)


def remove_leading_zeros_dots(foo):
    # local variable
    index = 0

    # if foo is null ignore it 
    if foo == '':
        return(0)

    # return most significant bits without decimal points (not needed now)
    for j in range(0, len(foo)):
        if foo[j] in ['0', '.']:
            index += 1
        else:
            break

    return(foo[index:].replace('.', ''))

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

    # Remove all null values and leading zeros. Convert to a float w/ 3dp to account for COMPUSTAT data. 
    # We will later normalise all data to this convention
    # Save this result in input_data_sanitised
    print("[Debug] Sanitising Input Data")

    try:
        input_data.remove('0')
        input_data.remove('0.')
    except:
        pass

    input_data_sanitised = input_data

    for x in range(0, len(input_data)):
        input_data_sanitised[x] = str(input_data[x])
        # input_data_sanitised[x] = str(int(int(input_data_sanitised[x].replace('.', '')) / 1000))
        # while input_data_sanitised[x][0] == "0":
        #         input_data_sanitised[x] = input_data_sanitised[x][1:]
        #         if input_data_sanitised[x] == '':
        #             break
        # if len(input_data_sanitised[x]) == 1:
        #     input_data_sanitised[x] = input_data_sanitised[x] + "0"

    # Remove all null entries from input_data_sanitised
    
    input_data_sanitised = ' '.join(input_data_sanitised).split()
    # print(input_data_sanitised)
    print("[Debug] Input Data Sanitised Successfully")  
    return(input_data_sanitised) 






### --------------------------------------- SECOND DIGIT TEST --------------------------------------------- ###



#Output the results of first_digit_test to file argv[2]
def output_second_digit_test(digit_occurance, benford_occurance, z_stat):
    #Round output figures to 3.d.p. Output relevant entries as percentages.
    digit_frequency = []
    benford_frequency = []
    for x in range(0,len(digit_occurance)):
        digit_frequency.append(str(round(digit_occurance[x])))
        benford_frequency.append(str(round(benford_occurance[x])))
        z_stat[x] = '{:.3f}'.format(z_stat[x])

    #Identify significant deviations based on Z statistic. 
    for x in range(0, len(z_stat)):
        if float(z_stat[x]) >= 1.96 and float(z_stat[x]) < 2.576:
            z_stat[x] = z_stat[x] + " *"
        elif float(z_stat[x]) >= 2.576:
            z_stat[x] = z_stat[x] + " **"
    
    print("Digit        Observed Distribution Occurance        Synthetic Distribution Occurance        Z-Statistic")
    print("-----------------------------------------------------------------------------------------------")

    #write results to the file with table formats. 
    for x in range(0,len(z_stat)):
        line = ""
        line = str(x) + '&' + " " * (8 + len("Digit") - len(str(x)) - 1)
        line += benford_frequency[x] + '&' + " " * (8 + len("Synthetic Distribution Occurance") - len(benford_frequency[x]))
        line += digit_frequency[x] + '&' + " " * (8 + len("Observed Distribution Occurance") - len(digit_frequency[x]))
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

def second_digit_analysis(set_to_test, lower_lim, upper_lim):
    # local occurence variable
    occurence = [0] * 10
    
    # loop through local set_to_test and calculate second digit finite range occurence
    for x in range(0, len(set_to_test)):
        #print(int(set_to_test[x]), len(set_to_test))
        try:
            # determine whether current entry in finite range
            if lower_lim <= float(set_to_test[x]) and upper_lim >= float(set_to_test[x]):
                # extract significant bits of data
                sanitised_entry = remove_leading_zeros_dots(set_to_test[x])
                # ignore entries that do not have at least 2 sig figs
                if len(sanitised_entry) >= 2:
                    # add to local occurence variable
                    occurence[int(sanitised_entry[1])] += 1

        except Exception as e: print(e)
    
    return(occurence)

def second_digit_test(input_data_raw, benford_distribution_expectation):
    #Calculate the frequency of each character {0,1,2,...,9} in the second digit. 
    print("[Debug] Calculating second digit frequency")
    digit_frequency = [0] * 10
    second_digit = 0
    for x in input_data_raw:
        try:
            second_digit = int(x[1])
            digit_frequency[second_digit] += 1
        except:
            #account for single digit values
            continue
        

    #Convert frequncies to percentage expressed as a decimal. 
    print("[Debug] Converting to percentages")
    digit_frequency_percent = [0] * 10

    for x in range(0, len(digit_frequency)):
        digit_frequency_percent[x] = float(digit_frequency[x] / len(input_data_raw))

    #Compute Benford distribution for data of length equal to dataset
    benford_raw = []

    for x in benford_distribution_expectation:
        benford_raw.append(float(x * len(input_data_raw)))

    #Compute Z statistic for this data:
    print("[Debug] Computing Z statistic")
    z_stat = []
    for x in range(0, len(digit_frequency)):
        z_stat.append(compute_z_statistic(digit_frequency_percent[x], benford_distribution_expectation[x], len(input_data_raw)))

    #Compute von-mises statistics
    von_mises_stat = compute_von_mises(benford_raw, digit_frequency, benford_distribution_expectation, len(input_data_raw))

    #Compute d* statistic
    d_star_stat = compute_dstar(digit_frequency_percent, benford_distribution_expectation, len(input_data_raw))

    return(digit_frequency, benford_raw, digit_frequency_percent, benford_distribution_expectation, z_stat, von_mises_stat, d_star_stat)




### --------------------------------------- THIRD DIGIT TEST --------------------------------------------- ###





def third_digit_analysis(set_to_test, lower_lim, upper_lim):
    # local occurence variable
    occurence = [0] * 10
    
    # loop through local set_to_test and calculate third digit finite range occurence
    for x in range(0, len(set_to_test)):
        #print(int(set_to_test[x]), len(set_to_test))
        try:
            # determine whether current entry in finite range
            if lower_lim <= float(set_to_test[x]) and upper_lim >= float(set_to_test[x]):
                # extract significant bits of data
                sanitised_entry = remove_leading_zeros_dots(set_to_test[x])
                # ignore entries that do not have at least 2 sig figs
                if len(sanitised_entry) >= 3:
                    # add to local occurence variable
                    occurence[int(sanitised_entry[2])] += 1

        except Exception as e: print(e)
    
    return(occurence)

def third_digit_test(input_data_raw, benford_distribution_expectation):
    #Calculate the frequency of each character {0,1,2,...,9} in the third digit. 
    print("[Debug] Calculating third digit frequency")
    digit_frequency = [0] * 10
    second_digit = 0
    for x in input_data_raw:
        try:
            second_digit = int(x[2])
            digit_frequency[second_digit] += 1
        except:
            #account for single and two digit values
            continue
        

    #Convert frequncies to percentage expressed as a decimal. 
    print("[Debug] Converting to percentages")
    digit_frequency_percent = [0] * 10

    for x in range(0, len(digit_frequency)):
        digit_frequency_percent[x] = float(digit_frequency[x] / len(input_data_raw))

    #Compute Benford distribution for data of length equal to dataset
    benford_raw = []

    for x in benford_distribution_expectation:
        benford_raw.append(float(x * len(input_data_raw)))

    #Compute Z statistic for this data:
    print("[Debug] Computing Z statistic")
    z_stat = []
    for x in range(0, len(digit_frequency)):
        z_stat.append(compute_z_statistic(digit_frequency_percent[x], benford_distribution_expectation[x], len(input_data_raw)))

    #Compute von-mises statistics
    von_mises_stat = compute_von_mises(benford_raw, digit_frequency, benford_distribution_expectation, len(input_data_raw))

    #Compute d* statistic
    d_star_stat = compute_dstar(digit_frequency_percent, benford_distribution_expectation, len(input_data_raw))

    return(digit_frequency, benford_raw, digit_frequency_percent, benford_distribution_expectation, z_stat, von_mises_stat, d_star_stat)





### --------------------------------------- Generate and Import Synthetic Benford Set --------------------------------------------- ###


def generate_benford_set_from_c_program(lower, upper, size_set):
    call(["generate_benford", "/tmp/generate_benford_output.txt", str(size_set), str(lower), str(upper)])
    return(0)

def generate_benford_set_from_py_program(lower, upper, size_set):
    call(["generate_benford_geometric.py", "/tmp/generate_benford_output.txt", str(size_set), str(lower), str(upper)])
    return(0)

def import_process_benford_set(size_set, lower, upper, mode):
    # Location of saved benford set 
    filename = "/tmp/generate_benford_output.txt"
    # local variables
    benford_set_raw = []
    if mode in ['2', '3']:
        benford_set_counts = [0] * 10
    
    return_size = 0

    # open file for reading
    filehandle = open(filename, 'r')

    for i in range(0, size_set + 1):
        # read a single line and append to benford_set_raw
        line = filehandle.readline().replace('\n', '')

        # check that the line is not empty
        if line != '':
            benford_set_raw.append(line.split('.')[0])
        
        # every 10000 lines determine second digit occurence of Benford subset in finite range
        if i % 10000 == 0 and i != 0:
            # print(benford_set_raw)
            # exit()
            if mode == '2':
                observed_counts = second_digit_analysis(benford_set_raw, lower, upper)
            elif mode == '3':
                observed_counts = third_digit_analysis(benford_set_raw, lower, upper)
            # print(observed_counts)

            # calculate local number of entries in finite range
            for j in observed_counts:
                return_size += j

            # add to global count of digits
            for x in range(0, len(benford_set_counts)):
                benford_set_counts[x] += observed_counts[x]
            
            # set local counts to zero
            benford_set_raw = []
            continue

        # analyses remaining entries. Determine second digit occurence of Benford subset in finite range
        if i == size_set:
            observed_counts = second_digit_analysis(benford_set_raw, lower, upper)
            # print(len(observed_counts))

             # calculate local number of entries in finite range
            for j in observed_counts:
                return_size += j

            # add to global count and exit for loop
            for x in range(0, len(benford_set_counts)):
                benford_set_counts[x] += observed_counts[x]
            break
    
    # close the file handle
    filehandle.close()
    
    # verify corrrect normalisation
    print(f"[Debug] Size of benford set in range [{lower}, {upper}] is {return_size}")
    total = 0

    print(benford_set_counts)

    for z in benford_set_counts:
        total += z / return_size

    print(f"[Test] Normalisation (should be one) {total}")

    # return synthetic finite range 
    return([float(y / return_size) for y in benford_set_counts])


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
    
    ind = np.arange(0, 10, 1)
    width = 0.7

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    ax0 = plt.subplot(gs[0])
    
    ax0.errorbar(bins, benford_freq, yerr=yerror, label="Synthetic Occurrence", color='black', marker='x', fmt='x', capsize=3, elinewidth=1, zorder=1)
    ax0.bar(ind, frequency, width, color='grey', label="Observed Occurrence", zorder=-1)

    plt.xlabel("Digit Value")
    plt.ylabel("Observed Occurence")
    plt.xticks(ind, "")
    
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
    elif y_range > 6 and y_range <= 10:
        plt.yticks((-y_range + 3, 0, y_range - 3))
        #y_range = y_range + 2
    elif y_range > 10:
        plt.yticks((-int(y_range/2), 0, int(y_range/2)))

    
    plt.xlabel("Second Digit Value")
    plt.ylabel("Normalised Residual")
    plt.ylim(-y_range - 0.75, y_range + 0.75) 

    #format graph in general
    plt.axhline(linewidth=0.5, color='black')

    if y_range < 10:
        plt.axhline(y=1, linewidth=0.75, color='black', linestyle='--')
        plt.axhline(y=-1, linewidth=0.75, color='black', linestyle='--')

    #plt.legend(handles=legend_elements, loc='best')
    fig.align_ylabels()
    print('[Debug] Saving Plot as {}'.format(sys.argv[3]))
    plt.savefig('{}'.format(sys.argv[3]), bbox_inches='tight')
    return(0)



### --------------------------------------- MAIN --------------------------------------------- ###


def main(mode):
    mode = str(mode)
    # Import test data
    test_data = input_data_from_file(sys.argv[1])
    print(f"[Debug] Imported test set from {sys.argv[1]}")
    print(f"[Debug] Reduce to range specified by the user.")
    
    # Select data in specific range. Obtain lower/upper limit form cli arguements
    lowerlimit = int(sys.argv[4])
    upperlimit = int(sys.argv[5])
    test_data = cut_data_range(lowerlimit, upperlimit, test_data)

    # Convert limits to standard form
    lower = '{:e}'.format(lowerlimit)
    upper = '{:e}'.format(upperlimit)

    # Calculate lower magnitude and upper magnitude from standard form. +1 for off-by-one-error
    lower_mag = int(str(lower).split('e')[1]) + 1
    upper_mag = int(str(upper).split('e')[1]) + 1
    # print(f"{lower_mag} < x < {upper_mag}")

    # generate synthetic Benford set of a calculated size
    # size = (len(test_data) * 10) - (len(test_data) * 10 % 1000)
    size = len(test_data) * 100 - len(test_data) * 100 % 1000

    if size > 10000000:
        # print("here")
        size = 10000000

    # print(size)
    print(f"[Debug] Generating Benford set of size {size}. This could take a while zzz")
    #exit()
    # generate_benford_set_from_c_program(lower_mag, upper_mag, size)
    generate_benford_set_from_py_program(10**(lower_mag - 1), 10**(upper_mag), size)

    # Compute the expected distribution from the imported Benford set in our range
    beford_distribution_expectation = import_process_benford_set(size, lowerlimit, upperlimit, mode)
    print(f"[Test] Benford Expectation Values {beford_distribution_expectation}")
    
    # Perfrom GFRBL
    if mode == '2':
        test_data_raw, benford_data_raw, test_data_expectation, benford_data_expectation, z_statistic, von_mises_statistic, d_star_statistic = second_digit_test([str(y) for y in test_data], beford_distribution_expectation)
    elif mode == '3':
        test_data_raw, benford_data_raw, test_data_expectation, benford_data_expectation, z_statistic, von_mises_statistic, d_star_statistic = third_digit_test([str(y) for y in test_data], beford_distribution_expectation)

    # print(test_data_raw, "\n", benford_data_raw)
    
    # setup data to plot
    bins_to_plot = []

    if mode in ['2', '3']:
        for x in range(0, 10):
            bins_to_plot.append(x)

    # output second digit test and statistics
    output_second_digit_test(benford_data_raw, test_data_raw, z_statistic)
    print("Cramer-von Mises test: {} {} {}".format(von_mises_statistic[0],von_mises_statistic[1],von_mises_statistic[2]))
    print("d* test: {}, {}{}".format(d_star_statistic[0], d_star_statistic[1], d_star_statistic[2]))
    #Legend significance levels
    print("\n * significant at the .05 level\n** significant at the .01 level\n")

    #Output plot
    print("[Debug] Generating Plot of the data.")
    plot_bar_chart(bins_to_plot, test_data_raw, benford_data_raw, len(test_data) , von_mises_statistic[2], d_star_statistic[1], 3)

    print("[Debug] Output complete. Exiting.")
    exit()

    return(0)


if __name__ == '__main__':
    # print(remove_leading_zeros_dots('11.333'))
    # print(remove_leading_zeros_dots('0.077'))
    # print(remove_leading_zeros_dots('.083'))
    # print(remove_leading_zeros_dots('1200'))
    # print(remove_leading_zeros_dots(''))
    if len(sys.argv) != 6:
        usage()
        exit()
    else:
        main(int(sys.argv[2]))