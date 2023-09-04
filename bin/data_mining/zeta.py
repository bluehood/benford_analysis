import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy.special
from matplotlib import gridspec
import matplotlib.patches as mpatches
from matplotlib import ticker
import argparse 


def getFirstSignificantDigit(number):
    # Convert the number to a string
    num_str = str(number)

    # Find the index of the first non-zero digit
    for char in num_str:
        if char.isdigit() and char != '0':
            return int(char)

    # If no non-zero digit is found, return 0
    return 0

def movingAverage(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

def checkNormalisation(set):
    normalisation = 0
    for x in set:
        normalisation = normalisation + x

    print(f'Normalised to {normalisation}')

    return

def compute_normalised_residuals(observed, expected):
    # Compute errors for expected distribution
    yerror = []
    for x in range(0, len(observed)):
        yerror.append(math.sqrt(observed[x]))
    
    # Calculate Normalised residuals
    difference = []
    y_colours = []
    for x in range(0, len(yerror)):
        if yerror[x] == 0:
            yerror[x] = 0.01

        #edge case for finite range second digit law
        if yerror[x] == 0.01:
            difference.append(0)
        else:
            difference.append((observed[x] - expected[x]) / yerror[x])
        
        if abs(difference[x]) > 1:
            y_colours.append('firebrick')
        else:
            y_colours.append('green')
        
    # print(difference)
    print(yerror)
    return(difference, y_colours, yerror)

def ChiSquared(observed):
    expected = {
        '1' : 0.3547514430742842,
        '2' : 0.1725053566247906,
        '3' : 0.12195303947017247,
        '4' : 0.07713882190756487,
        '5' : 0.1007579940811139,
        '6' : 0.05191670572170785,
        '7' : 0.046931694039928105,
        '8' : 0.038562420211001064,
        '9' : 0.03548252486949083
    }

    chi2 = 0
    for key in expected.keys():
        tmp = ((observed[key][-1] - expected[key])**2)/expected[key]
        chi2 = chi2 + tmp
    return chi2


def ChiSquaredBenford(observed):
    expected = {
        '1' : np.log10(2),
        '2' : np.log10(1+1/2),
        '3' : np.log10(1+1/3),
        '4' : np.log10(1+1/4),
        '5' : np.log10(1+1/5),
        '6' : np.log10(1+1/6),
        '7' : np.log10(1+1/7),
        '8' : np.log10(1+1/8),
        '9' : np.log10(1+1/9)
    }

    chi2 = 0
    for key in expected.keys():
        tmp = ((observed[key][-1] - expected[key])**2)/expected[key]
        chi2 = chi2 + tmp
    return chi2

def BenfordRatioGivenDigit(terms, digit, epsilon):
    possible_digits = [str(x) for x in range(1,10)]
    if digit not in possible_digits:
        return None
    
    numerator = 0
    denominator = 0
    series = []
    for n in range(1, int(terms) + 1):
        current_term = 1/(np.power(n, 1+epsilon)) # np.power(x, 2)
        denominator = denominator + current_term
        
        # Numerator
        first_digit = str(getFirstSignificantDigit(current_term))
        if first_digit == digit:
            numerator = numerator + current_term
            series.append(numerator/denominator)
        else:
            if len(series) > 0:
                tmp = series[-1]
            else:
                tmp = 0
            series.append(tmp)

    # print(f'{digit} : {series[-1]}')

    return(digit, series)
    


def BenfordRatio(terms):
    numerator = [0] * 9
    denominator = 0
    for n in range(1, int(terms) + 1):
        # Denominator
        current_term = 1/n
        denominator = denominator + current_term
        
        # Numerator
        first_digit = str(getFirstSignificantDigit(current_term))
        numerator[int(first_digit)-1] = numerator[int(first_digit)-1] + current_term

    print('# Ordinary Zeta-Benford Ratio')
    normalised_terms = []
    for x in range(0, len(numerator)):
        normalised_terms.append(numerator[x] / denominator)
        # print(f'{x+1} : {normalised_terms[-1]}')
    
    return normalised_terms


#########
# Plots #
#########

def plotTimeSeries(data):
    '''
    Plot digit series using matplotlib. The data object is a dictionary containing the data to plot. The label is the key of the dictionary.

    Args:
        data: a dictionary containing the data to plot.

    Returns:
        None
    '''

    # Create a figure with subplots
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))

    # Plot the time series data on the subplot
    for plot_num in range(1, 10):
        # Calculate the moving average of the data
        ax1.plot(data[str(plot_num)], label=f'Digit {str(plot_num)}')

    ax1.set_xlabel('Terms', fontsize=14)
    ax1.set_ylabel('Benford Ratio', fontsize=14)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    ax1.legend(fontsize=14)
    plt.show()

def plotEpsilonDeviations(X,Y, Y2):
    # Create a figure with subplots
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))

    # Plot the time series data on the subplot
    ax1.plot(X, Y, label=f'Zipf\'s Law Metric')
    ax1.plot(X, Y2, label=f'Benford\'s Law Metric')

    ax1.set_xlabel('Epsilon', fontsize=14)
    ax1.set_ylabel('Benford Ratio', fontsize=14)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    ax1.legend(fontsize=14)
    plt.show()

    return
    

def plotBenfordGraph(benford_ratio_data_set):

    # Expected values due to Benford's law and Zipfs law
    expected_benford = {
        '1' : np.log10(2),
        '2' : np.log10(1+1/2),
        '3' : np.log10(1+1/3),
        '4' : np.log10(1+1/4),
        '5' : np.log10(1+1/5),
        '6' : np.log10(1+1/6),
        '7' : np.log10(1+1/7),
        '8' : np.log10(1+1/8),
        '9' : np.log10(1+1/9)
    }

    expected_zipf = {
        '1' : 0.3547514430742842,
        '2' : 0.1725053566247906,
        '3' : 0.12195303947017247,
        '4' : 0.07713882190756487,
        '5' : 0.1007579940811139,
        '6' : 0.05191670572170785,
        '7' : 0.046931694039928105,
        '8' : 0.038562420211001064,
        '9' : 0.03548252486949083
    }
    # Get total number of observations
    total_number_datapoints = 0
    for key in benford_ratio_data_set.keys():
        total_number_datapoints = total_number_datapoints + len(benford_ratio_data_set[key])
    
    observed_frequencies_length = total_number_datapoints / len(benford_ratio_data_set.keys())

    # Convert percentages to observations
    expected_zipf_percentage_list = [ expected_zipf[key] for key in expected_zipf.keys()]
    expected_zipf_list = [ x * observed_frequencies_length for x in expected_zipf_percentage_list ]
    expected_benford_percentage_list = [ expected_benford[key] for key in expected_benford.keys()]
    expected_benford_list = [ x * observed_frequencies_length for x in expected_benford_percentage_list ]

    print(f'Expected Zipf list: {expected_zipf_list}')
    

    # Get benford probabilites from Benford Ratio Dataset 
    observed_frequencies_percentage = [ benford_ratio_data_set[key][-1] for key in benford_ratio_data_set.keys() ] 
    observed_frequencies = [ x * observed_frequencies_length for x in observed_frequencies_percentage ]
    print(f'observed_frequencies list: {observed_frequencies}')
    
    # Setup figure
    plt.rcParams.update({'font.size': 13.5})
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    ax0 = plt.subplot(gs[0])

    difference, y_colours, yerror = compute_normalised_residuals(observed_frequencies, expected_benford_list)

    # Plot error bars
    bins = [x for x in range(0,9)]
    ax0.errorbar(bins, observed_frequencies, yerr=yerror, label="Observed Occurrence", color='black', marker='x', fmt='x', capsize=3, elinewidth=1, zorder=1)

    ax0.bar(bins, expected_benford_list, 0.7, color='grey', label="Expected Occurrence", zorder=-1)

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
    if total_number_datapoints >= 10000:
        N_mag = math.floor(np.log10(total_number_datapoints))
        
        # Compute magnitude to report on ylabel
        # N_mean_mag_diff = math.floor(np.log10(yticks[lower_index])) + N_mag_diff
        # print(f'{round(dataset_size / 10**N_mag, 1) } * 10** {N_mag}')
        dataset_size_report = r'{} \times 10^{}'.format(round(total_number_datapoints / 10**N_mag, 1),N_mag)

    else:
        dataset_size_report = str(total_number_datapoints)

    # Format axis labels
    plt.xlabel("Digit Value")
    if report_mag_diff != "":
        plt.ylabel(r"Digit Occurrence (${}$)".format(report_mag_diff), fontsize=15)
    else:
        plt.ylabel(r"Digit Occurrence", fontsize=15)
    plt.xticks(bins, "")

    patch = []
    handles, labels = ax0.get_legend_handles_labels()
    patch.append(mpatches.Patch(color='green', label=r'$|\sigma|$ < 1'))
    patch.append(mpatches.Patch(color='firebrick', label=r'$|\sigma|$ > 1,    $N={}$'.format(observed_frequencies_length)))
        
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

    bins_ticks = []
    for x in bins:
        bins_ticks.append(x + 1)

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

    
    plt.xlabel("First Digit Value", fontsize=15)
    plt.ylabel("Normalised Residual", fontsize=15)
    plt.ylim(-y_range - 0.75, y_range + 0.75)

    #format graph in general
    plt.axhline(linewidth=0.5, color='black')
    plt.axhline(y=1, linewidth=0.75, color='black', linestyle='--')
    plt.axhline(y=-1, linewidth=0.75, color='black', linestyle='--')
    #plt.legend(handles=legend_elements, loc='best')
    fig.align_ylabels()

    plt.show()

    return

def main():
    parser = argparse.ArgumentParser(description="Investigate Benford ratios -- a continuation of the Zeta function and Zipf's law using first digit analysis.")
    parser.add_argument("-t", "--terms", help="Number of terms to create.", default='30')
    parser.add_argument("-o", "--output", help="Filepath to output data.", default='zeta_output.txt')
    parser.add_argument("-e", "--epsilon", help="Deviations to introduce", default='0')
    
    args = parser.parse_args()
    number_of_terms = args.terms
    epsilon = float(args.epsilon)
    output_file = args.output

    original_benford_ratios = BenfordRatio(number_of_terms)
    checkNormalisation(original_benford_ratios)
    
    # Ordinary Benford Ratio with no deviations
    benford_ratio_data_set = {}
    print(f'# Zeta-Benford Ratio w/ Epsilon = 0')
    for digit in range(1, 10):
        digit_analysed, time_series = BenfordRatioGivenDigit(number_of_terms, str(digit), 0)
        benford_ratio_data_set[digit_analysed] = time_series

    plotBenfordGraph(benford_ratio_data_set)
    
    # # Benford ratio with Epsilon Deviations
    # benford_ratio_data_set_epsilon = {}
    # print(f'# Zeta-Benford Ratio w/ Epsilon = {epsilon}')
    # for digit in range(1, 10):
    #     digit_analysed, time_series = BenfordRatioGivenDigit(number_of_terms, str(digit), epsilon)
    #     benford_ratio_data_set_epsilon[digit_analysed] = time_series

    # # Check normalisation
    # chi2 = ChiSquared(benford_ratio_data_set_epsilon)
    # print(f'Chi2 = {chi2}')
    # plotTimeSeries(benford_ratio_data_set_epsilon)

    # Variability of chi-squared as a function of Epsilon
    # chi2_list = []
    # chi2_Benford_metric_list = []
    # chi_squared_E_values = []
    # for E in np.arange(-1.5, 0.5 + 0.001, 0.001):
    #     benford_ratio_data_set_epsilon = {}
    #     for digit in range(1, 10):
    #         digit_analysed, time_series = BenfordRatioGivenDigit(number_of_terms, str(digit), E)
    #         benford_ratio_data_set_epsilon[digit_analysed] = time_series
    #     chi2 = ChiSquared(benford_ratio_data_set_epsilon)
    #     chi2_benford = ChiSquaredBenford(benford_ratio_data_set_epsilon)
    #     chi_squared_E_values.append(E)
    #     chi2_list.append(chi2)
    #     chi2_Benford_metric_list.append(chi2_benford)

    # plotEpsilonDeviations(chi_squared_E_values, chi2_list, chi2_Benford_metric_list)

    # print(f'Minima Zipfs Law: {min(chi2_list)} at {chi2_list.index(min(chi2_list))}')

    # min_index, min_value = min(enumerate(chi2_Benford_metric_list), key=lambda x: x[1])
    # print("Minima Benfords Law:", min_value)
    # print("Index of minimum value:", min_index)



    return

main()