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
    

def plotBenfordPlot(benford_ratio_data_set):

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
    #increase font size
    plt.rcParams.update({'font.size': 13.5})
    
    # Setup figure
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    ax0 = plt.subplot(gs[0])

    ax0.errorbar(bins, frequency, yerr=yerror, label="Observed Occurrence", color='black', marker='x', fmt='x', capsize=3, elinewidth=1, zorder=1)

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