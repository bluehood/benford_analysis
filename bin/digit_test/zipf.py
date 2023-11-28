import argparse
import sys
import string
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def finiteGeneralisedZipfDistribution(zipf_frequencies_dict, s):
    wordcount = len(zipf_frequencies_dict.keys())
    harmonic_number_N = 0
    for n in range(1, wordcount + 1):
        harmonic_number_N = harmonic_number_N + (1/(n**s))

    zipf_distribution = {}
    for n in range(1, wordcount + 1):
        zipf_distribution[str(n)] = (1/harmonic_number_N) * (1/(n**s))

    return zipf_distribution

def finiteZipfDistribution(zipf_frequencies_dict):
    wordcount = len(zipf_frequencies_dict.keys())
    harmonic_number_N = 0
    for n in range(1, wordcount + 1):
        harmonic_number_N = harmonic_number_N + (1/n)

    zipf_distribution = {}
    for n in range(1, wordcount + 1):
        zipf_distribution[str(n)] = (1/harmonic_number_N) * (1/n)

    return zipf_distribution

def processTextFile(text_list):
    zipf_frequencies_dict = {}
    for word in text_list:
        if word in zipf_frequencies_dict.keys():
            zipf_frequencies_dict[word] = zipf_frequencies_dict[word] + 1
        else:
            zipf_frequencies_dict[word] = 1

    # Order distribution from highest to lowest
    zipf_frequencies_dict = dict(sorted(zipf_frequencies_dict.items(), key=lambda item: item[1], reverse=True))
    return zipf_frequencies_dict

def calculateStat(Y1, Y2):
    chi_squared = 0
    for x in range(0, len(Y1)):
        tmp = (Y2[x] - Y1[x])**2 / Y1[x]
        chi_squared = chi_squared + tmp

    return chi_squared

def loglogPlot(X, Y1, Y2):
    # Take log of data
    X_log = np.log10(X)
    Y1_log = np.log10(Y1)
    Y2_log = np.log10(Y2)
    plt.rcParams.update({'font.size': 13.5})
    
    # Graph
    plt.figure(figsize=(8, 6))
    plt.loglog(X, Y1, color='b', marker='x', label='Ideal Zipf Distribution')
    plt.loglog(X, Y2, color='black', marker='x', label='Observed Distribution')

    # Line of best fit 
    # Perform a linear fit (line of best fit)
    slope, intercept = np.polyfit(X_log, Y1_log, 1)
    # mx = [ x * slope for x in X ]
    # predicted_y = mx + intercept
    # plt.plot(X, predicted_y, label='Best Fit', color='red')
    
    # Add value of s to histogram 
    plt.axhline(y=0, color='white', linestyle='--', label=r'$S=$'+ str(round(-1*slope,3)))

    plt.legend()

    plt.ylabel('Zipfs Law (log scale)')
    plt.xlabel('k (log scale)')
    plt.show()
    return 

def main(input_file):
    # Read text into a file 
    with open(input_file, 'r') as file:
        # Read the entire contents of the file into a variable
        text_to_analyse = file.read()   
    
    # Remove punctuation. Define a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text_to_analyse = text_to_analyse.translate(translator)
    text_to_analyse = text_to_analyse.lower()

    # Remove \n chars and replace with spaces
    text_to_analyse = text_to_analyse.replace('\n', ' ')

    # Split by space character and remove empty values from the list
    text_to_analyse_list = text_to_analyse.split(' ')
    text_to_analyse_list = [ item for item in text_to_analyse_list if item is not None and item != '' ]

    # Calculate Zipf frequencies
    zipf_frequencies_dict = processTextFile(text_to_analyse_list)
    # for key in zipf_frequencies_dict.keys():
    #     print(f'{key} : {zipf_frequencies_dict[key]}')
    
    # Fit data to the generalised Zipf law 
    interval = 0.1
    lower = 0.1
    middle = 1.0
    upper = 10

    while interval != 0.0001:
        statistic_values_dict = {}
        print(f'Testing intervals at the {interval} level.')
        for s in np.arange(lower, upper, interval):
            # Calculate the Zipf distribution 
            zipf_distribution = finiteGeneralisedZipfDistribution(zipf_frequencies_dict, s) 
            X, Y1, Y2 = [], [], []
            index = 1
            for key in zipf_frequencies_dict.keys():
                X.append(index)
                Y1.append(zipf_distribution[str(index)])
                Y2.append(zipf_frequencies_dict[key] / len(zipf_frequencies_dict.keys()))
                index += 1

            statistic_values_dict[str(s)] = calculateStat(Y1,Y2)

        # Order statistics from lowest to highest. Print the lowest statistic value 
        statistic_values_dict = dict(sorted(statistic_values_dict.items(), key=lambda item: item[1], reverse=False))
        statistic_values_dict_keys_list = list(statistic_values_dict.keys())
        middle = float(statistic_values_dict_keys_list[0])
        lower = middle - interval
        upper = middle + interval
        interval = interval / 10
        s_refined = statistic_values_dict_keys_list[0]
        print(f'{statistic_values_dict_keys_list[0]}: {statistic_values_dict[statistic_values_dict_keys_list[0]]}')
    

    # Plot data for the best statistic value
    zipf_distribution = finiteGeneralisedZipfDistribution(zipf_frequencies_dict, float(s_refined))
    X, Y1, Y2 = [], [], []
    index = 1
    for key in zipf_frequencies_dict.keys():
        X.append(index)
        Y1.append(zipf_distribution[str(index)])
        Y2.append(zipf_frequencies_dict[key] / len(zipf_frequencies_dict.keys()))
        index += 1

    loglogPlot(X,Y1,Y2)

    return


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Apply Zipf's law to text. Both the infinite word case or finite words are considered.")
        
        parser.add_argument("-i", "--infile", help="Input file containing text to analyse.")

        args = parser.parse_args()
        inputfile = args.infile

    except Exception as e:
        parser.print_help()
        exit()
    else:
        main(sys.argv[2])
    