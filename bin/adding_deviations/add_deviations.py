#!/usr/bin/python3
import sys
import math
import numpy as np
import random
import time

### --------------------------------------- USAGE --------------------------------------------- ###


def usage():
    print(f'{sys.argv[0]} <mode> <input_file (base set)> <output_file> <parameters for the mode>\n\nModes:')
    print(f'noise - introduce random Gaussian noise into the set. <sigma for deviations (higher = more deviation!)>')
    print(f'pro - pronounce a digit at a given index. <digit value (e.g. 5)> <index (>1)> <rate (percentage)>')
    print(f'round - introduce rounding behaviour. Pronounce lowest digit and reduce highest digit. <index (>1)> <rate (percentage)>')
    return(0)

### --------------------------------------- IMPORT/EXPORT DATA --------------------------------------------- ###



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

    print("[Debug] Input Data recieved from", input_filename)                       

    # Remove all null entries from the input data
    input_data = ' '.join(input_data).split()

    for x in range(0, len(input_data)):
        input_data[x] = float(input_data[x])

    return(input_data)


def output_numbers(output_filename, data):
    print(f'[Debug] Writing results to {output_filename}')
    with open(output_filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)
    return(0)

### --------------------------------------- DEVIATION FUNCTIONS --------------------------------------------- ###


def introduce_noise(set, params):
    print(f'[Debug] Introducing Gaussian noise with sigma = {params[0]}')
    sigma = float(params[0])

    np.random.seed(int(time.time()))
    for x in range(0, len(set)):
        epsilon = np.random.normal(0, sigma)
        set[x] += set[x] * epsilon

    return(set)

def introduce_noise_two(set, params):
    print(f'[Debug] Introducing noise with sigma = {params[0]}')
    sigma = float(params[0])

    np.random.seed(int(time.time()))
    for x in range(0, len(set)):
        set[x] = np.random.normal(set[x], math.floor(np.log10(set[x])) * sigma)
        # set[x] += set[x] * (sigma)

    return(set)

def pronounce(set, params):
    print(f'[Debug] Pronouncing the digit {params[0]} in the index {params[1]} at the {params[2]}% rate.')
    
    digit, index, rate = params[0], int(params[1]), float(params[2]) / 100
    
    # Perform checks on index and rate
    if index <= 0:
        print(f'[Error] The index {index} is less than one! Must be one or greater.')
        exit()
    elif rate < 0 or rate >= 100:
        print(f'[Error] The rate {rate} must be a valid percentage and not zero or one hundred.')
        exit()
    elif int(digit) < 0 or int(digit) >= 10:
        print(f'[Error] The digit {digit} is not a valid single digit.')
        exit()
    elif int(digit) == 0 and index == 1:
        print(f'[Error] The index {index} cannot have a digit of {digit}.')
        exit()

    for x in range(0, len(set)):
        rand_number = random.random()
        if rand_number <= rate:
            set[x] = str(set[x])
            temp_string = set[x][0:index - 1] + digit + set[x][index:]
            # print(temp_string)
            set[x] = float(temp_string)

    return(set)

def rounding(set, params):
    print(f'[Debug] Introducing upward rounding behaviour in the index {params[0]} at a rate of {params[1]}%. That is {params[1]}% of all numbers with 9 in the index {params[0]} are rounded upward.')
    index, rate = int(params[0]), float(params[1]) / 100
    
    # Perform checks on index and rate
    if index <= 0:
        print(f'[Error] The index {index} is less than one! Must be one or greater.')
        exit()
    elif rate < 0 or rate >= 100:
        print(f'[Error] The rate {rate} must be a valid percentage and not zero or one hundred.')
        exit()

    for x in range(0, len(set)):
        # See of the second digit of element is nine
        temp_string = str(set[x])
        if temp_string[index - 1] == '9':
            rand_number = random.random()
            if rand_number <= rate:
                magnitude = 10 ** math.floor(np.log10(set[x]))
                set[x] = float(math.ceil(set[x] / magnitude)) * magnitude
                print(set[x])
    
    return(set)


### --------------------------------------- MAIN --------------------------------------------- ###


def main(mode, base_set, dev_set, parameters):
    # import data from base_set
    data = input_numbers(base_set)
    print(parameters)
    # set the mode of operation
    # if mode == 'noise':
    #     modified_set = introduce_noise(data, parameters)
    if mode == 'noise':
        modified_set = introduce_noise_two(data, parameters)
    elif mode == 'pro':
        modified_set = pronounce(data, parameters)
    elif mode == 'round':
        modified_set = rounding(data, parameters)

    # write out to file
    output_numbers(dev_set, modified_set)

    return(0)



if __name__ == '__main__':
    try:
        param = []
        for x in range(4, len(sys.argv)):
            param.append(sys.argv[x])

        main(sys.argv[1], sys.argv[2], sys.argv[3], param)
    
    except Exception as e:
        print(e)
        usage()
        exit()