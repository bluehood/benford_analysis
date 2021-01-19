#!/usr/bin/python3
import sys
import math
import numpy as np
import random
import subprocess
from subprocess import call

### --------------------------------------- MAIN --------------------------------------------- ###

def main(mode, base_set, dev_set, parameters):
    # define output filenames to analyse
    filenames = []

    # set the mode of operation
    if mode == 'noise':
        for x in np.arange(float(parameters[0].split(':')[0]),  float(parameters[0].split(':')[1]) + 0.01, 0.01):
            name = str(np.around(x, 2)).replace('.', '-')
            filenames.append([f'{dev_set}_{name}.txt', np.around(x, 2)])
            call(["add_deviations.py", mode, base_set, f'{dev_set}_{name}.txt', str(float(x))], stdout=subprocess.DEVNULL)
    if mode == 'noise_two':
        for x in np.arange(float(parameters[0].split(':')[0]),  float(parameters[0].split(':')[1]) + 0.02, 0.02):
            name = str(np.around(x, 2)).replace('.', '-')
            filenames.append([f'{dev_set}_{name}.txt', np.around(x, 2)])
            call(["add_deviations.py", mode, base_set, f'{dev_set}_{name}.txt', str(float(x))], stdout=subprocess.DEVNULL)
    elif mode == 'pro':
        for x in np.arange(float(parameters[2].split(':')[0]),  float(parameters[2].split(':')[1]), 1):
            call(["add_deviations.py", mode, base_set, f'{dev_set}_{x}.txt', parameters[0],parameters[1], x], stdout=subprocess.DEVNULL)
    elif mode == 'round':
        for x in np.arange(float(parameters[1].split(':')[0]),  float(parameters[1].split(':')[1]), 1):
            call(["add_deviations.py", mode, base_set, f'{dev_set}_{x}.txt', parameters[0], x], stdout=subprocess.DEVNULL)
    
    # compute test statistics
    if mode in ['noise', 'noise_two']:
        for x in filenames:
            call(["benford_test_statistics.py", x[0], '1'])

        print('')

        for x in filenames:
            call(["benford_test_statistics.py", x[0], '2'])

    

    return(0)


if __name__ == '__main__':
    param = []
    for x in range(4, len(sys.argv)):
        param.append(sys.argv[x])

    main(sys.argv[1], sys.argv[2], sys.argv[3], param)



# <mode> <base set> <output file without extention> <lowerbound:upperbound> <other params> <>