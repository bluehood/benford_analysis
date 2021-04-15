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
from subprocess import Popen, PIPE
from tqdm import tqdm
from time import sleep

def calculate_percentile(list_raw, percentile_value):
    # print(sorted(list_raw))
    list_raw = [float(x) for x in list_raw]
    p = np.percentile(list_raw, percentile_value)
    return(p)

def export_results(export_filename, export_data):
    # Write to file
    with open(export_filename, 'w') as f:
        for item in export_data:
            f.write("%s\n" % item)
    
    return(0)


def main(digit_test):
    d_star_stats = []
    a_sqaured_stats = []

    # process 1000 benford sets with 10000 elements in the range 10^4 to 10^9
    for i in tqdm(range(1000)):
        # generate a benford set using generate_benford
        Popen(['generate_benford', '/tmp/benford_set.txt', '10000', '4', '9'],  stdin=PIPE, stdout=PIPE, stderr=PIPE)

        # compute A^2 and d^*
        p = Popen(['/home/odestorm/Documents/physics_project/analysis/benford_analysis/bin/digit_test/benford_test_statistics.py', '/tmp/benford_set.txt', str(digit_test)], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
        rc = p.returncode
        output = output.decode("utf-8").replace('\n', '')

        # store test statistic values
        d_star_stats.append(output.split(',')[2])
        a_sqaured_stats.append(output.split(',')[1])

        # ensures time seed works correctly for generate_benford.py
        sleep(1)

    # print(d_star_stats)
    # print(a_sqaured_stats)
    print('d_star 95th percentile:', calculate_percentile(d_star_stats, 95))
    print('d_star 99th percentile:', calculate_percentile(d_star_stats, 99))
    print()
    print('a_squared 95th percentile:', calculate_percentile(a_sqaured_stats, 95))
    print('a_squared 99th percentile:', calculate_percentile(a_sqaured_stats, 99))

    # save d_star and a_squared to file
    export_results('/home/odestorm/Documents/physics_project/weekly_reports/project_report/p_values/d_star_second_digit.txt', d_star_stats)
    export_results('/home/odestorm/Documents/physics_project/weekly_reports/project_report/p_values/a_squared_second_digit.txt', a_sqaured_stats)
    
    return(0)


if __name__ == '__main__':
    main(int(sys.argv[1]))
    