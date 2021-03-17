#!/usr/bin/python3
import sys
from subprocess import call


if __name__ == '__main__':
    print(f'Create multiple figures for a given data file.\n')
    print(f'Usage: {sys.argvp[0]} <datafile> <directory_to_save>\n')
    # mode_names = [['f1', 'FR_first_digit'], ['f2', 'FR_second_digit']]
    mode_names = [['1', 'first_digit'], ['2', 'second_digit'], ['12hn', 'first_second_heatmap'], ['23hn', 'second_third_heatmap']]
    

    for x in mode_names:
        filename = sys.argv[2] + '/' + sys.argv[1].split('/')[-1].split('.')[0] + '_' + x[1] + '.png'
        call(["benford.py", sys.argv[1], x[0], filename])
        # print(filename)
    