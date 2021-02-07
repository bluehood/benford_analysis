#!/usr/bin/python3
import sys
from subprocess import call
import time

if __name__ == '__main__':
    for x in range(0, 1000):
        # create benford set using bruteforce techniques. Save to /home/odestorm/Documents/physics_project/weekly_reports/week14/brute_force_data_sets/
        call(["generate_benford", f'/home/odestorm/Documents/physics_project/weekly_reports/week14/brute_force_data_sets/set_{str(x)}.txt', str(50000), str(4), str(8)])
        time.sleep(1)