#!/usr/bin/python3
import sys
import numpy as np

def main(savefile, size, lowerbound, upperbound):
    benford_set = []
    # Check that the difference is an integer value
    if np.log10(upperbound / lowerbound) % 1 != 0.0:
        print(f'[Fatal] log10({upperbound} / {lowerbound}) is not an integer. Exiting.')

    # Calculate the difference
    print(f'[Debug] Creating a Benford set of size {size} in the range [{lowerbound}, {upperbound}) and saving to {savefile}')
    d = np.log10(upperbound) - np.log10(lowerbound)
    
    # Calculate the common ratio
    r = 10 ** (d / size)

    print(f'[Debug] d = {d}, r = {r}')

    # Make benford set
    for x in range(0, size + 1):
        benford_set.append(np.floor(lowerbound * r ** x))

    # Write to file
    print(f'[Debug] Writing to file')
    with open(savefile, 'w') as f:
        for item in benford_set:
            f.write("%s\n" % item)

    return(0)


if __name__ == '__main__':
    try:
        main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    except:
         print(f"Usage: {sys.argv[0]} <savefile> <set_size> <lowerbound> <upperbound>")
         exit()
