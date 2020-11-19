import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab as pl


def main():
    #setup variables 
    file_to_add = '620_920_dstar.txt'
    # files_to_add = ['120_420_dstar.txt', '220_820_dstar.txt', '620_920_dstar.txt']
    pwd = '/home/odestorm/Documents/physics_project/analysis/data/synthetic/'
    #import d* values and add to figure
    
    # Import and sanitise data
    entries = []
    f = open(pwd + file_to_add, "r")
        
    for x in f:
        x = x.replace('\n', '')
        try:
            if "." in x:
                entries.append(float(str(x)))
            else:
                entries.append(int(str(x)))
        except:
            if " " in x or "£" in x or x == '':
                continue
            try:
                entries.append(int(str(x).replace(" ", "")))
            except:
                continue
            
    # Sort data and plot using pylab
    fit = stats.norm.pdf(entries, np.mean(entries), np.std(entries))
    print(fit)
    pl.plot(entries, fit,'-o', linewidth=2, markersize=5)
    pl.xlabel(r'$d^*$', size=12)
    pl.ylabel("Normalised Probability", size=12)
    pl.hist(entries, density=True, color = "skyblue", ec="black", lw=1)
    pl.show() 

if __name__ == '__main__':
    main()