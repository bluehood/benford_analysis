#!/usr/bin/python3
import sys

def usage():
    print(f'Adjust for inflation in financial data. The rate of inflation will need to be specified as a parameter to the program.')
    print(f'{sys.argv[0]} <datafile> <rate of inflation (percentage)> <savefile>')
    return(0)

def import_data(input_filename): 
    # Input data from argv[1] into input_data (newline delimited)
    try:
        open_file = open(input_filename, "r")
        raw_data = open_file.readlines() 

        input_data = []
        for x in raw_data:
            input_data.append(x.replace('\n', ''))
        
        open_file.close()  

    except:
        print("[Fatal Error] Failed to read data from report file. Exiting.")
        # usage()
        exit()

    input_data_float = [float(x) for x in input_data]
    print(type(input_data_float[0]))
    
    return(input_data_float)

def export_results(export_filename, export_data):
    # Write to file
    with open(export_filename, 'w') as f:
        for item in export_data:
            f.write("%s\n" % item)
    
    return(0)

if __name__ == '__main__':
    # import dataset 
    raw_data = import_data(sys.argv[1])
    processed_data = []

    # increase by inflation amount given in sys.argv[2] as a percentage
    inflation_amount = float(sys.argv[2])
    for x in raw_data:
        processed_data.append(str(round(x * (1 + (inflation_amount / 100)), 3)))

    # write out to file
    export_results(sys.argv[3], processed_data)
    
    print(processed_data)
