#!/usr/bin/python3
import sys

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

    # print(input_data)
    return(input_data)

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
    years_to_extract = []
    # Define the years to extract 
    # years_to_extract = ['2000', '2001', '2002', '2003']

    
    for x in range(1993, 2004 + 1):
        years_to_extract.append(str(x))
    years_to_extract.append('Misc')

    # Determine the index at which each year we are interested in is located 
    years_index = []

    for x in range(0, len(raw_data[0].split(','))):
        for y in years_to_extract:
            if raw_data[0].split(',')[x] == y:
                years_index.append(x)

    # Extract data from raw_data and remove any null entries
    for x in raw_data[1:]:
        for y in range(0, len(x.split(','))):
            if y in years_index:
                processed_data.append(x.split(',')[y])

    processed_data = [y for y in processed_data if y]
    
    export_results(sys.argv[2], processed_data)