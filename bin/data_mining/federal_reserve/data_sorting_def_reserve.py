#!/usr/bin/python3

def usage():
    print(f'Process federal reserve public releases downloaded from https://www.federalreserve.gov/releases/lbr/ for consolidated and domestic assets of the largest commercial banks. The data has aready been downloaded and is stored locally before processing using fed_reserve.py.')
    return(0)

# read and sanitise dataset
def read_san_data_from_file(input_file):
    # read in all data from file
    open_file = open(input_file, "r")
    input_data = open_file.read()

    # split by newline character
    input_data = input_data.splitlines() 
    open_file.close()

    # remove lines at the begining that are useless. finding the ----- technique
    i = 0

    # find start of interesting data
    for x in range(0, 20):
        if "-------------------" in input_data[x]:
            i += 1
            if i == 2:
                input_data = input_data[x + 1:]
                break
     
    # remove every second line. This will include one of the two null lines in the large gaps between chunks of data. 
    # extract consolidated and domestic assets removing trailing and leading whitspace 
    san_input_data = []
    # for x in range(0, 100):
    #     print(input_data[x])

    for x in range(0, len(input_data)):
        # stop once we reach the end of the file
        if 'Summary' in input_data[x]:
            break
        
        #base case for x = 0
        if x == 0:
            # remove multiple whitespaces after we have extracted the correct columns
            entry = ""
            entry = " ".join(input_data[0][77:97].strip().split())
            san_input_data.append(entry)
        
        # remaining cases
        if x % 2 == 0:
            # remove multiple whitespaces after we have extracted the correct columns
            entry = ""
            entry = " ".join(input_data[x][77:97].strip().split())
            san_input_data.append(entry)

    # remove remaining empty line from chunks of data
    input_data = [j for i, j in enumerate(san_input_data) if i%6 !=0]
        
    # split consol and domestic assests
    # case 1: whitespace between the two entries. This is happy days
    # case 2: no whitespace between the two entries. Must look at commas/ length of string to figure out where one string ends and another begins. 

    # lists for consolidated data and domestic data
    consol_list = []
    domestic_list = []

    # print(len(input_data))
    for x in range(0, len(input_data)):
        # case 1 
        if ' ' in input_data[x]:
            # removing ','s from figures
            consol_list.append(input_data[x].split(' ')[0].replace(',', ''))
            domestic_list.append(input_data[x].split(' ')[1].replace(',', ''))
        
        # case 2
        else:
            # remove remainin null lines at the end of the file.
            # if input_data[x].split(' ')[0] != "":
            # removing ','s from figures
            consol_list.append(input_data[x][0:9].replace(',', ''))
            domestic_list.append(input_data[x][9:].replace(',', ''))

    return(consol_list, domestic_list)

# write sanitised data to disk
def output_to_file(dataset, location):
    with open(location, 'w') as f:
        for item in dataset:
            f.write("%s\n" % item)
            
    return(0)


def main():
    # define directory where the files are
    file_directory = '/home/odestorm/Documents/physics_project/analysis/data/collected/fedral_reserve_banks/'
    directory_output = "/home/odestorm/Documents/physics_project/data/federal_reserve/"
    
    url_quarter = ['1231','0930', '0630', '0331']
    for year in range(2002, 2013):
        for quarter in url_quarter:
            # read in and sanitise all datasets
            filename_full = file_directory + str(year) + quarter + ".txt"
            consolidated, domestic = read_san_data_from_file(filename_full)
            output_to_file(consolidated, directory_output + "consolidated/" + str(year) + quarter + ".txt")
            # write results to disk
            output_to_file(domestic, directory_output + "domestic/" + str(year) + quarter + ".txt")

    #/home/odestorm/Documents/physics_project/data/federal_reserve/         save location
    return(0)

if __name__ == '__main__':
    usage()
    main()