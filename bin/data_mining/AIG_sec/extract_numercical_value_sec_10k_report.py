#!/usr/bin/python3
import sys
import re
import requests
from subprocess import call
import random
import string

def import_report(input_filename): 
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

def remove_column_years(datafile):
    entry_index = 0
    datafile_sanitised = []

    while entry_index < len(datafile):
        # print(entry_index)
        if float(datafile[entry_index]) < 2005 and float(datafile[entry_index]) > 1990:
            
            if float(datafile[entry_index + 1]) == float(datafile[entry_index]) - 1:
                first_year_index = entry_index
                local_index = first_year_index + 2
                local_index_two = 2
                reading_years = True

                while reading_years == True:
                    if float(datafile[local_index]) != float(datafile[first_year_index]) - local_index_two:
                        final_year_index = local_index
                        reading_years = False
                        break

                    local_index += 1
                    local_index_two += 1

                # print(datafile[first_year_index:final_year_index])
                entry_index = final_year_index
                continue

            elif float(datafile[entry_index + 1]) == float(datafile[entry_index]) + 1:
                first_year_index = entry_index
                local_index = first_year_index + 2
                local_index_two = 2
                reading_years = True

                while reading_years == True:
                    if float(datafile[local_index]) != float(datafile[first_year_index]) + local_index_two:
                        reading_years = False
                        break

                    local_index += 1
                    local_index_two += 1

                # print(datafile[first_year_index:final_year_index])
                entry_index = local_index
                continue

        datafile_sanitised.append(datafile[entry_index])
        entry_index += 1
    
    return(datafile_sanitised)

def download_file(url, save_name):
    f = open(save_name, 'w')
    call(['curl', url], stdout=f)
    # r = requests.get(url, allow_redirects=True)
    
    return(0)

def export_results(export_filename, sec_report_extracted_data):
    # Write to file
    with open(export_filename, 'w') as f:
        for item in sec_report_extracted_data:
            f.write("%s\n" % item)

    return(0)

def add_to_global_dataset(global_set, local_insert, local_years):
    # add data that has no associated year
    if local_years == []:
        for x in local_insert:
            global_set[-1][1].append(x)

    else:
        global_set_years = []
        for x in global_set:
            global_set_years.append(x[0])
        
        # determine the index in global set of each element of local years
        local_years_index = []
        for x in local_years:
            local_years_index.append(global_set_years.index(x))

        # combine local_insert into global set
        for x in range(0, len(local_insert)):
            global_set[local_years_index[x]][1] += local_insert[x][1]

    return(global_set)

if __name__ == '__main__':
    # variable definition
    extracted_data_directory = sys.argv[2]
    extracted_data = []
    rejected_data = []
    extracted_data_sanitised = []
    

    # read in data file containing all uri's to download and evaluate 
    uri_to_download = import_report(sys.argv[1])
    
    # proceed for each uri we download the file, extract the data and remove the file (space concerns)

    for current_uri in uri_to_download:
        extracted_data = []
        rejected_data = []
        # download our file to analyse
        download_file(current_uri, '/tmp/current_sec_report.txt')
        sec_report_raw = import_report('/tmp/current_sec_report.txt')
        
        for line in sec_report_raw:
            # Find all values between html tags
            strings_to_search_for = [r'"2">(.*)', r'"2">(.*)</', r'"1">(.*)', r'"1">(.*)</', r'"3">(.*)', r'"3">(.*)</']
            
            for i in strings_to_search_for:
                string_identified = re.findall(i, line)
                if string_identified != []:
                    break
                
            
            # print(string_identified)
            # continue
            # if there is a value determine if it is numerical or not. 
            
            if string_identified != []:
                for datapoint_entry in string_identified:
                    # remove html elements from the result
                    datapoint_extracted = datapoint_entry.split('<')[0].split('>')[0]
                    # convert to integer representation by removing special characters
                    datapoint_extracted = datapoint_extracted.replace('(', '-')
                    for i in [')', ',', '%', '$', ' ', '&#9;']:
                        datapoint_extracted = datapoint_extracted.replace(i, '')

                    # determine if the datapoint is numerical or not
                    # print(datapoint_extracted)
                    try:
                        float(datapoint_extracted)
                        extracted_data.append(datapoint_extracted)
                    except:
                        rejected_data.append(datapoint_extracted)
                    
            else:
                continue

        # write data to file
        letters = string.ascii_lowercase
        save_file = extracted_data_directory + '/' + current_uri.split('/')[-1].split('.')[0] + ''.join(random.choice(letters) for i in range(10)) + '.txt'
        print(f'[Debug] saving extracted data to {save_file}')
        export_results(save_file, extracted_data)

        # for x in range(0, 200):
        #     print(extracted_data[x])
        
        # print(len(extracted_data))
        