#!/usr/bin/python3
import sys
import re
import requests
from subprocess import call
from subprocess import Popen, PIPE
import random
import string
from itertools import groupby
from operator import itemgetter

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

def export_test_stats(export_filename, sec_report_extracted_data):
    # Write to file
    with open(export_filename, 'w') as f:
        for item in sec_report_extracted_data:
            item = item[0]
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

def extract_values_between_tags(foo):
    # indexes of the < and > tags in foo respectively
    start_tag_location = []
    end_tag_location = []
    values_extracted = []
    values_extracted_numeric = []

    # determine the locaion of tags in the string
    for x in range(0, len(foo)):
        if foo[x] == '<':
            start_tag_location.append(x)
            continue

        if foo[x] == '>':
            end_tag_location.append(x)
            continue

    # if we find now tags on the line see if it contains a numerical value anyway
    foo_striped = foo.strip()
    
    if len(start_tag_location) == 0 and len(end_tag_location) == 0:
        for i in ['(',')', ',', '%', '$', ' ', '&#9;','&nbsp;']:
            foo_striped = foo_striped.replace(i, '')

        try:
            float(foo_striped)
            values_extracted_numeric.append(foo_striped)
        except:
            pass

        return(values_extracted_numeric)

    # return if either of the tag location variables is empty. 
    if len(start_tag_location) == 0:
        return([])
    elif len(end_tag_location) == 0:
        return([])
    
    # extract data entries between tags
    if len(start_tag_location) == len(end_tag_location):
        for y in range(1, len(start_tag_location)):
            values_extracted.append(foo[end_tag_location[y-1] + 1:start_tag_location[y]])
    
    # extract data points at the end of lines
    if end_tag_location[-1] != len(foo) - 1:
        if foo[end_tag_location[-1] + 1:].strip() != '':
            values_extracted.append(foo[end_tag_location[-1] + 1:])
    
    
    # determine if the values are numerical or not
    # remove special characters
    for x in range(0, len(values_extracted)):
        for i in ['(',')', ',', '%', '$', ' ', '&#9;','&nbsp;']:
            values_extracted[x] = values_extracted[x].replace(i, '')

    # remove null values from extracted data from foo
    values_extracted = [x for x in values_extracted if x]
    
    # Numeric or not
    for y in values_extracted:
        try:
            float(y)
            values_extracted_numeric.append(y)
        except:
            pass
    
    return(values_extracted_numeric)

def data_cleanup(foo):
    print(len(foo))
    foo_clean_ascend = []
    foo_clean_descend = []
    # removing asscending values from foo
    x = 0
    # print(foo[-10:])
    while x < len(foo) - 2:
        if float(foo[x]) + 1 == float(foo[x + 1]) and float(foo[x]) + 2 == float(foo[x + 2]):
            
            try:
                i = 3
                while float(foo[x]) + i == float(foo[x + i]):
                    # print(foo[x + i-1], foo[x + i])
                    if x + i != len(foo) - 1:
                        i += 1
                    else:
                        break
                    
                x = x + i + 1
                continue

            except:
                break

        else:
            foo_clean_ascend.append(foo[x])
            x += 1
    
    foo = foo_clean_ascend

    # removing decsending values from foo
    x = 0
    while x < len(foo) - 2:
        if float(foo[x]) - 1 == float(foo[x + 1]) and float(foo[x]) - 2 == float(foo[x + 2]):
            try:
                i = 3
                while float(foo[x]) - i == float(foo[x + i]):
                    if x + i != len(foo) - 1:
                        i += 1
                    else:
                        break

                x = x + i + 1
                continue

            except:
                break

        else:
            foo_clean_descend.append(foo[x])
            x += 1

    foo = foo_clean_descend
    foo_clean_descend = []

    # removing duplicate values from foo
    x = 0
    while x < len(foo) - 2:
        if float(foo[x]) == float(foo[x + 1]) and float(foo[x]) == float(foo[x + 2]):
            
            try:
                i = 3
                while float(foo[x]) == float(foo[x + i]):
                    if x + i != len(foo) - 1:
                        i += 1
                    else:
                        break

                x = x + i + 1
                continue

            except:
                break

        else:
            foo_clean_descend.append(foo[x])
            x += 1

    print(len(foo_clean_descend))
    return(foo_clean_descend)

if __name__ == '__main__':
    # variable definition
    # extracted_data_directory = '/home/odestorm/Documents/physics_project/weekly_reports/week18/SEC_collection/report_database/'
    extracted_data_directory = '/home/odestorm/Documents/physics_project/weekly_reports/week19/consolidated_financial_data/test/'
    extracted_data = []
    rejected_data = []
    extracted_data_sanitised = []
    first_digit_test_statistics = []
    second_digit_test_statistics = []
    url_list_lengths = []

    # read in data file containing all uri's to download and evaluate 
    uri_to_download = import_report(sys.argv[1])
    
    # proceed for each uri we download the file, extract the data and remove the file (space concerns)

    # download data for each of these random URIs

    for current_uri in uri_to_download:
        # index = url_index_random[index_counter]
        # index_counter += 1
    # for index in url_index_random:
    #     current_uri = uri_to_download[index]
        extracted_data = []
        rejected_data = []
        consolidated_data = False

        # download our file to analyse
        print(f'[Debug] Downloading {current_uri}')
        download_file(current_uri, '/tmp/current_sec_report.txt')
        sec_report_raw = import_report('/tmp/current_sec_report.txt')
        
        for line in sec_report_raw:
            # Check to see if we are extracting consolidated data based on heading symbols. 
            # consolidated_strings = ['CONSOLIDATED', 'Consolidated', 'consolidated']
            consolidated_strings = ['CONSOLIDATED', 'Consolidated']
            for m in consolidated_strings:
                if m in line:
                    # check to see if notes is in line. This Corresponds to non-genuine heading in files
                    if 'Notes' not in line and 'notes' not in line:
                        consolidated_data = True
                        break
            
            # Check to see if we are leaving consolidated data section based on table tags
            table_strings = ['</table>', '</TABLE>']
            for m in table_strings:
                if m in line:
                    consolidated_data = False
                    break
            
            # only extract data if we are in consolidated data. Else move to next line
            if consolidated_data == False:
                continue

            # Find all values between html tags > and <
            values_between_tags = extract_values_between_tags(line)
            
            for x in values_between_tags:
                extracted_data.append(x)

        # data cleanup. Remove any datapoints obviously corresponding to years or page numbers
        extracted_data = data_cleanup(extracted_data)
        # print(current_uri, len(extracted_data))

        # continue if length of extratced data is zero. Add null (0) to test statistic values
        if len(extracted_data) == 0:
            url_list_lengths.append(f'{current_uri} {len(extracted_data)}')
            first_digit_test_statistics.append([0,0,0])
            second_digit_test_statistics.append([0,0,0])
            continue
        else:
            url_list_lengths.append(f'{current_uri} {len(extracted_data)}')

        # write data to file
        letters = string.ascii_lowercase
        # save_file = extracted_data_directory + current_uri.split('/')[-1].split('.')[0] + f'{str(index)}' + '.txt'
        save_file = extracted_data_directory + 'current_report_extracted.txt'
        # save_file = extracted_data_directory + '/' + current_uri.split('/')[-1].split('.')[0] + ''.join(random.choice(letters) for i in range(10)) + '.txt'
        print(f'[Debug] saving extracted data to {save_file}\n\n')
        export_results(save_file, extracted_data)

        

        # compute test statistics for each test
        # first digit mode 
        # test = call(['/home/odestorm/Documents/physics_project/analysis/benford_analysis/bin/digit_test/benford_test_statistics.py', save_file, '1'])
        p = Popen(['/home/odestorm/Documents/physics_project/analysis/benford_analysis/bin/digit_test/benford_test_statistics.py', save_file, '1'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
        rc = p.returncode
        output = output.decode("utf-8").replace('\n', '')
        first_digit_test_statistics.append([output])
        
        # second digit mode
        p = Popen(['/home/odestorm/Documents/physics_project/analysis/benford_analysis/bin/digit_test/benford_test_statistics.py', save_file, '2'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
        rc = p.returncode
        output = output.decode("utf-8").replace('\n', '')
        second_digit_test_statistics.append([output])
        # print(second_digit_test_statistics)
    
    for x in url_list_lengths:
        print(x)
    
    print('First Digit Test Statistics')
    print(first_digit_test_statistics)

    print('Second Digit Test Statistics')
    print(second_digit_test_statistics)

    # export test statistic results to file
    export_test_stats('/home/odestorm/Documents/physics_project/weekly_reports/week19/consolidated_financial_data/consolidated_data_first_digit_test_stats.txt', first_digit_test_statistics)
    export_test_stats('/home/odestorm/Documents/physics_project/weekly_reports/week19/consolidated_financial_data/consolidated_data_second_digit_test_stats.txt', second_digit_test_statistics)
        