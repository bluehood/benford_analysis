#!/usr/bin/python3
import sys
import re
import requests
from subprocess import call


def usage():
    print(f'Extract financial data from entire report for AIG\'s 10-K reports filed with the SEC in 2003 and 2004 (pre-downloaded). Error correction to remove unwanted values such as dates.\n')
    print(f'{sys.argv[0]} <Filename containing SEC report URL> <Directory to save extracted data>\n')
    print(f'<Filename containing SEC report URLs> - filename with a list of URL\'s of the 10-K report to download and process. Must be for AIG.')
    print(f'<Directory to save extracted data> - the directory to save the processed data. The exact filename is taken from the URL associated to each downloaded file.\n')
    return(0)

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
    usage()
    # variable definition
    extracted_data_directory = sys.argv[2]
    extracted_data = []
    extracted_data_sanitised = []
    strings_to_search_for = ['><FONT size="1">', '><B><FONT size="1">', '><FONT size="2">', '><B><FONT size="2">']

    # read in data file containing all uri's to download and evaluate 
    uri_to_download = import_report(sys.argv[1])
    
    # proceed for each uri we download the file, extract the data and remove the file (space concerns)

    for current_uri in uri_to_download:
        # download our file to analyse
        download_file(current_uri, '/tmp/current_sec_report.txt')
        sec_report_raw = import_report('/tmp/current_sec_report.txt')
        extracted_data = []
        
        for line in sec_report_raw:
            for search_parameter in strings_to_search_for:
                if search_parameter in line:
                    string_identified = re.search('">(.*)</FONT', line)

                    # exit for loop if no data point is found. This also helps avoid columns that span multiple lines
                    if string_identified == None:
                        break

                    datapoint_extracted = string_identified.group(1)
                    if datapoint_extracted == '':
                        break

                    # process datapoint. Negative if if starts with (
                    datapoint_extracted = datapoint_extracted.replace('(', '-').replace(',','').replace('%', '')
                    # print(int(datapoint_extracted))
                    
                    try:
                        # print(float(datapoint_extracted))
                        float(datapoint_extracted)
                        extracted_data.append(datapoint_extracted)
                    except:
                        # if datapoint_extracted in ['&#150;', '&nbsp;']:
                        #     extracted_data.append('')
                        pass

                    break
        
        # Remove years which are unimportant to the data 
        # Columns
        extracted_data = remove_column_years(extracted_data)
        
        # write data to file
        save_file = extracted_data_directory + '/' + current_uri.split('/')[-1].split('.')[0] + '.txt'
        export_results(save_file, extracted_data)
        extracted_data = []
        