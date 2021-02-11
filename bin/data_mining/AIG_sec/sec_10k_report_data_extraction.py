#!/usr/bin/python3
import sys

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

def extract_rows_contain_years(sec_report_subset):
    # loop through and find first instance of <B><FONT size="1">. This will contain the year. Continue to extract data for this year until we
    # encounter </TR> which ends the row. Add this data to current year and continue
    year_identified = False
    years_avaliable = []
    extracted_data_arranged_year_local = []
    strings_to_search_for = ['nowrap><FONT size="1">', 'nowrap><B><FONT size="1">', 'nowrap><FONT size="2">', 'nowrap><B><FONT size="2">']
    year_strings_to_search_for = ['<FONT size="1">(.*)', '<FONT size="1">(.*)</FONT>']

    for x in range(0, len(sec_report_subset)):
        # We are searching for the year assocaited with the row
        if '<FONT size="1">' in sec_report_subset[x]:
            if year_identified == False:
                # Refine search based on the length of the string
                for year_search_parameter in year_strings_to_search_for:
                    string_identified = re.search(year_search_parameter, sec_report_subset[x])
                    year_extracted = string_identified.group(1)

                    # year extracted succesfully 
                    if len(str(year_extracted)) == 4:
                        break
                    else:
                        continue
                
                try:
                    int(year_extracted)
                    # print(year)
                    if int(year_extracted) > 1992 and int(year_extracted) < 2007:
                        # check if there is an entry with the year already set
                        if year_extracted in years_avaliable:
                            current_index_extracted_data = years_avaliable.index(year_extracted)

                        # otherwise we need to create one
                        else:
                            years_avaliable.append(year_extracted)
                            extracted_data_arranged_year_local.append([year_extracted, []])
                            current_index_extracted_data = -1

                        year_identified = True
                        
                except:
                    pass
            
        if year_identified == True:   
            for search_parameter in strings_to_search_for:
                if search_parameter in sec_report_subset[x]:
                    string_identified = re.search('">(.*)</FONT>', sec_report_subset[x])
            
                    datapoint_extracted = string_identified.group(1)

                    # process datapoint. Negative if if starts with (
                    datapoint_extracted = datapoint_extracted.replace('(', '-').replace(',','').replace('%', '')
                    
                    # add to local datastore
                    try:
                        float(datapoint_extracted)
                        extracted_data_arranged_year_local[current_index_extracted_data][1].append(datapoint_extracted)
                        
                    except:
                        if datapoint_extracted in ['&#150;', '&nbsp;']:
                            extracted_data_arranged_year_local[current_index_extracted_data][1].append('')
                        pass
                        
                    break

        if sec_report_subset[x] == '</TR>':
            year_identified = False

    return(extracted_data_arranged_year_local, years_avaliable)

def extract_columns_contain_years(sec_report_subset):
    # Firstly we need to extract the years avaliable from column values 
    # Loop through the first 50 lines and extract years

    years_avaliable = []

    for x in range(0, 50):
        # Break if we are at the end of the years column section
        
        if sec_report_subset[x] == '</TR>':
            lower_subset = x
            break
        
        # Find string pattern in webpage corresponding to years data. Extract years data. If it corresponds to a valid year (is an integer) add to avaliable years. 
        if '<TD colspan="2" align="' in sec_report_subset[x]:
            string_identified = re.search('size="1">(.*)</FONT', sec_report_subset[x])
            if string_identified == None:
                continue

            year_extracted = string_identified.group(1)
            
            try:
                int(year_extracted)
                if int(year_extracted) > 1992 and int(year_extracted) < 2007:
                    years_avaliable.append(year_extracted)
            except:
                pass

    # Determine whether we could extract years in this case. If not we extract the data indiscrimiatly
    
    if years_avaliable == []:
        extracted_data_no_arrangement = []

        strings_to_search_for = ['nowrap><FONT size="1">', 'nowrap><B><FONT size="1">', 'nowrap><FONT size="2">', 'nowrap><B><FONT size="2">']

        for x in range(lower_subset, len(sec_report_subset)):
        # search for identifying string and extract value
            for search_parameter in strings_to_search_for:
                if search_parameter in sec_report_subset[x]:
                    string_identified = re.search('size="2">(.*)</FONT', sec_report_subset[x])
                    
                    # exit for loop if no data point is found. This also helps avoid columns that span multiple lines
                    if string_identified == None:
                        break
                    
                    datapoint_extracted = string_identified.group(1)

                    # process datapoint. Negative if if starts with (
                    datapoint_extracted = datapoint_extracted.replace('(', '-').replace(',','').replace('%', '')
                    # print(int(datapoint_extracted))
                    
                    try:
                        # print(float(datapoint_extracted))
                        float(datapoint_extracted)
                        extracted_data_no_arrangement.append(datapoint_extracted)
                    except:
                        if datapoint_extracted in ['&#150;', '&nbsp;']:
                            extracted_data_no_arrangement.append('')
                        pass
                    

        return(extracted_data_no_arrangement, years_avaliable)

    
    # Construct local datastore for each year as a list. Note the list descends by year 2003,2002,...,1999
    extracted_data_arranged_year_local = []
    for x in years_avaliable:
        extracted_data_arranged_year_local.append([str(x), []])

    # Loop through the rest of the data sec report subset and extract finacial data
    modular_limit = len(years_avaliable)
    modular_tracker = 0
    empty_value_tracker = 0
    strings_to_search_for = ['><FONT size="1">', '><B><FONT size="1">', '><FONT size="2">', '><B><FONT size="2">']

    for x in range(lower_subset, len(sec_report_subset)):
        # Reset empty value tracker and modular tracker at the begining of each table after then end of each row
        if sec_report_subset[x] in ['</TR>', '    </TD>']:
            empty_value_tracker = 0
            modular_tracker = 0

        # print(empty_value_tracker, sec_report_subset[x])

        # if we are not expecting an empty value search for identifying string and extract value
        # if x < lower_subset + 100:
        #     print(empty_value_tracker, modular_tracker % modular_limit, sec_report_subset[x])
        if empty_value_tracker == 3:
            # print(sec_report_subset[x])
            for search_parameter in strings_to_search_for:
            # search for identifying string and extract value
                if search_parameter in sec_report_subset[x]:
                    string_identified = re.search('">(.*)</FONT>', sec_report_subset[x])
                    # print(sec_report_subset[x])
                    datapoint_extracted = string_identified.group(1)

                    # process datapoint. Negative if if starts with (
                    datapoint_extracted = datapoint_extracted.replace('(', '-').replace(',','')
                    
                    # add to local datastore
                    try:
                        float(datapoint_extracted)
                        extracted_data_arranged_year_local[modular_tracker % modular_limit][1].append(datapoint_extracted)
                        # print(modular_tracker % modular_limit)
                        empty_value_tracker = -1
                        
                    except:
                        
                        if datapoint_extracted in ['&#150;', '&nbsp;']:
                            extracted_data_arranged_year_local[modular_tracker % modular_limit][1].append('')
                        empty_value_tracker = -1

                        pass
                     
                    break

            # if '<TD><FONT size="1">&nbsp;</FONT></TD>' not in sec_report_subset[x]:
            modular_tracker += 1

        empty_value_tracker = (empty_value_tracker + 1) % 4

    return(extracted_data_arranged_year_local, years_avaliable)

def export_results(export_filename, sec_report_extracted_data):
    # export data as comma delimited csv file
    # column labels and remove null entries from the data
    column_labels_string = ""
    export_lines = []
    for x in sec_report_extracted_data:
        column_labels_string += f'{x[0]},'
        x[1] = [y for y in x[1] if y]

    # remove leading comma from column labels
    column_labels_string = column_labels_string[0:-1]
    export_lines.append(column_labels_string)

    # print(sec_report_extracted_data[7][0])
    # for x in sec_report_extracted_data[7][1]:
    #     print(x)

    # determine the length of each data set 
    lengths = []
    for x in sec_report_extracted_data:
        lengths.append(len(x[1]))
    
    # loop through and begin writing lines
    current_line = ""
    for x in range(0, max(lengths)):
        for y in sec_report_extracted_data:
            if x < len(y[1]):
                current_line += f'{y[1][x]},'
            else:
                current_line += ','

        current_line = current_line[0:-1]
        export_lines.append(current_line)
        current_line = ""

    
    # Write to file
    with open(export_filename, 'w') as f:
        for item in export_lines:
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
    
    # import sec report from disk in first commandline arguement argv[1]
    sec_report_raw = import_report(sys.argv[1])

    # Define array used to extract data 
    extracted_data_arranged_year = []
    for x in range(1993, 2004 + 1):
        extracted_data_arranged_year.append([str(x), []])
    
    extracted_data_arranged_year.append(['Misc', []])

    # Define line segements to extract for each year
    # Extract relevant data sections from report defined by line numbers.

    # TEST CASES
    # sec_report_2013_sections = [['columns', 10905, 11971]] 
    

    # General development of the business of AIG on a consolidated basis 526 - 1425 
    # Identifiable assets, revenues and income derived from operations in the United States and Canada and from operations in other countries 1575 - 2289
    # Analysis of Consolidated Net Losses and Loss Expense Reserve Development 4706 - 6171
    # Analysis of Consolidated Losses and Loss Expense Reserve Development Excluding Asbestos and Environmental Losses and Loss Expense Reserve Development 3125 - 4585
    # Reconciliation of Net Reserves for Losses and Loss Expenses 6234 - 6492
    # Insurance Investment Operation 8454 - 9007
    # SELECTED CONSOLIDATED FINANCIAL DATA 10905 - 11971

    sec_report_2013_sections = [['columns', 526, 1425], ['columns', 1575, 2289], ['columns', 3125, 4585], ['columns', 4706, 6171], ['columns', 6234, 6492], ['rows', 8454, 9007], ['columns', 10905, 11971]]

    for section in sec_report_2013_sections:
        if section[0] == 'columns':
            local_data_extracted, local_years = extract_columns_contain_years(sec_report_raw[section[1]:section[2]])
            extracted_data_arranged_year = add_to_global_dataset(extracted_data_arranged_year, local_data_extracted, local_years)

        elif section[0] == 'rows':
            local_data_extracted, local_years = extract_rows_contain_years(sec_report_raw[section[1]:section[2]])
            extracted_data_arranged_year = add_to_global_dataset(extracted_data_arranged_year, local_data_extracted, local_years)

    export_results(sys.argv[2], extracted_data_arranged_year)
    # print(extracted_data_arranged_year[7][0])
    # for x in extracted_data_arranged_year[7][1]:
    #     print(x)

    # print(extracted_data_arranged_year)

    
    