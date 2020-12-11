import requests

# Script to download data about consolidated assets from the Fed Reserves wesbite

def download_file(url, save_name):
    r = requests.get(url, allow_redirects=True)
    open(save_name, 'wb').write(r.content)
    return(0)

def main():
    # define the directory to save data
    save_directory = '/home/odestorm/Documents/physics_project/analysis/data/collected/fedral_reserve_banks/'
    # define the quater for the url 
    url_quarter = ['1231','0930', '0630', '0331']

    # iterate and download all data files in the range 2002-2012
    for year in range(2002, 2013):
        for quarter in url_quarter:
            # format filename and url 
            file_to_save = str(year) + quarter
            full_url = "https://www.federalreserve.gov/releases/lbr/" + file_to_save + "/lrg_bnk_lst.txt"
            file_to_save = save_directory + file_to_save + ".txt"
            download_file(full_url, file_to_save)
    return(0)
    


if __name__ == '__main__':
    main()
