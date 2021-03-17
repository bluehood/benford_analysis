import statistics
import sys

#usage python3 mean_stdev.py tmp.txt

def usage():
    print('Calculate the mean and standard deviation of a list of data.')
    print(f'{sys.argv[0]} <datafile>\n')
    return(0)

if __name__ == '__main__':
    usage()
    entries = []
    filename = sys.argv[1]
    f = open(filename, "r")
    for x in f:
        x = x.replace('\n', '')
        try:
            if "." in x:
                entries.append(float(str(x)))
            else:
                entries.append(int(str(x)))
        except:
            # if " " in x or "£" in x or x == '':
            #     continue
            # try:
            #     entries.append(int(str(x).replace(" ", "")))
            # except:
            continue
            

    # print(entries)
    mean = statistics.mean(entries)
    st_dev = statistics.stdev(entries)

    print(f'&{mean}&{st_dev}\n')

