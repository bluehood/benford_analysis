import statistics
import sys

#usage python3 mean_stdev.py tmp.txt


if __name__ == '__main__':
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
            if " " in x or "£" in x or x == '':
                continue
            try:
                entries.append(int(str(x).replace(" ", "")))
            except:
                continue
            

    #print(entries)
    mean = statistics.mean(entries)
    st_dev = statistics.stdev(entries)

    if int(st_dev) == 0:
        st_dev = "-"
        print(st_dev + "&")
        exit()

    #print("Mean=" + str(mean), ":", "s.d.=" + str(st_dev))
    print(str('{:.3f}'.format(st_dev)) + "&")