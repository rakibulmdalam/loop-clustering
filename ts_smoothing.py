import numpy as np
from timedataframe import TimeDataFrame
from pointwiseanalysis import PointwiseAnalysis


def main(filename):
    '''
    Here, we are reading from the raw data (MA33_2011_1_15.csv) and creating tranformed time series with 96*7 = 672
    values each. Each value corresponds to the median of the values for this particular time-of-the-week, e.g. 9:00 on Mondays.
    The transformed time series are stored in a file called "medians_all.txt", one line per time series (i.e. per loop detector)
    Another auxiliary file called "indexes_all.txt" stores the LDs names for cross-checking.
    '''

    tdf = TimeDataFrame(filename)
    keys = tdf.fetch_keys()

    print keys

    medians = []
    # diff_medians = []

    ind = 1
    with open('medians_all.txt', 'w') as f, open('indexes_all.txt', 'w') as fi:
        for i in range(1, len(keys)):
            print(ind)
            print(i)
            print(keys[i])
            try:
                key_series = tdf.fetch_series(keys[i])
                pa = PointwiseAnalysis(key_series, 96)
                # diff_medians.append(pa.diff_med_series().values.tolist())
                # fi.write(str(ind) + "," + str(keys[i]))
                fi.write("%d," % ind)
                fi.write("%s\n" % keys[i])
                f.write("%s\n" % pa.med_series().values.tolist())
                ind += 1
            except:
                print('key {} processing error'.format(keys[i]))


if __name__ == '__main__':
    filename = 'MA33_2011_1_15.csv'
    main(filename)

