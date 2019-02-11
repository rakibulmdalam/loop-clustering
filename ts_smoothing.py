import numpy as np
from timedataframe import TimeDataFrame
from pointwiseanalysis import PointwiseAnalysis


def main(filename):
    tdf = TimeDataFrame(filename)
    keys = tdf.fetch_keys()

    medians = []
    diff_medians = []

    for i in range(1, len(keys)):
        print(i)
        print(keys[i])
        try:
            key_series = tdf.fetch_series(keys[i])
            pa = PointwiseAnalysis(key_series, 96)
            medians.append(pa.med_series().values.tolist())
            diff_medians.append(pa.diff_med_series().values.tolist())
        except:
            print('key {} processing error'.format(keys[i]))

    with open('{}_medians_all.txt'.format(filename), 'w') as f:
        for item in medians:
            f.write("%s\n" % item)

    with open('{}_diff_medians_all.txt'.format(filename), 'w') as f:
        for item in diff_medians:
            f.write("%s\n" % item)



if __name__ == '__main__':
    filename = 'raw_data_files/MA33_2011_1_15.csv'
    main(filename)

