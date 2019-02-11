import locale
import time
import datetime as d
import random
import math
import numpy as np
import pandas as pd

class PointwiseAnalysis:

    def __init__(self, timeseries, freq):

        self.freq = freq
        self.timeseries = timeseries
        self.df = timeseries.to_frame().reset_index()  # create dataframe from given timeseries
        self.df.columns = ['time', 'value']   # add column names to dataframe
        self.preprocess()
        self._exec()

    def preprocess(self):
        self.df['diff'] = self.df['value'].diff()
        self.df['med'] = 0
        self.df['diff_med'] = 0
        self.df['value'].fillna(-1, inplace=True)

    def add_day_column(self, row):
        locale.setlocale(locale.LC_ALL,'en_US.UTF-8')
        return d.datetime.strptime(row['time'], '%d.%m.%Y %H:%M').date().strftime("%A").lower()

    def _exec(self):
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        all_days = []
        for day in days:
            day = day.lower()
            self.df['day'] = self.df[['time']].apply(self.add_day_column, axis=1)
            day_df = self.df[self.df['day'] == day].reset_index()
            day_df.columns = ['oi', 'time', 'value', 'diff', 'med', 'diff_med', 'day']
            day_df = self._qd(day_df)
            day_df = self._diff_qd(day_df)
            all_days.append(day_df)
        
        self.new_df = pd.concat(all_days)
        self.new_df = self.new_df.sort_values('oi').set_index('oi').reset_index(drop=True)

    def add_interval_column(self, row): 
        return str(row['time']).split(" ")[1]
        
    def _qd(self, day_df):
        day_df['interval'] = day_df[['time']].apply(self.add_interval_column, axis=1)
        intervals = day_df['interval'].unique().tolist()
        all_intervals = []
        for interval in intervals:
            interval_df = day_df[day_df['interval'] == interval]
            median = np.percentile(interval_df[interval_df['value'] > -1]['value'].dropna().values, [50])
            interval_df['med'] = median[0]
            all_intervals.append(interval_df)

        day_df = pd.concat(all_intervals)
        return day_df.sort_index()

    def _diff_qd(self, day_df):
        day_df['interval'] = day_df[['time']].apply(self.add_interval_column, axis=1)
        intervals = day_df['interval'].unique().tolist()
        all_intervals = []
        for interval in intervals:
            interval_df = day_df[day_df['interval'] == interval]
            median = np.percentile(interval_df['diff'].dropna().values, [50])
            interval_df['diff_med'] = median[0]
            all_intervals.append(interval_df)

        day_df = pd.concat(all_intervals)
        return day_df.sort_index()

    def write(self, filename):
        self.new_df.to_csv(filename)

    def med_series(self):
        return self.new_df['med'][0:7*96]

    def diff_med_series(self):
        return self.new_df['diff_med'][0:7*96]
