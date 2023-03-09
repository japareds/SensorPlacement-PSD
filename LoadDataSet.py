#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Reference Stations data set and pre-processing

Created on Fri Feb 24 12:55:10 2023

@author: jparedes
"""
import os
import pandas as pd
#%%
class dataSet():
    def __init__(self,pollutant,start_date,end_date,RefStations):
        self.RefStations = RefStations # reference stations to load
        self.pollutant = pollutant # pollutant to load
        self.startDate = start_date 
        self.endDate = end_date 
        self.ds = pd.DataFrame()
    def load_dataSet(self,file_path):
        for rs in self.RefStations:
            fname = f'{file_path}{self.pollutant}_{rs}_{self.startDate}_{self.endDate}.csv'
            print(f'Loading data set {fname}')
            df_ = pd.read_csv(fname,header=None,index_col=0,names=[self.pollutant+'_'+rs])
            df_.index = pd.to_datetime(df_.index)
            self.ds = pd.concat([self.ds,df_],axis=1)
        print(f'All data sets loaded\n{self.ds.shape[0]} measurements for {self.ds.shape[1]} reference stations')
    def cleanMissingvalues(self,strategy='remove'):
        print(f'Missing values found in data set:\n{self.ds.isna().sum()}')
        if strategy == 'remove':
            print('Removing missing values')
            self.ds.dropna(inplace=True)
            print(f'Entries with missing values remiaining:\n{self.ds.isna().sum()}')
            print(f'{self.ds.shape[0]} remaining measurements')
        elif strategy == 'interpolate':
            print('Interpolating missing data')
            self.ds = self.ds.interpolate(method='linear')
            print(f'Entries with missing values remiaining:\n{self.ds.isna().sum()}')

#%%

def main():
   dataset = dataSet(pollutant,start_date,end_date)
   dataset.load_dataSet(RefStations,file_path)
   dataset.cleanMissingvalues(strategy='interpolate')
   dataset.cleanMissingvalues(strategy='remove')# remove persisting missing values
   return dataset
   
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    RefStations = ['Badalona','Ciutadella','Eixample','El-Prat','Fabra','Gracia','Manlleu','Palau-Reial','Sant-Adria','Tona','Vall_Hebron','Vic']
    pollutant = 'O3'
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    ds = main()
    
    
    
    
