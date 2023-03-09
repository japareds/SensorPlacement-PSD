# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:22:06 2021

@author: HP 840 G3
"""

#!/usr/bin/env python

import pandas as pd
from sodapy import Socrata
import os
#%%
abs_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'

client = Socrata("analisi.transparenciacatalunya.cat",None, username="pferrer@ac.upc.edu", password="ref-stat-CAT-captor-2020")

keyinput1 = input("Start date, formato YYYY-MM-DD: ")

keyinput2 = input("End date, formato YYYY-MM-DD: ")

cont = input("Specify pollutant (O3):")
if not cont:
   cont = "O3"

# Reference stations: names and code
RefStations = {'Palau-Reial':'08019057',
               'Ciutadella':'08019050',
               'Eixample':'08019043',
               'Gracia':'08019044',
               'Vall Hebron':'08019054',
               'Fabra':'08019058',
               'Sant-Adria':'08194008',# not in BCN
               'Badalona':'08015021',
               'El-Prat':'08169009',
               'Tona':'08283004',
               'Manlleu':'08112003',
               'Vic':'08298008'}

print('List of Reference Stations')
for i,j in enumerate(list(RefStations)):
    print(i,j)
    
idx = input("Select station number (0): ")
localitzacio = RefStations[list(RefStations)[int(idx)]]
if not localitzacio:
    print('Location not specified. Palau Reial data will be downloaded')
    localitzacio = "08019057"
    

filename = input("Output file name:")
if not filename:
    filename = f'{cont}_{list(RefStations)[int(idx)]}_{keyinput1}_{keyinput2}'

filename = file_path + filename
print(f'Data set will be downloaded into:\n{filename}')
results = client.get("tasf-thgu", where="data between '" + keyinput1 + "' and '" + keyinput2 + "'", contaminant=cont,
                     codi_eoi=localitzacio, limit=2000)
# Convert to pandas DataFrame
results_df = pd.DataFrame.from_records(results)

#%% Download data set
file = open(filename+"aux.csv","w+")

for index, row in results_df.iterrows():
    date = row["data"].split("T")[0]
    for i in range(0, 24):
        file.write(date)
        file.write("T"+str(i).zfill(2)+":00;")
        data = "h"+str(i+1).zfill(2)
        value = row[data]
        file.write(str(value))
        file.write("\n")
file.close()

os.system("sort "+filename+"aux.csv > "+filename+".csv")
os.system("rm "+filename+"aux.csv")