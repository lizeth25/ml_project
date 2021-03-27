"""
Lizeth Lucero
"""


import csv
import math


# Load data into python from CSV file
with open("deidentified_data.csv") as fp:
    rows = csv.reader(fp)
    header = next(rows) # I assume your datasets have a row with column-labels!
    for row in rows:
        print(header) # print the labels
        print(row) # print the current row
        # entry = dict(zip(header, row)) # glue them into a dict
        # print(entry) # print that dict
	# want to remove rows that have NaN for ECG or skin tone
	if (not row[0] == 'NaN') and (not row[8] == 'NaN'):
		entry = dict(zip(header, row)) # glue them into a dict
        	break # stop after 1 row of data, so you can inspect, choose which columns matter, etc.


'''
I think that my task definition works well for this project
I think that I need to think more about the training error that I will accept. 
Because this project examines heartbeats, I think that I want to do 
some more research with error rates that I can accept. 


Cleaning up data: 
I decided to remove any data points which had a NaN for ECG because
I am using those as the 'true' result. Additionally, I will be removing any data that has
NaN for skin color because I am hoping to use that for my algorithm. 


Data Types:


'''
