# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 01:54:29 2021

@author: Truong Tran
Project 1: Extracting Time series properties of Glucose Levels in Artificial Pancreas
"""
import pandas as pd


"""
STEP 0: Define functions: find_automode and data_cleaning
Function find_automode: Find the time Auto mode start.
   The function find_automod() will read 3 columns Date, Time, and Alarm
   of the data Insulin. Keep only Alarm with 'AUTO MODE ACTIVE PLGM OFF'
   And return the earliest time this change happened. 
"""
def find_automode(data):
    col_list = ['Date','Time','Alarm']
    df = pd.read_csv(data, usecols = col_list, low_memory=False)

    df['DT']=pd.to_datetime(df['Date'] + ' ' +df['Time'])
#    df['Alarm']=df['Alarm'].str.lstrip().str.rstrip()
    df=df[df['Alarm']=='AUTO MODE ACTIVE PLGM OFF']
    return df['DT'].min()

"""
Function data_cleaning: This function will drop all rows have missing value
    and drop all days have less than 288 segments. For the days have more
    than 288 segments, only first 288 rows will be kept, there rest of the
    data will be excluded.
"""
def data_cleaning(dataframe):
    # Drop missing data
    df.dropna(axis=0, how='any', inplace=True)
    # Count number of segments and keep only data has 288 segment
    df['seg_count']=df.groupby(['Date'])['Sensor Glucose (mg/dL)'].transform('count')
    df=df[df['seg_count']==288]
    df=df.groupby('Date').head(288)
    return df

"""
STEP 2: Data manipulation
In this step: we will do some work below:
    - Step 2a: load INsulin Data and find the time auto mode starts
    - Step 2b: Load CGM data and cleaning, drop missing value and over collected
    - Step 2c: Creating some categorical variables for estimation
"""
# Step 2a: finding auto mode start and store at auto_mode variable
auto_mode=find_automode('InsulinData.csv')


col_list = ['Date','Time','Sensor Glucose (mg/dL)','ISIG Value']
df = pd.read_csv('CGMData.csv', usecols = col_list, low_memory=False)

#Step 2b: Cleaning data
#df = data_cleaning('CGMData.csv')

#Step 2c: Creating metrics features for CGM level and time intervals
# Create a new column with a shorter name for convenient in writing
df['cgm']=df['Sensor Glucose (mg/dL)']

# 1st metric: hyperglycemia (CGM>180 mg/dL)
df['cgm0']= [1 if 180<=x else 0 for x in df['cgm']]

# 2nd metric: hyperglycemia (CGM>250 mg/dL)
df['cgm1']= [1 if 250<x else 0 for x in df['cgm']]

#3rd metric: hyperglycemia (70 <= CGM <=180)
df['cgm2']= [1 if 70<x <180 else 0 for x in df['cgm']]

#4rd metric: hyperglycemia (70 <= CGM <=150)
df['cgm3']= [1 if 70<x <150 else 0 for x in df['cgm']]

#5th metric: hyperglycemia (CGM <70)
df['cgm4']= [1 if x<70 else 0 for x in df['cgm']]

#6th metric: hyperglycemia (CGM <54)
df['cgm5']= [1 if x<54 else 0 for x in df['cgm']]


# Create binary feature for daytime or overnight

df['DT']=pd.to_datetime(df['Date']+ ' ' + df['Time'])
df['diff']=(df['DT']-pd.to_datetime(df['Date']+ ' ' + '00:00:00'))
df['diff']=df['diff'].dt.total_seconds().astype(int)

df['daytime0']= [1 if x< 6*60*60 else 0 for x in df['diff']]
df['daytime1']= (df['daytime0']-1)*(-1)
df['daytime2']=1
    
df['mode']=[0 if x<auto_mode else 1 for x in df['DT']]

#Step 2b: Cleaning data
df.dropna(axis=0, how='any', inplace=True)
# Count number of segments and keep only data has 288 segment
df['seg_count']=df.groupby(['Date'])['cgm'].transform('count')
df=df[df['seg_count']==288]
#df=df.groupby('Date').head(288)
    

"""
From this point, we already have all features to estimate:
    6 metrics with name from cgm0 to cgm5
    3 binary features for daytime, nighttime and whole day
    auto-mode with 0 is manual mode and 1 is auto mode
STEP 3: A list name result will be create to store output values. 
    This list will have dimension 2x18 with 1st row is manual mode 
    and 2nd row is auto mode

To estimate the output values, the data will be filtered by mode type,
    daytime type. All cgm catergories will be aggregate percent value
    of each day. The mean value of percent of whole data will be estimate
    and append to result list.
Convert result list to DataFrame and write to Results.csv file
"""

result=[]
for i in range(2):
    df1 = df[df['mode']==i]
    row=[]
    for j in range(3):
        df2 = df1[df1['daytime'+str(j)]==1]
        varlist = ['cgm0','cgm1','cgm2','cgm3','cgm4','cgm5']
        df3=df2.groupby('Date')[varlist].sum()/288*100
        for k in range(6):
            output=df3['cgm'+str(k)].mean()
            row.append(output)
    result.append(row)
    
print(result)
final=pd.DataFrame(result)
final.to_csv('Results.csv',header=False,index=False)

        