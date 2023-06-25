import pandas as pd
import numpy as np
from numpy import mean, std
from sklearn.preprocessing import normalize
import warnings
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import pickle


def extract_startp(insulinFile):
    col_list = ['Date','Time','BWZ Carb Input (grams)']
    df = pd.read_csv(insulinFile, 
                 usecols = col_list, 
                 low_memory=False,
                 )
    df=df[df['BWZ Carb Input (grams)']>0] 
    df['DT'] = pd.to_datetime(df['Date'] + ' ' +df['Time']).astype('int64')//1e9

    """ This is the time ponts of starting meal in data.
    Each element in the list is timestamp of Carb input greater than 0
    The the timestamps in list carb_in are sorted descending. 
    if time[n] - time[n+1]< 2hours, we eliminate time[n+1]. Recurse this
    process until we get the time duration greater than 2
    From this list, we extract to lists: Meal and noMeal
    
    """    

    carb_in = df['DT'].to_numpy()
    meal_time = []
    nomeal_time = []


    """ Getting no Meal time first:
    The no Meal time will start from the smaller timestamp if the gap between
    two consecutive timestamps is greater than or equa to 4 hours.
    a - b > 4 hours, the fist no meal timestamp will be b + 2 hours.
    Repeat this interval until no meal timestamp greater than a.
    Note: 2 hours equal to 7200 seconds
    """
    i = 0
    while i < len(carb_in)-1:
        duration = carb_in[i] - carb_in[i+1]
        if duration >= 7200*2 :
            nomeal= carb_in[i+1]+7200
            nomeal_time.append(nomeal)
            while nomeal + 7200 <= carb_in[i]:
                nomeal += 7200
                nomeal_time.append(nomeal)
        i += 1

    """ Getting Meal time
    The meal timestamp will be extract from carb_in list
    We calculate duration of 2 consecutive timestamps. If the duration is less
    than 2 hours, the earlier timestamp will be deleted.
    """
    meal_time.append(carb_in[0])
    i = 0
    while i < len(carb_in)-1:
        duration = carb_in[i] - carb_in[i+1]
        if duration < 7200 :
            carb_in= np.delete(carb_in, i+1)
        else:
            meal_time.append(carb_in[i+1])
            i += 1
    
    meal_st = pd.DataFrame(meal_time,columns = ['st'])
    meal_st['st']=meal_st['st']-1800
    meal_st['DT'] = meal_st['st']

    nomeal_st = pd.DataFrame(nomeal_time, columns = ['st'])
    nomeal_st['DT'] = nomeal_st['st']
    return meal_st, nomeal_st


def extract_cgm(meal,cgmfile,mtype):
    if mtype == 1:
        k = 30
    else:
        k = 24
        
    col_list = ['Date','Time','Sensor Glucose (mg/dL)']
    df = pd.read_csv(cgmfile, usecols = col_list, low_memory=False)

    df['DT'] = pd.to_datetime(df['Date'] + ' ' +df['Time']).astype('int64')//1e9
    df['cgm']=df['Sensor Glucose (mg/dL)']

    df= df.drop(col_list,axis = 1)
    df = pd.concat([df,meal])

    df = df.sort_values(by=['DT'],ascending = False)
    df['st'] = df['st'].fillna(method = 'backfill')
    df = df.set_index('st')
    df['timestamp']=(df['DT']-df.index)//300
    df = df[df['timestamp']<k]
    df = df.pivot_table(values = 'cgm', index = df.index, columns = 'timestamp')
    return df


# Feature engineering for Meal
def feature_extraction(df):
    ft = pd.DataFrame()
    ft['f1'] = df.max(axis = 1, skipna = True) - df.min(axis = 1, skipna = True) 

    # Feature 3: normalized the range of data (max-min)/min
    ft['f3'] =(df.max(axis=1,skipna=True) - df.min(axis=1,skipna=True))\
        /df.min(axis=1,skipna=True)
        
    # Feature 4: Mean value in time series
    ft['f4']= df.mean(axis = 1, skipna = True)
    
    # Feature 5: Starndard diviation of time series
    ft['f5'] = df.std(axis = 1, skipna= True)
    
    # Feature 6: Max CGM - CGM at meal time 
    ft['f6'] = (df.max(axis = 1, skipna = True) -df.min(axis=1,skipna=True)) \
        /df.min(axis=1,skipna=True)
    
    # Feature 7 and 8: Duration of first max and second max GCM
    """
    arr = np.argsort(-df.values, axis = 1)
    fft_index = pd.DataFrame(df.columns[arr], index = df.index)
    ft['f7'] = fft_index[1]
    ft['f8'] = fft_index[2]
    
    fft = df.fillna(0)
    fft_val = pd.DataFrame(np.sort(fft.values)[:,-3:], columns = [1,2,3], index = df.index)
    ft['f9'] = fft_val[2]
    ft['f10'] = fft_val[1]
    """

    return ft


# First patient data
meal_st, nomeal_st = extract_startp("InsulinData.csv")
meal_cgm_p1 = extract_cgm(meal_st,"CGMData.csv",1)
nomeal_cgm_p1 = extract_cgm(nomeal_st,"CGMData.csv",0)

# Second patient data
meal_st, nomeal_st = extract_startp("Insulin_patient2.csv")
meal_cgm_p2 = extract_cgm(meal_st,"CGM_patient2.csv",1)
nomeal_cgm_p2 = extract_cgm(nomeal_st,"CGM_patient2.csv",0)

meal_cgm = pd.concat([meal_cgm_p1, meal_cgm_p2])
nomeal_cgm = pd.concat([nomeal_cgm_p1, nomeal_cgm_p2])

# Drop instances with more than 1/3 is missing values
meal_cgm = meal_cgm.dropna(thresh=20)
nomeal_cgm = nomeal_cgm.dropna(thresh = 16)


# Feature extraction for Meal and No meal data 
meal_fe = feature_extraction(meal_cgm)
meal_fe['Class'] = 1

nomeal_fe = feature_extraction(nomeal_cgm)    
nomeal_fe['Class'] = 0

df = pd.concat([meal_fe, nomeal_fe])


X = df.drop('Class',axis = 1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82)

clf = RandomForestClassifier().fit(X_train, y_train)
pickle.dump(clf, open('model.pickle', 'wb'))


cv = KFold(n_splits = 5, random_state = 1, shuffle = True)
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, scoring = 'accuracy', cv=cv, n_jobs = -1)

print('Accuracy: %.3f (%.3f)' %(mean(scores), std(scores)))