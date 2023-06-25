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

# Read test.csv file.
df = pd.read_csv("test.csv", low_memory=False)

test_result = feature_extraction(df)

clf = pickle.load(open('model.pickle', 'rb'))
result = clf.predict(test_result)
result = result.transpose()
print(result)
# np.savetxt('result.txt', result, fmt='%1.4e')
#np.savetxt('result.csv', result, fmt="%d", delimiter=",")
result= pd.DataFrame(result)
result.to_csv('result.csv', header = False, index=False)