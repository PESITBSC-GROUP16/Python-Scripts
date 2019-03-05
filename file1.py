import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
    
df = pd.read_csv('dataset.csv')
# remove duplicate column
df = df.drop(['Measurement Timestamp Label'], axis = 1)
    
#sort based  on beach name
df.sort_values('Beach Name', inplace = True)

#df.count: 34923

#count missing number of missing value
print(df.isnull().sum())

"""
    Beach Name                   0
    Measurement Timestamp        6
    Water Temperature            6
    Turbidity                    6
    Transducer Depth         24889
    Wave Height                233
    Wave Period                233
    Battery Life                 6
    Measurement ID               0
    dtype: int64
"""
#remove rows without timestamp
df = df[df['Measurement Timestamp'].notnull()]
#df.count: 34917
print(df.isnull().sum())
"""
    Beach Name                   0
    Measurement Timestamp        0
    Water Temperature            0
    Turbidity                    0
    Transducer Depth         24883
    Wave Height                227
    Wave Period                227
    Battery Life                 0
    Measurement ID               0
    dtype: int64
"""

df2 = df.drop(df.loc[df['Wave Height'] == -99999.992].index)
df2 = df.drop(df.loc[df['Wave Period'] == -100000].index)
df2.sort_values('Beach Name', inplace = True)

df2.median(axis = 0)

waveHeight_median = 0.156
wavePeriod_median = 3.0
tras_depth = 1.578

df2 = df2.fillna({'Wave Height':waveHeight_median, 'Wave Period':wavePeriod_median})
print(df2.isnull().sum())

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
df2.iloc[:, 0] = labelencoder_X.fit_transform(df2.values[:, 0])

df2.drop(['Transducer Depth'], axis = 1, inplace = True)

df3 = df2

#Feature Scaling
#Water Temperature, Turbidity, Wave Height, Wave Period, Battery Life, Measurement Timestamp, Measurement ID
scaler = StandardScaler()
df3[['Water Temperature', 'Turbidity', 'Wave Height', 'Wave Period', 'Battery Life']] = scaler.fit_transform(df3[['Water Temperature', 'Turbidity', 'Wave Height', 'Wave Period', 'Battery Life']])

#df3 = df3.reset_index('Beach Name', drop = True)
df3 = df3.rename(columns={'Beach Name': 'BeachName'})

#write to csv
df3.to_csv("preprocessed.csv", encoding = 'utf-8', index = False, mode = 'w')

df3 = pd.read_csv("preprocessed.csv")

beaches = list(df3["BeachName"].unique())
numBeaches = len(beaches)
divideByBeachName = {}
for i in range(0, numBeaches, 1):
    divideByBeachName[i] = df3.query("BeachName=={0}".format(i))
    divideByBeachName[i].reset_index(inplace=True, drop=True)
    dfTemp = pd.DataFrame(divideByBeachName[i], columns = list(df3.columns))
    csvName = "Beach"+str(beaches[i])+".csv"
    dfTemp.to_csv(csvName, encoding = 'utf-8', index = False, mode = 'w')
