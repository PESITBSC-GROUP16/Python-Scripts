import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope
from IPython.display import display

def ellipticCurve(dataset):
    classifier = EllipticEnvelope(contamination = outlierFraction)
    classifier.fit(dataset)
    predScore = classifier.decision_function(dataset)
    pred = classifier.predict(dataset)
    outlierRows = [i for i in range(len(pred)) if pred[i]==-1]
    return predScore, outlierRows

def oneClassSVM(dataset):
    classifier = OneClassSVM(nu = outlierFraction, gamma = 0.03)
    classifier.fit(dataset)
    predScore = classifier.decision_function(dataset).T[0]
    pred = classifier.predict(dataset)
    outlierRows = [i for i in range(len(pred)) if pred[i]==-1]
    return predScore, outlierRows

df = pd.read_csv("./preprocessed.csv")
beaches = list(df["BeachName"].unique())
numBeaches = len(beaches)

cols = list(df.columns)
colSize = len(cols)
noStrCols = cols
del(noStrCols[1])
del(noStrCols[6])

divideByBeachName = {}
for i in range(0, numBeaches, 1):
    divideByBeachName[i] = df.query("BeachName=={0}".format(i))
    divideByBeachName[i].reset_index(inplace=True, drop=True)

#check if dataset has outliers
"""
for i in range(0, numBeaches):
    fig = plt.figure()
    subPlot = fig.add_subplot(111)
    subPlot.boxplot(np.array(divideByBeachName[i][noStrCols]))
    subPlot.set_xticklabels(noStrCols)
    plt.title(beaches[i])
    plt.grid()
plt.show()
""" 

dfDic = {}

for beach in beaches:
    csvName = "Beach"+str(beach)+".csv"
    dfDic[beach] = pd.read_csv(csvName)

colsToAnalyze = noStrCols
numRows = {}
for i in range(0, numBeaches, 1):
    numRows[i] = dfDic[i].shape[0]

outlierFraction = 0.01
ran = np.random.RandomState(123)
#anomalyList = ["ellCurve", "svm"]
anomalyList = ["svm"]

dfDic[beach][noStrCols]

predictions = {}
for beach, data in divideByBeachName.items():
    predictions[beach] = {}
    
    """s, o = ellipticCurve(dfDic[noStrCols])
    predictions[beach]["ellCurve"] = {"ScorePred":s, "outliers":o}}}"""
    
    s, o = oneClassSVM(dfDic[beach][noStrCols])
    predictions[beach]["svm"] = {"ScorePred":s, "outliers":o}

statsCols = ["BeachName", "dataSize", "normals", "anomalies", "anomaliesRate"]
outliers = {}
for index in beaches:
    outliers[index] = {}

for anomalyType in anomalyList:
    dataSize = []
    oks = []
    ngs = []
    ngRate = []
    for i in range(0, len(beaches)):
        p_ng = predictions[i][anomalyType]["outliers"]
        p_ok = np.delete(np.arange(0, numRows[i]), p_ng)
        p_ng_score = predictions[i][anomalyType]["ScorePred"][p_ng]
        p_ok_score = predictions[i][anomalyType]["ScorePred"][p_ok]
        
        dataSize.append(numRows[i])
        oks.append(len(p_ok))
        ngs.append(len(p_ng))
        ngRate.append(round(100*len(p_ng)/numRows[i], 2))
        
        for n in p_ng:
            outliers[beaches[i-1]][n] = 1 if n not in outliers[beaches[i-1]].keys() else outliers[beaches[i-1]][n]+1
            
        plt.scatter(p_ok_score, np.zeros(len(p_ok_score))+i, c = "blue")
        plt.scatter(p_ng_score, np.zeros(len(p_ng_score))+i, c = "red")
        
    print(anomalyType)
    stats = pd.DataFrame(np.array([beaches, dataSize, oks, ngs, ngRate]).T, columns = statsCols)
    display(stats)
    
    plt.title(anomalyType)
    plt.xlabel("Scores")
    plt.ylabel("beaches")
    plt.grid(True)
    plt.show()


train, test = train_test_split(dfDic[0][noStrCols], test_size = .2)
    
classifier = OneClassSVM(nu = outlierFraction, gamma = 0.03)
classifier.fit(train)
predScore = classifier.decision_function(train).T[0]
pred = classifier.predict(test)
outlierRows = [i for i in range(len(pred)) if pred[i]==-1]
        
    

    