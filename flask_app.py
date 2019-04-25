from flask import Flask, request, jsonify, Response
import pandas as pd
import numpy as np
from random import randint
import gspread
import pickle
import io
import matplotlib.pyplot as plt
import base64
import json
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.svm import OneClassSVM


app = Flask(__name__)

#root
@app.route("/")
def index():
    return "ROOT"

def getOutliers(dataset, model):
    pred = model.predict(dataset)
    pred2 = model.decision_function(dataset)
    outlierRows = [i for i in range(len(pred)) if pred[i]==-1]
    print("Pred: ",pred, "  ", pred2)
    ctr = 0
    for i in range(len(pred)):
        if pred[i] == -1:
            print("Outlier", " ", i)
            ctr = ctr + 1
    return ctr, outlierRows, pred2

    
@app.route("/beach", methods = ['POST', 'GET'])
def getDeviceId1():
    if request.method =='POST':
    #if True:
        devId = request.form['device_id']
        #devId = "IOT01"
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
        client = gspread.authorize(creds)
        
        sheet = client.open("Copy of Final_Year_Dataset")

        device = int(devId[-1])
        worksheet = sheet.get_worksheet(device)
        deviceParams = worksheet.get_all_values()
        length = len(deviceParams)
        cols = deviceParams[0]
        waterTemp = []
        turbidity = []
        waveHeight = []
        wavePeriod = []
        battery = []
        for i in range(1, len(deviceParams)):
            row = deviceParams[i]
            beachName = (int(row[0]))
            waterTemp.append(float(row[2]))
            turbidity.append(float(row[3]))
            waveHeight.append(float(row[4]))
            wavePeriod.append(float(row[5]))
            battery.append(float(row[6]))

        dataset = [waterTemp, turbidity, waveHeight, wavePeriod, battery]
        
        
        """print(waterTemp)
        print(turbidity)
        print(waveHeight)
        print(wavePeriod)
        print(battery)"""

        outliers = []

        print(cols)
        print("################################")
        print(dataset)
        print("################################")

        i = 0
        outlierRows = []
        predScore = []
        for col in range(2, len(cols)-1):
            modelFile = "./Models/"+str(device)+"_"+str(cols[col])+".pkl"
            print(modelFile)
            with open(modelFile, 'rb') as f:
                model = pickle.load(f)
            print("Col: ", cols[col])
            temp, outlierRow, pred = getOutliers(np.reshape(dataset[i], (-1, 1)), model)
            outlierRows.append(outlierRow)
            predScore.append(pred)
            i = (i + 1)%len(dataset)
            outliers.append(temp)
        
        #row index starts from 1
        worksheet2 = sheet.get_worksheet(6)
        threshold = worksheet2.row_values(device+1)
        
        print("################################")
        print("NUMS: ",outliers)
        percent = [x/length for x in outliers]
        print("Percent: ", percent)
        for i in range(1, len(percent)+1):
            worksheet2.update_cell(device+1, i, percent[i-1])
        
        combinedList = list(percent)
        combinedList.extend(threshold)
        
        j = 2
        plt.figure(figsize=(15,15))
        for i in range(0, 5):
            p_ng = outlierRows[i]
            p_ok = np.delete(np.arange(0, length-1), p_ng)
            p_ng_score = predScore[i][p_ng]
            p_ok_score = predScore[i][p_ok]
            
            plt.subplot(2, 3, i+1)
            plt.scatter(p_ok_score, np.zeros(len(p_ok_score))+i, c = "green")
            plt.scatter(p_ng_score, np.zeros(len(p_ng_score))+i, c = "purple")
            plt.title(cols[j])
            plt.xlabel("Scores")
            plt.ylabel(cols[j])
            j=j+1
        
        img = io.BytesIO()
        plt.savefig(img, format='jpg')
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
        
        response = {"out": combinedList,"image": plot_url}
        print('Sending:', json.dumps(response))

        return Response(json.dumps(response), mimetype='application/json', status=200)
        
        
        #combinedList is the one you want
        
        #reset worksheet
        """worksheet.resize(rows = 1)
        worksheet.resize(rows = 1000)"""
        
        
    return "OK2"
    #return send_file("temp.jpg", mimetype='image/jpg')
    
    
@app.route('/plot')
def build_plot():

    img = io.BytesIO()

    y = [1,2,3,4,5]
    x = [0,2,1,3,4]
    plt.plot(x,y)
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return '<img src="data:image/png;base64,{}">'.format(plot_url)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


"""
---> decision fuction usage to be discusses
---> spliting test and train to be discussed
---> time order selection to be discusses

RESULTS

---> Decision function
---> 

"""
