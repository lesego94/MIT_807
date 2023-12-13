import pandas as pd
import numpy as np
# python -m pip install yahoo-finance; pip install yahoo-finance; pip install yfinance --upgrade --no-cache-dir
import yfinance as yf
# !pip install --upgrade ta; pip install ta
# import ta as ta
import sklearn as sk
from sklearn import preprocessing
from scipy.stats import t
import tensorflow as tf
from datetime import date, datetime, timedelta
from arch import arch_model
import matplotlib.pyplot as plt

from tqdm import tqdm

import re
from arch.__future__ import reindexing

import multiprocessing as mp



#Return calculation
def ReturnCalculation (Database,lag):
    dimension=Database.shape[0];dif=lag;Out=np.zeros([dimension-dif])
    for i in range(dimension-dif):
        Out[i]=(np.log(Database['Close'][i+dif])-np.log(Database['Close'][i]))
    return np.append(np.repeat(np.nan, dif),Out), Database.index

#STD Calculation
def SDCalculation (DailyReturns, LagSD):
    dimension=DailyReturns.shape[0]; dif=LagSD; Out=np.zeros([dimension-dif])
    for i in range (dimension-dif):
        Out[i]=np.std(DailyReturns[i:i+LagSD],ddof=1)
    return np.append(np.repeat(np.nan, dif),Out)

#STD Calculation
def TrueSDCalculation (DailyReturns, LagSD):
    dimension=DailyReturns.shape[0]; dif=LagSD; Out=np.zeros([dimension-dif+1])
    for i in range (dimension-dif+1):
        Out[i]=np.std(DailyReturns[i:i+LagSD],ddof=1)
    return np.append(Out,np.repeat(np.nan, dif-1))

#Database is calculated
def M_DatabaseGeneration (Database_daily, Lag, LagSD):
    DailyReturns, Index = ReturnCalculation(Database_daily,Lag)    
    TrueSD = TrueSDCalculation(DailyReturns, LagSD)    
    Data = pd.DataFrame({'DailyReturns': DailyReturns,'TrueSD': TrueSD})
    Data = Data.set_index(Index)
    Data = Data.dropna() 
    weekly_returns = Data['DailyReturns'].resample('W-FRI').sum()
    weekly_average_volatility = Data['TrueSD'].resample('W-FRI').mean()*np.sqrt(5)
 
    
    Data = pd.DataFrame({'DailyReturns': weekly_returns,'TrueSD': weekly_average_volatility})
    return Data.dropna()

#Fitting of GARCH(1,1)
def GARCH_Model_Student (Data):
    AR_Data=Data['DailyReturns']*100
    GARCH11 = arch_model(AR_Data, dist ='t')
    res_GARCH11 = GARCH11.fit(disp='off')
    CV_GARCH11 = res_GARCH11.conditional_volatility
    For_CV_GARCH11 = np.array(res_GARCH11.forecast(horizon=4).variance.dropna())
    return GARCH11, res_GARCH11, CV_GARCH11, For_CV_GARCH11

#Fitting of GJR_GARCH(1,1)
def GJR_GARCH_Model_Student (Data):
    AR_Data=Data['DailyReturns']*100
    GJR_GARCH11 = arch_model(AR_Data, p=1, o=1, q=1, dist ='t')
    res_GJR_GARCH11 = GJR_GARCH11.fit(disp='off')
    CV_GJR_GARCH11 = res_GJR_GARCH11.conditional_volatility
    For_CV_GJR_GARCH11 = np.array(res_GJR_GARCH11.forecast(horizon=4).variance.dropna())
    return GJR_GARCH11, res_GJR_GARCH11, CV_GJR_GARCH11, For_CV_GJR_GARCH11

#Fitting of TARCH(1,1)
def TARCH_Model_Student(Data):
    AR_Data=Data['DailyReturns']*100
    TARCH11 = arch_model(AR_Data, p=1, o=1, q=1, power=1.0, dist ='t')
    res_TARCH11 = TARCH11.fit(disp='off')
    CV_TARCH11 = res_TARCH11.conditional_volatility
    For_CV_TARCH11 = np.array(res_TARCH11.forecast(horizon=4,method= "bootstrap").variance.dropna())
    return TARCH11, res_TARCH11, CV_TARCH11, For_CV_TARCH11

# #Fitting of EGARCH(1,1)
# def EGARCH_Model_Student(Data):
#     AR_Data=Data['DailyReturns']*100
#     EGARCH11 = arch_model(AR_Data, dist ='t', vol="EGARCH")
#     res_EGARCH11 = EGARCH11.fit(disp='off')
#     CV_EGARCH11 = res_EGARCH11.conditional_volatility
#     For_CV_EGARCH11 = np.array(res_EGARCH11.forecast(horizon=4,method="bootstrap").variance.dropna())
#     return EGARCH11, res_EGARCH11,CV_EGARCH11, For_CV_EGARCH11

#Fitting of TARCH(1,1)
def EGARCH_Model_Student(Data):
    AR_Data=Data['DailyReturns']*100
    EGARCH11 = arch_model(AR_Data, p=1, o=1, q=1, power=1.0, dist ='t')
    res_EGARCH11 = EGARCH11.fit(disp='off')
    CV_EGARCH11 = res_EGARCH11.conditional_volatility
    For_CV_EGARCH11 = np.array(res_EGARCH11.forecast(horizon=4,method= "bootstrap").variance.dropna())
    return EGARCH11, res_EGARCH11,CV_EGARCH11, For_CV_EGARCH11


#Fitting of Absolute Value GARCH(1,1)
def AVGARCH_Model_Student(Data):
    AR_Data=Data['DailyReturns']*100
    AVGARCH11 = arch_model(AR_Data, dist ='t', power=1)
    res_AVGARCH11 = AVGARCH11.fit(disp='off',options={'maxiter': 1000})
    CV_AVGARCH11 = res_AVGARCH11.conditional_volatility
    For_CV_AVGARCH11 = np.array(res_AVGARCH11.forecast(horizon=4,method="bootstrap").variance.dropna())
    return AVGARCH11, res_AVGARCH11, CV_AVGARCH11, For_CV_AVGARCH11

#Fitting of FIGARCH11(1,1)
def FIGARCH_Model_Student(Data):
    AR_Data=Data['DailyReturns']*100
    FIGARCH11 = arch_model(AR_Data, dist ='t', vol="FIGARCH")
    res_FIGARCH11 = FIGARCH11.fit(disp='off')
    CV_FIGARCH11 = res_FIGARCH11.conditional_volatility
    For_CV_FIGARCH11 = np.array(res_FIGARCH11.forecast(horizon=4,method="bootstrap").variance.dropna())
    return FIGARCH11, res_FIGARCH11, CV_FIGARCH11, For_CV_FIGARCH11


def process_index(index):
    #Index of end dates, database for validation and dataframe to collect the results are created. Model variables are defined.
    Start='2008-01-01'; End='2015-12-31'; 
    asset = index
    asset_name = re.sub('[\W\d_]+', '', asset)
    IndexEndDays=yf.download(asset,start=Start,  end=End, progress=False).resample('W-FRI').last().index

    Lag=1; LagSD=5; Timestep=10; Dropout=0.1; LearningRate=0.01; Epochs=100

    DataValidation = M_DatabaseGeneration(yf.download(asset,start='2000-01-01', end=date.today()+timedelta(days=1), progress=False).resample('W-FRI').last(), Lag, LagSD)

    ResultsCollection=pd.DataFrame({'Date_Forecast': [], 'h1': [], 'h2': [], 'h3':[], 'h4': [],'TrueSD':[]})
    #Loop for generating the results
    for i in tqdm(range(IndexEndDays.shape[0])):
        #Database is downloaded from yahoo finance and lag of returns defined
        # Database=yf.download(asset,start=IndexEndDays[i].date()-timedelta(days=780), end=IndexEndDays[i].date(), progress=False).resample('W-FRI').last()
        #Daily finance datga
        Database_daily = yf.download(asset,start=IndexEndDays[i].date()-timedelta(days=780), end=IndexEndDays[i].date(), progress=False)
        #Database for fitting the models is generated
        Data = M_DatabaseGeneration(Database_daily, Lag, LagSD)
        #Fitting of Transformed ANN-ARCH model, ARCH models and forecasting of the next volatility value
        # T_ANN_ARCH_Model = GARCH_Model_Student (Data,Database_daily, Lag, LagSD, Timestep, Dropout, LearningRate, Epochs)
        # ARCH11, res_GARCH11, CV_GARCH11, GARCH = GJR_GARCH_Model_Student(Data)
        ARCH11, res_GARCH11, CV_GARCH11, GARCH =FIGARCH_Model_Student(Data)
        #Results are collected
        Date_Forcast = IndexEndDays[i]
        IterResults={'Date_Forecast':Date_Forcast.date(),'h1': GARCH[0][0]/100,'h2': GARCH[0][1]/100,'h3': GARCH[0][2]/100,'h4': GARCH[0][3]/100,'TrueSD':Data['TrueSD'][-1]}

        ResultsCollection.loc[len(ResultsCollection)] = IterResults
        #Results are saved
        ResultsCollection.to_csv(f'./assets/Drop001/FIGARCH/FIGARCH_{asset_name}.csv',index=False)
        
# Define the indices to process
indices = ['DHR','GILD','ISRG','JNJ','LLY','PFE','REGN','TMO','AZN','MRK'] 


def main():
    # Create a pool of 4 workers within a context
    with mp.Pool(6) as pool:
        # Start a process for each index
        results = pool.map(process_index, indices)



if __name__ == "__main__":
    main()


