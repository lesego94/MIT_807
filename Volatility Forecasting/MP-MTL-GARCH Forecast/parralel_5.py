import pandas as pd
import numpy as np
# python -m pip install yahoo-finance; pip install yahoo-finance; pip install yfinance --upgrade --no-cache-dir
import yfinance as yf
# !pip install --upgrade ta; pip install ta
import ta as ta
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

#AR models are fitted. As requested by arma package, returns are multiplied by 100 in order to improve the fitting process.
#GARCH(1,1), GJR_GARCH(1,1), TARCH(1,1), EGARCH(1,1), AVGARCH(1,1) and FIGARCH(1,1) volatility models are fitted.
#T student is assumed as distribution.
def AR_Models (Data):
    GARCH, GARCH_Parameters, CV_GARCH, For_CV_GARCH = GARCH_Model_Student(Data)
    GJR_GARCH, GJR_GARCH_Parameters, CV_GJR_GARCH, For_CV_GJR_GARCH = GJR_GARCH_Model_Student(Data)
    TARCH, TARCH_Parameters, CV_TARCH, For_CV_TARCH = TARCH_Model_Student(Data)
    EGARCH, EGARCH_Parameters,CV_EGARCH, For_CV_EGARCH = EGARCH_Model_Student(Data)
    AVGARCH, AVGARCH_Parameters,CV_AVGARCH, For_CV_AVGARCH = AVGARCH_Model_Student(Data)
    FIGARCH, FIGARCH_Parameters,CV_FIGARCH, For_CV_FIGARCH  = FIGARCH_Model_Student(Data)
    return GARCH_Parameters, CV_GARCH, For_CV_GARCH, GJR_GARCH_Parameters, CV_GJR_GARCH, For_CV_GJR_GARCH, TARCH_Parameters, CV_TARCH, For_CV_TARCH, EGARCH_Parameters,CV_EGARCH, For_CV_EGARCH, AVGARCH_Parameters,CV_AVGARCH, For_CV_AVGARCH, FIGARCH_Parameters,CV_FIGARCH, For_CV_FIGARCH

#MultiHeadSelfAttention
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        return output
        
#Transformer Keras Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.nb_dict = {}; self.Bagging=5
        for i in range(self.Bagging):
          self.nb_dict["att{0}".format(i)]=MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    def call(self, inputs, training):
        self.att_dict = {}
        for i in range(self.Bagging):
          self.att_dict["att{0}".format(i)]=self.nb_dict["att{0}".format(i)](tf.keras.layers.Dropout(.1)(inputs))
          if i==0: 
            self.att_dict["attn_output"]=self.att_dict["att{0}".format(i)]/self.Bagging 
          else: 
            self.att_dict["attn_output"]=self.att_dict["attn_output"]+self.att_dict["att{0}".format(i)]/self.Bagging
        attn_output = self.dropout1(self.att_dict["attn_output"], training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def Transformer_Model (Shape1, Shape2, HeadsAttention,Dropout, LearningRate):
    #Model struture is defined
    Input = tf.keras.Input(shape=(Shape1,Shape2), name="Input")
    #LSTM is applied on top of the transformer
    X = tf.keras.layers.LSTM(units=16, dropout=Dropout, return_sequences=True)(Input)
    #Tranformer architecture is implemented
    transformer_block_1 = TransformerBlock(embed_dim=16, num_heads=HeadsAttention, ff_dim=8, rate=Dropout)
    X = transformer_block_1(X)
    #Dense layers are used
    X = tf.keras.layers.GlobalAveragePooling1D()(X)
    X = tf.keras.layers.Dense(8, activation=tf.nn.sigmoid)(X)
    X = tf.keras.layers.Dropout(Dropout)(X)
    Output = tf.keras.layers.Dense(4, activation=tf.nn.sigmoid, name="Output")(X)
    model = tf.keras.Model(inputs=Input, outputs=Output)
    #Optimizer is defined
    Opt = tf.keras.optimizers.legacy.Adam(learning_rate=LearningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')
    #Model is compiled
    model.compile(optimizer=Opt, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def Transformer_Database (Timestep, XData_AR, YData_AR):
    Features = XData_AR.shape[1]
    Sample = XData_AR.shape[0] - Timestep - 2  # Adjusted to allow for a 4-step-ahead target
    XDataTrainScaledRNN = np.zeros([Sample, Timestep, Features])
    YDataTrainRNN = np.zeros([Sample, 4])  # Adjusted for 4-step-ahead forecasts
    
    for i in range(Sample):
        XDataTrainScaledRNN[i,:,:] = XData_AR[i:(Timestep+i)]
        YDataTrainRNN[i, :] = YData_AR[(Timestep+i-1):(Timestep+i+3)]  # 4-step-ahead target
    
    return XDataTrainScaledRNN, YDataTrainRNN


# #Database is calculated
# def DatabaseGenerationForecast (Database, Lag, LagSD):
#     DailyReturns, Index = ReturnCalculation(Database,Lag)
#     DailyReturnsOld =  np.append(np.repeat(np.nan, 1),DailyReturns[0:(DailyReturns.shape[0]-1)])
#     SD = SDCalculation (DailyReturns, LagSD)
#     TrueSD = TrueSDCalculation(DailyReturns, LagSD)
#     Data = pd.DataFrame({'DailyReturns': DailyReturns, 'SD': SD, 'TrueSD': TrueSD, 'DailyReturnsOld': DailyReturnsOld})
#     Data = Data.set_index(Index) 
#     return Data

#Database is calculated
def M_DatabaseGenerationForecast (Database_daily, Lag, LagSD):
    DailyReturns, Index = ReturnCalculation(Database_daily,Lag)    
    TrueSD = TrueSDCalculation(DailyReturns, LagSD)    
    Data = pd.DataFrame({'DailyReturns': DailyReturns,'TrueSD': TrueSD})
    Data = Data.set_index(Index)
    Data = Data.dropna() 
    weekly_returns = Data['DailyReturns'].resample('W-FRI').sum()
    weekly_average_volatility = Data['TrueSD'].resample('W-FRI').mean() *np.sqrt(5)
    
    Data = pd.DataFrame({'DailyReturns': weekly_returns,'TrueSD': weekly_average_volatility})
    return Data.dropna()

#Final AR database for forcasting is generated
def DatabaseGenerationForecast_AR (Database, Lag, LagSD, For_CV_GARCH, For_CV_GJR_GARCH, For_CV_TARCH, For_CV_EGARCH, For_CV_AVGARCH, For_CV_FIGARCH):
    Data_Forecast=M_DatabaseGenerationForecast(Database, Lag, LagSD).iloc[(-2+1)]
    Index_Forecast=M_DatabaseGenerationForecast(Database, Lag, LagSD).index[(-2+1)]
    XDataForecast=[]
    # Flatten the double-nested lists
    For_CV_GARCH = [item for sublist in For_CV_GARCH for item in sublist]
    For_CV_GJR_GARCH = [item for sublist in For_CV_GJR_GARCH for item in sublist]
    For_CV_TARCH = [item for sublist in For_CV_TARCH for item in sublist]
    For_CV_EGARCH = [item for sublist in For_CV_EGARCH for item in sublist]
    For_CV_AVGARCH = [item for sublist in For_CV_AVGARCH for item in sublist]
    For_CV_FIGARCH = [item for sublist in For_CV_FIGARCH for item in sublist]
    for i in range(len(For_CV_AVGARCH)):
        forecast={'CV_GARCH' : For_CV_GARCH[i]/100, 'CV_GJR_GARCH' : For_CV_GJR_GARCH[i]/100, 'CV_TARCH' : For_CV_TARCH[i]/100, 
               'CV_EGARCH' : For_CV_EGARCH[i]/100, 'CV_AVGARCH' : For_CV_AVGARCH[i]/100, 'CV_FIGARCH' : For_CV_FIGARCH[i]/100}
        XDataForecast.append(pd.DataFrame([forecast], index=[Index_Forecast]))
    XDataForecast = pd.concat(XDataForecast)
    return XDataForecast, Data_Forecast['DailyReturns']

#Transformed ANN-ARCH model forecast
def T_ANN_ARCH_Forecast (Database,Timestep, Lag, LagSD, For_CV_GARCH, For_CV_GJR_GARCH, For_CV_TARCH, For_CV_EGARCH, For_CV_AVGARCH, For_CV_FIGARCH,Scaled_Norm, XData_AR, model):
    XDataForecast, ReturnForecast = DatabaseGenerationForecast_AR (Database, Lag, LagSD, For_CV_GARCH, For_CV_GJR_GARCH, For_CV_TARCH, For_CV_EGARCH, For_CV_AVGARCH, For_CV_FIGARCH)
    XDataForecast = pd.concat([XData_AR,XDataForecast*1/np.sqrt(5)])
    XDataForecastTotalScaled = Scaled_Norm.transform(XDataForecast)
    XDataForecastTotalScaled_T, Y_T = Transformer_Database(Timestep, XDataForecastTotalScaled, np.zeros(XDataForecastTotalScaled.shape[0]))
    TransformerPrediction = model.predict(XDataForecastTotalScaled_T)
    return TransformerPrediction[-2], XDataForecast.index[-1], TransformerPrediction[0:(XDataForecastTotalScaled_T.shape[0]-1)], ReturnForecast


#Fitting of Transformed ANN-ARCH model and forecasting of the next volatility value
def T_ANN_ARCH_Fit (Data,Database,Lag=1, LagSD=5, Timestep=10, Dropout=0.05, LearningRate=0.01, Epochs=1000, BatchSize=64):
    GARCH, GARCH_Parameters, CV_GARCH, For_CV_GARCH = GARCH_Model_Student(Data)
    GJR_GARCH, GJR_GARCH_Parameters, CV_GJR_GARCH, For_CV_GJR_GARCH = GJR_GARCH_Model_Student(Data)
    TARCH, TARCH_Parameters, CV_TARCH, For_CV_TARCH = TARCH_Model_Student(Data)
    EGARCH, EGARCH_Parameters,CV_EGARCH, For_CV_EGARCH = EGARCH_Model_Student(Data)
    AVGARCH, AVGARCH_Parameters,CV_AVGARCH, For_CV_AVGARCH = AVGARCH_Model_Student(Data)
    FIGARCH, FIGARCH_Parameters,CV_FIGARCH, For_CV_FIGARCH  = FIGARCH_Model_Student(Data)
    #Database contaning AR models is generated
    Data_AR=pd.concat([Data, CV_GARCH.rename('CV_GARCH')/100, CV_GJR_GARCH.rename('CV_GJR_GARCH')/100, CV_TARCH.rename('CV_TARCH')/100, 
                        CV_EGARCH.rename('CV_EGARCH')/100, CV_AVGARCH.rename('CV_AVGARCH')/100, CV_FIGARCH.rename('CV_FIGARCH')/100], axis=1)
    if Data_AR.shape[0]!=Data.shape[0]: print("Error in DB Generation")
    # #Original explanatory and response variables are generated
    XData_AR = Data_AR.drop(['TrueSD','DailyReturns'], axis=1);YData_AR = Data_AR['TrueSD']
    # #Data is normalized
    Scaled_Norm = preprocessing.StandardScaler().fit(XData_AR); XData_AR_Norm = Scaled_Norm.transform(XData_AR)
    #Data for fitting the transformer model is generated
    XData_AR_Norm_T, YData_AR_Norm_T= Transformer_Database(Timestep, XData_AR_Norm, YData_AR)
    #Model with transformer layer is defined
    model = Transformer_Model(XData_AR_Norm_T.shape[1], XData_AR_Norm_T.shape[2], HeadsAttention=4, Dropout=Dropout, LearningRate=LearningRate)
    model.fit(XData_AR_Norm_T, YData_AR_Norm_T, epochs=Epochs, verbose=0, batch_size=BatchSize); tf.keras.backend.clear_session()
    Forecast, Date_Forecast, TrainPrediction, ReturnForecast = T_ANN_ARCH_Forecast (Database,Timestep, Lag, LagSD, For_CV_GARCH, For_CV_GJR_GARCH, For_CV_TARCH, For_CV_EGARCH, For_CV_AVGARCH, For_CV_FIGARCH,Scaled_Norm, XData_AR, model)
    return {'Date_Forecast':Date_Forecast,'Forecast_T_ANN_ARCH':Forecast }



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
        Database=yf.download(asset,start=IndexEndDays[i].date()-timedelta(days=780), end=IndexEndDays[i].date(), progress=False).resample('W-FRI').last()
        #Daily finance datga
        Database_daily = yf.download(asset,start=IndexEndDays[i].date()-timedelta(days=780), end=IndexEndDays[i].date(), progress=False)
        #Database for fitting the models is generated
        Data = M_DatabaseGeneration(Database_daily, Lag, LagSD)
        #Fitting of Transformed ANN-ARCH model, ARCH models and forecasting of the next volatility value
        T_ANN_ARCH_Model = T_ANN_ARCH_Fit (Data,Database_daily, Lag, LagSD, Timestep, Dropout, LearningRate, Epochs)
        
        #Results are collected
        IterResults={'Date_Forecast': T_ANN_ARCH_Model['Date_Forecast'].date(), 'h1': T_ANN_ARCH_Model['Forecast_T_ANN_ARCH'][0], 'h2': T_ANN_ARCH_Model['Forecast_T_ANN_ARCH'][1],
                 'h3': T_ANN_ARCH_Model['Forecast_T_ANN_ARCH'][2], 'h4': T_ANN_ARCH_Model['Forecast_T_ANN_ARCH'][3],'TrueSD':Data['TrueSD'][-1]}

        ResultsCollection.loc[len(ResultsCollection)] = IterResults
        #Results are saved
        ResultsCollection.to_csv(f'./assets/5_MTL_GARCH_{asset_name}.csv',index=False)
        
# Define the indices to process

indices = ['^GSPC']

def main():
    # Create a pool of 4 workers within a context
    with mp.Pool(6) as pool:
        # Start a process for each index
        results = pool.map(process_index, indices)



if __name__ == "__main__":
    main()


