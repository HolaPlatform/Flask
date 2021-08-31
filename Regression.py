import os
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')
import pickle
from datetime import timedelta, date
import FinanceDataReader as fdr

import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', None)

import statsmodels.api as sm
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults

from sklearn.model_selection import ParameterGrid

from fbprophet import Prophet

# stock_code = pd.read_csv('../3.프로젝트/Data/KOSPI_200.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]

def load_stocks_data(name, stock_code):
    
    codes_dic = dict(stock_code.values)
    code = codes_dic[name]

    # today = date.today()
    today = date(2021, 7, 8)
    # today = date.today() - timedelta(days=30)
    diff_day = timedelta(days=365)
    print(f'TODAY: {today}')

    start_date = str(today - diff_day)
    finish_date = str(today)
    
    try:
        data = fdr.DataReader(f'{code}', start_date, finish_date)
        # time.sleep(1)
        print(f'Row: {data.shape[0]}\nColumn: {data.shape[1]}')
        
        data = data.reset_index()
        data = data[['Date', 'Close']]
        data.columns = ['ds', 'y']

        return data, code

    except:
        print(f'     LOAD ERROR: {name}     ')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Stocks:

    alpha_dict = {}

    for pkl in os.listdir('./model'):
        if pkl.split('_')[0] == 'prophet':
            k = pkl.split('_')[1] + '_' + pkl.split('_')[2]
            v = float(pkl.split('_')[3].split('.p')[0])
            alpha_dict[k] = v

    params_grid = {'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
                   'seasonality_prior_scale': [5, 10, 15],
                   'yearly_seasonality': [5, 10, 15]
                   }

    grid = ParameterGrid(params_grid)
    cnt = 0

    for p in grid:
        cnt = cnt+1

    model_parameters = pd.DataFrame(columns = ['MAPE',
                                               'changepoint_prior_scale',
                                               'seasonality_prior_scale',
                                               'yearly_seasonality'
                                               ]
                                    )

    def __init__(self, data):
        self.data = data
    
    def modeling(self, code, day, grid=grid, model_parameters=model_parameters):
        
        train_data = self.data.iloc[:-day, :]
        test_data = self.data.iloc[-day:, :]
        test_data = test_data.set_index('ds')

        #ARIMA
        arima_train_data = train_data.copy()
        arima_train_data['y'] = arima_train_data['y'].drop_duplicates()
        arima_train_data = arima_train_data.replace([np.inf, -np.inf], np.nan)
        arima_train_data = arima_train_data.dropna()

        model_arima = auto_arima(arima_train_data['y'].values, trace=False, 
                             error_action='ignore', 
                             start_p=0, start_q=0, max_p=2, max_q=2, 
                             suppress_warnings=True, stepwise=False, seasonal=False)
        model_fit = model_arima.fit(arima_train_data['y'].values)
        
        ORDER = model_fit.order

        X = arima_train_data['y'].values
        X = X.astype('float64')

        model = ARIMA(X, order = ORDER)
        model_fit = model.fit(trend = 'c', full_output = True, disp = 1)

        forecast = model_fit.forecast(day)

        #하락추세
        if forecast[0][0] > forecast[0][-1]:
            value = (forecast[0] + forecast[2].reshape(-1)[1::2]) / 2
        
        #상승추세
        elif forecast[0][0] < forecast[0][-1]:
            value = (forecast[0] + forecast[2].reshape(-1)[0::2]) / 2
        
        #변동없음
        else:
            value = forecast[0]

        #Prophet
        idx = 0
            
        for p in tqdm(grid):
            prophet = Prophet(yearly_seasonality=p['yearly_seasonality'],
                              weekly_seasonality=False,
                              daily_seasonality=False,
                              changepoint_prior_scale=p['changepoint_prior_scale'],
                              seasonality_prior_scale=p['seasonality_prior_scale']
                              )

            prophet.fit(train_data)

            future_data = prophet.make_future_dataframe(periods=day, freq='D')
            forecast_data = prophet.predict(future_data)

            pred_y = forecast_data.yhat.values[-day:]
            MAPE = mean_absolute_percentage_error(test_data.values, pred_y)
            
            if idx == 0:
                idx += 1
                model_parameters = model_parameters.append({'MAPE':MAPE,
                                                            'changepoint_prior_scale':p['changepoint_prior_scale'],
                                                            'seasonality_prior_scale':p['seasonality_prior_scale'],
                                                            'yearly_seasonality':p['yearly_seasonality']
                                                            },ignore_index=True)

                
            else:
                if MAPE < model_parameters['MAPE'].iloc[-1]:
                    model_parameters = model_parameters.append({'MAPE':MAPE,
                                                                'changepoint_prior_scale':p['changepoint_prior_scale'],
                                                                'seasonality_prior_scale':p['seasonality_prior_scale'],
                                                                'yearly_seasonality':p['yearly_seasonality']
                                                                },ignore_index=True)
                else:
                    continue

        parameter = model_parameters.iloc[-1, :]

        prophet = Prophet(yearly_seasonality=parameter['yearly_seasonality'],
                          changepoint_prior_scale=parameter['changepoint_prior_scale'],
                          seasonality_prior_scale=parameter['seasonality_prior_scale'])

        prophet.fit(train_data)

        future_data = prophet.make_future_dataframe(periods=day, freq='D')
        forecast_data = prophet.predict(future_data)

        pred = forecast_data.yhat.values[-day:]

        mape = 10000
        select_alpha = 0

        for alpha in [0.2, 0.4, 0.6, 0.8]:
            mean_pred = (value * (1-alpha)) + (pred * alpha)
            ans = mean_absolute_percentage_error(test_data.values, mean_pred)

            if mape > ans:
                mape = ans
                select_alpha = alpha
            else:
                continue

        #ARIMA Model Save

        arima_data = self.data.copy()
        arima_data['y'] = arima_data['y'].drop_duplicates()
        arima_data = arima_data.replace([np.inf, -np.inf], np.nan)
        arima_data = arima_data.dropna()

        model_arima = auto_arima(arima_data['y'].values, trace=False, 
                             error_action='ignore', 
                             start_p=0, start_q=0, max_p=2, max_q=2, 
                             suppress_warnings=True, stepwise=False, seasonal=False)
        model_fit = model_arima.fit(arima_data['y'].values)
        
        ORDER = model_fit.order

        X = arima_data['y'].values
        X = X.astype('float64')

        model = ARIMA(X, order = ORDER)
        model_fit = model.fit(trend = 'c', full_output = True, disp = 1)

        model_fit.save(f'./model/arima_{code}_{day}.pkl')

        #Prophet Model Save
        parameter = model_parameters.iloc[-1, :]

        prophet = Prophet(yearly_seasonality=parameter['yearly_seasonality'],
                          changepoint_prior_scale=parameter['changepoint_prior_scale'],
                          seasonality_prior_scale=parameter['seasonality_prior_scale'])

        prophet.fit(self.data)

        with open(f'./model/prophet_{code}_{day}_{select_alpha}.pkl', "wb") as f:
            pickle.dump(prophet, f)
        
        return print(f'{code}_{day}_Modeling Finish!!')

    def predict(self, code, day, alpha_dict=alpha_dict):
        
        alpha = float(alpha_dict[f'{code}_{day}'])
        arima = ARIMAResults.load(f'./model/arima_{code}_{day}.pkl')

        forecast = arima.forecast(day)

        #하락추세
        if forecast[0][0] > forecast[0][-1]:
            value = (forecast[0] + forecast[2].reshape(-1)[1::2]) / 2
        
        #상승추세
        elif forecast[0][0] < forecast[0][-1]:
            value = (forecast[0] + forecast[2].reshape(-1)[0::2]) / 2
        
        #변동없음
        else:
            value = forecast[0]

        with open(f'./model/prophet_{code}_{day}_{alpha:.1f}.pkl', 'rb') as f:
            prophet = pickle.load(f)

        future_data = prophet.make_future_dataframe(periods=day, freq='D')
        forecast_data = prophet.predict(future_data)

        pred = forecast_data.yhat.values[-day:]

        #alpha가 높을수록 Prophet의 비중이 높아짐
        mean_pred = (value * (1-alpha)) + (pred * alpha)
        # mean_pred = (value + pred) / 2

        week = day//5

        first_day = future_data.iloc[-day, 0]
        finish_day = future_data.iloc[-day, 0] + timedelta(weeks=week) - timedelta(days=1)
        
        day_range = pd.date_range(first_day, finish_day)
        pred_day = np.array(day_range[day_range.dayofweek < 5].strftime('%Y-%m-%d'))

        # dictonary생성, (key:날짜 value:예측값)
        result_dic = {}
        for i, j in zip(pred_day, mean_pred):
            result_dic[i] = j

        return result_dic
        # return pred_day, mean_pred
        # return mean_pred