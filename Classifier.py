import time
from tqdm import tqdm
from datetime import timedelta, date
import warnings
warnings.filterwarnings('ignore')
import joblib
import FinanceDataReader as fdr

import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 주가데이터 로드, (name='종목명', stock_code=종목코드와 종목명이 들어있는 데이터프레임)
def load_stocks_data(name, stock_code):
    
    codes_dic = dict(stock_code.values)
    code = codes_dic[name]

    # today = date.today()
    today = date(2021, 7, 8)
    # today = date.today() - timedelta(days=30)
    diff_day = timedelta(days=10000)
    print(f'TODAY: {today}')

    start_date = str(today - diff_day)
    finish_date = str(today)
    
    try:
        data = fdr.DataReader(f'{code}', start_date, finish_date)
        # time.sleep(1)
        print(data.shape)
        return data, code
    
    except:
        print(f'     LOAD ERROR: {name}     ')
    
# 주가데이터 객체 생성
class Stocks:

    columns = ['MACD', 'MACD_Signal', 'MACD_Oscillator', 'RSI', 'RSI_Signal', 
               'kdj_k', 'kdj_d', 'kdj_j', 'CCI', 'PDI', 'MDI', 'ADX', 'OBV', 'OBV_EMA']

    sign_columns = ['MACD_sign', 'MACD_Oscillator_sign', 'MACD_diff_sign', 
                   'RSI_sign', 'Stochastic_K_sign', 'Stochastic_KD_sign', 
                   'CCI_sign', 'CCI_shift_sign', 'DMI_sign', 'OBV_sign', 'target']

    target_dic = {5: 'ma5', 20:'ma20', 60:'ma60', 120:'ma120'}

    # data='주가데이터', code='종목코드', day='예측기간 ex)5, 20, 60, 120'
    def __init__(self, data):
        self.data = data

    # 모델링에 활용할 보조지표 생성 및 전처리
    def preprocessing(self):
        #MA
        self.data['ma5'] = self.data['Close'].rolling(window=5).mean()
        self.data['ma60'] = self.data['Close'].rolling(window=60).mean()
        self.data['ma20'] = self.data['Close'].rolling(window=20).mean()
        self.data['ma120'] = self.data['Close'].rolling(window=120).mean()
        
        #MACD
        ShortEMA = self.data['Close'].ewm(span=12, adjust=False).mean()
        LongEMA = self.data['Close'].ewm(span=26, adjust=False).mean()

        self.data['MACD'] = ShortEMA - LongEMA
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['MACD_Oscillator'] = self.data['MACD'] - self.data['MACD_Signal']
        
        #RSI
        delta = self.data['Close'].diff(1)
        delta = delta[1:]
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        
        self.data['up'] = up
        self.data['down'] = down
        
        AVG_Gain = self.data['up'].rolling(window=14).mean()
        AVG_Loss = abs(self.data['down'].rolling(window=14).mean())
        RS = AVG_Gain / AVG_Loss
        
        RSI = 100.0 - (100.0 / (1.0 + RS))
        self.data['RSI'] = RSI
        self.data['RSI_Signal'] = self.data['RSI'].rolling(window=9).mean()

        #Stochastic
        ndays_high = self.data['High'].rolling(window=10, min_periods=1).max()
        ndays_low = self.data['Low'].rolling(window=10, min_periods=1).min()
    
        self.data['kdj_k'] = ((self.data.Close - ndays_low) / (ndays_high - ndays_low))*100
        self.data['kdj_d'] = self.data['kdj_k'].ewm(span=5, adjust=False).mean()
        self.data['kdj_j'] = self.data['kdj_d'].ewm(span=5, adjust=False).mean()

        #CCI
        M = (self.data['High'] + self.data['Low'] + self.data['Close']) /3
        m = M.rolling(window=20).mean()
        d = abs(M-m).rolling(window=20).mean()
        self.data['CCI'] = (M-m) / (d * 0.015)

        #DMI
        self.data = self.data.reset_index()

        i = 0 
        UpI = [0] 
        DoI = [0] 
        while i + 1 <= self.data.index[-1]: 
            UpMove = self.data.loc[i + 1, "High"] - self.data.loc[i, "High"] 
            DoMove = self.data.loc[i, "Low"] - self.data.loc[i+1, "Low"] 
            if UpMove > DoMove and UpMove > 0: 
                UpD = UpMove 
            else: 
                UpD = 0 
            UpI.append(UpD) 
            if DoMove > UpMove and DoMove > 0: 
                DoD = DoMove 
            else: 
                DoD = 0 
            DoI.append(DoD) 
            i = i + 1 
        
        i = 0 
        TR_l = [0] 
        while i < self.data.index[-1]: 
            TR = max(self.data.loc[i + 1, 'High'], self.data.loc[i, 'Close']) - min(self.data.loc[i + 1, 'Low'], self.data.loc[i, 'Close']) 
            TR_l.append(TR) 
            i = i + 1
            
        TR_s = pd.Series(TR_l)
        ATR = pd.Series(TR_s.ewm(span=14, min_periods=14).mean())
        UpI = pd.Series(UpI)
        DoI = pd.Series(DoI)
        PosDI = pd.Series(UpI.ewm(span=14, min_periods=14).mean() / ATR)
        NegDI = pd.Series(DoI.ewm(span=14, min_periods=14).mean() / ATR)
        ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span=14, min_periods=14).mean())
        
        self.data['PDI'] = PosDI.values
        self.data['MDI'] = NegDI.values
        self.data['ADX'] = ADX.values
        
        self.data.index = self.data.Date
        self.data = self.data.drop('Date', axis=1)

        #OBV
        OBV = []
        OBV.append(0)
        
        for i in range(1, len(self.data.Close)):
            if self.data.Close[i] > self.data.Close[i-1]:
                OBV.append(OBV[-1] + self.data.Volume[i])
            elif self.data.Close[i] < self.data.Close[i-1]:
                OBV.append(OBV[-1] - self.data.Volume[i])
            else:
                OBV.append(OBV[-1])
        
        self.data['OBV'] = OBV
        self.data['OBV_EMA'] = self.data['OBV'].ewm(span=20, adjust=False).mean()

        self.data = self.data.dropna().reset_index()
        return print('Preprocessing Done!!')

    def stocksign(self, data, day, columns=columns, sign_columns=sign_columns, target_dic=target_dic):
        
        target = target_dic[day]
        # target = 'Close'

        for col in columns:
            data[f'{col}'] = data[f'{col}'].ewm(span=day, adjust=False).mean()

        # 매수: 1, 매도: -1, 중립: 0
        # 양수: 1, 음수: 0
        
        data['MACD_sign'] = data['MACD'].apply(lambda x: 1 if x >0 else -1)
        data['MACD_Oscillator_pm'] = data['MACD_Oscillator'].apply(lambda x: 1 if x > 0 else 0)
        data['MACD_Oscillator_sign'] = data['MACD_Oscillator_pm'] - data['MACD_Oscillator_pm'].shift(1)
        data['MACD_diff'] = (data['MACD'] - data['MACD_Signal']).apply(lambda x: 1 if x > 0 else 0)
        data['MACD_diff_sign'] = data['MACD_diff'] - data['MACD_diff'].shift(1)

        data['RSI_sign'] = data['RSI'].apply(lambda x: 1 if x >= 70 else (-1 if x <= 30 else 0))

        data['kdj_d_pm'] = data['kdj_d'].apply(lambda x: 1 if x >= 80 else(-1 if x <= 20 else 0))
        data['kdj_d_pm_shift'] = data['kdj_d_pm'] - data['kdj_d_pm'].shift(1)
        data.loc[(data['kdj_d'] < 80) & (data['kdj_d'] > 20) & (data['kdj_d_pm_shift'] == 1), 'Stochastic_K_sign'] =  1
        data.loc[(data['kdj_d'] < 80) & (data['kdj_d'] > 20) & (data['kdj_d_pm_shift'] == -1), 'Stochastic_K_sign'] =  -1
        data['Stochastic_K_sign'] = data['Stochastic_K_sign'].fillna(0)

        data['kdj_pm'] = (data['kdj_d'] - data['kdj_j']).apply(lambda x: 1 if x > 0 else 0)
        data['Stochastic_KD_sign'] = data['kdj_pm'] - data['kdj_pm'].shift(1)

        data['CCI_pm'] = data['CCI'].apply(lambda x: 1 if x > 0 else 0)
        data['CCI_sign'] = data['CCI_pm'] - data['CCI_pm'].shift(1)
        data['CCI_range'] = data['CCI'].apply(lambda x: -1 if x < -100 else (1 if x > 100 else 0))
        data['CCI_shift'] = data['CCI'] - data['CCI'].shift(1)

        data.loc[(data['CCI_range'] == -1) & (data['CCI_shift'] > 0), 'CCI_shift_sign'] = 1
        data.loc[(data['CCI_range'] == 1) & (data['CCI_shift'] < 0), 'CCI_shift_sign'] = -1
        data['CCI_shift_sign'] = data.fillna(0)['CCI_shift_sign']
        data['ADX_shift'] = (data['ADX'] - data['ADX'].shift(1)).rolling(5).mean()

        data.loc[(data['PDI'] > data['MDI']) & (data['ADX'] > data['MDI']) & (data['ADX_shift'] > 0.01), 'DMI_sign'] = 1
        data.loc[(data['MDI'] > data['PDI']) & (data['ADX'] > data['PDI']) & (data['ADX_shift'] > 0.01), 'DMI_sign'] = -1
        data['DMI_sign'] = data['DMI_sign'].fillna(0)

        data['OBV_diff'] = (data['OBV'] - data['OBV_EMA']).apply(lambda x: 1 if x > 0 else 0)
        data['OBV_sign'] = data['OBV_diff'] - data['OBV_diff'].shift(1)

        data['target'] = (data[target].shift(-day) - data[target]).apply(lambda x: 1 if x > 0 else 0)
        
        data = data.dropna().set_index('Date')

        for c in sign_columns:
            data[f'{c}'] = data[f'{c}'].astype('int')

        print('StockSign Done!!')
        return data

    # 모델링, financedata='모델링 시행할 처리 완료된 데이터'
    def modeling(self, data, code, day):

        data = data.iloc[:-day, :]

        X = data.drop(['High', 'Low', 'Open', 'Close', 'Volume', 'Change', 'up', 'down',
                        'ma5', 'ma20', 'ma60', 'ma120', 'target'], axis=1)
        y = data[['target']]
        
        # 최대한 최근의 데이터를 학습하기 위해 뒷부분의 0.05%만 test 데이터로 활용
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=False)
        
        # 스케일링 적용하여 모든 변수 수치 정규화
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        params_grid = {'n_estimators' : [1000], 
                       'eta' : [0.01, 0.1, 0.3], 
                       'min_child_weight' : [1, 2, 4, 8], 
                       'max_depth' : [3, 5, 7, 9] 
                        }

        grid = ParameterGrid(params_grid)

        model_parameters = pd.DataFrame(columns = ['ACC',
                                                   'n_estimators', 
                                                   'eta',
                                                   'min_child_weight',
                                                   'max_depth']
                                                   )

        idx = 0

        for p in grid:
            model = XGBClassifier(n_estimators=p['n_estimators'],
                                  learning_rate=p['eta'],
                                  min_child_weight=p['min_child_weight'],
                                  max_depth=p['max_depth']
                                  )

            evals = [(X_val, y_val)]

            model.fit(X_train, y_train, early_stopping_rounds=50, eval_metric='logloss', eval_set=evals, verbose=0)
            pred = model.predict(X_test)

            acc = accuracy_score(pred, y_test)
            # print(f'Accuracy: {acc}')

            if idx == 0:
                idx += 1
                model_parameters = model_parameters.append({'ACC':acc,
                                                            'n_estimators':p['n_estimators'],
                                                            'eta':p['eta'],
                                                            'min_child_weight':p['min_child_weight'],
                                                            'max_depth':p['max_depth']
                                                            },ignore_index=True)
                
            else:
                if acc > model_parameters['ACC'].iloc[-1]:
                    model_parameters = model_parameters.append({'ACC':acc,
                                                                'n_estimators':p['n_estimators'],
                                                                'eta':p['eta'],
                                                                'min_child_weight':p['min_child_weight'],
                                                                'max_depth':p['max_depth']
                                                                },ignore_index=True)
                else:
                    continue
        
                    
        parameter = model_parameters.iloc[-1, :]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, shuffle=False)
        
        # 스케일링 적용하여 모든 변수 수치 정규화
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        evals = [(X_val, y_val)]

        model = XGBClassifier(n_estimators=int(parameter['n_estimators']),
                              learning_rate=parameter['eta'],
                              min_child_weight=int(parameter['min_child_weight']),
                              max_depth=int(parameter['max_depth'])
                              )

        model.fit(X_train, y_train, early_stopping_rounds=50, eval_metric='logloss', eval_set=evals, verbose=0)

        # 지정경로에 생성한 모델과 스케일러 저장
        model_file_name = f'./model/xgb_model_{code}_{str(day)}.pkl' 
        scaler_file_name = f'./model/scaler_{code}_{str(day)}.pkl'

        joblib.dump(model, model_file_name)
        joblib.dump(scaler, scaler_file_name)

        return parameter, print(f'{code}_{day}_Modeling Finish!!')


    def modelings(self, data, code, day):
        
        data = data.iloc[:-day, :]

        X = data.drop(['High', 'Low', 'Open', 'Close', 'Volume', 'Change', 'up', 'down',
                        'ma5', 'ma20', 'ma60', 'ma120', 'target'], axis=1)
        y = data[['target']]
        
        # 최대한 최근의 데이터를 학습하기 위해 뒷부분의 0.05%만 test 데이터로 활용
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=False)
        
        # 스케일링 적용하여 모든 변수 수치 정규화
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        #RandomForest
        params_grid = {'n_estimators': [100, 300, 500],
                        'max_depth': [5, 15, 25],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 5, 10]}
        
        grid = ParameterGrid(params_grid)

        model_parameters = pd.DataFrame(columns = ['ACC',
                                                    'n_estimators', 
                                                    'max_depth',
                                                    'min_samples_split',
                                                    'min_samples_leaf']
                                                    )

        idx = 0

        for p in grid:
            model = RandomForestClassifier(n_estimators=p['n_estimators'],
                                            max_depth=p['max_depth'],
                                            min_samples_split=p['min_samples_split'],
                                            min_samples_leaf=p['min_samples_leaf']
                                            )

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(pred, y_test)
            # print(f'Accuracy: {acc}')

            if idx == 0:
                idx += 1
                model_parameters = model_parameters.append({'ACC':acc,
                                                            'n_estimators':p['n_estimators'],
                                                            'max_depth':p['max_depth'],
                                                            'min_samples_split':p['min_samples_split'],
                                                            'min_samples_leaf':p['min_samples_leaf']
                                                            },ignore_index=True)

            else:
                if acc > model_parameters['ACC'].iloc[-1]:
                    model_parameters = model_parameters.append({'ACC':acc,
                                                                'n_estimators':p['n_estimators'],
                                                                'max_depth':p['max_depth'],
                                                                'min_samples_split':p['min_samples_split'],
                                                                'min_samples_leaf':p['min_samples_leaf']
                                                                },ignore_index=True)
                else:
                    continue

        parameter = model_parameters.iloc[-1, :]
    
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, shuffle=False)
        
        # 스케일링 적용하여 모든 변수 수치 정규화
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)

        model = RandomForestClassifier(n_estimators=int(p['n_estimators']),
                                       max_depth=p['max_depth'],
                                       min_samples_split=p['min_samples_split'],
                                       min_samples_leaf=p['min_samples_leaf']
                                       )

        model.fit(X_train, y_train)

        # 지정경로에 생성한 모델과 스케일러 저장
        model_file_name = f'./model/rf_model_{code}_{str(day)}.pkl' 
        scaler_file_name = f'./model/scaler_{code}_{str(day)}.pkl'

        joblib.dump(model, model_file_name)
        joblib.dump(scaler, scaler_file_name)

        return parameter, print(f'{code}_{day}_Modeling Finish!!')

    # 예측값 반환, path='모델, 스케일러 경로'
    # 예측값 딕셔너리 형태로 반환
    def predict(self, data, code, day):
        xgb_model = joblib.load(f'./model/xgb_model_{code}_{str(day)}.pkl')
        rf_model = joblib.load(f'./model/rf_model_{code}_{str(day)}.pkl')
        scaler = joblib.load(f'./model/scaler_{code}_{str(day)}.pkl')

        data = data.drop(['High', 'Low', 'Open', 'Close', 'Volume', 'Change', 'up', 'down',
                          'ma5', 'ma20', 'ma60', 'ma120', 'target'], axis=1)

        # test데이터의 뒷부분을 예측데이터로 활용
        test_data = data.iloc[-day:, :]
        
        test = scaler.transform(test_data)
        stock_data = test_data.reset_index()
        
        week = day//5

        first_day = stock_data.iloc[-1, 0] + timedelta(days=1)
        finish_day = stock_data.iloc[-1, 0] + timedelta(days=week*7)
        # print(week)
        
        day_range = pd.date_range(first_day, finish_day)
        pred_day = np.array(day_range[day_range.dayofweek < 5].strftime('%Y-%m-%d'))
        
        xgb_prob = xgb_model.predict_proba(test)
        rf_prob = rf_model.predict_proba(test)

        prob = (xgb_prob * 0.8) + (rf_prob * 0.2)
        pred = np.argmax(prob, axis=1)

        # dictonary생성, (key:날짜 value:예측값)
        pred_dic = {}
        for i, j in zip(pred_day, pred):
            pred_dic[i] = j

        prob_dic = {}
        for i, j in zip(pred_day, prob):
            prob_dic[i] = j

        # 시각화를 위한 강도 생성
        # alpha_lst = []
        # alpha = 0
        # for i in range(len(pred)):
        #     if pred[i] == 1:
        #         alpha += 1
        #         alpha_lst.append(alpha)
        #     else:
        #         alpha = 0
        #         alpha_lst.append(alpha)

        return pred_dic
        # return pred_dic, prob_dic
        # return result_dic, alpha_lst


    def predicts(self, data, code, day, alpha):
        xgb_model = joblib.load(f'./model/xgb_model_{code}_{str(day)}.pkl')
        rf_model = joblib.load(f'./model/rf_model_{code}_{str(day)}.pkl')
        scaler = joblib.load(f'./model/scaler_{code}_{str(day)}.pkl')

        data = data.iloc[:-day, :]

        X = data.drop(['High', 'Low', 'Open', 'Close', 'Volume', 'Change', 'up', 'down',
                        'ma5', 'ma20', 'ma60', 'ma120', 'target'], axis=1)
        y = data[['target']]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
        X_test = scaler.transform(X_test)
        
        xgb_pred = xgb_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)

        xgb_prob = xgb_model.predict_proba(X_test)
        rf_prob = rf_model.predict_proba(X_test)

        prob = (xgb_prob * (1-alpha)) + (rf_prob * alpha)
        pred = np.argmax(prob, axis=1)
        
        xgb_acc = accuracy_score(xgb_pred, y_test)
        rf_acc = accuracy_score(rf_pred, y_test)
        acc = accuracy_score(pred, y_test)

        return xgb_acc, rf_acc, acc
