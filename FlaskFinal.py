## import 하기
import pymysql
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
from urllib.request import urlretrieve # 이미지 경로를 파일로 저장
import time
from flask import Flask, request, jsonify, render_template, url_for, redirect
import pymysql
import pandas as pd
import Classifier as cls
import Regression as st
import joblib
from datetime import timedelta

## DB 연결하기
conn = pymysql.connect(
    host='localhost',  # ip
    port=3306,  # 방번호
    user='root',  # user 이름
    password='root',  # 비밀번호
    db='db_test',  # db 이름
    autocommit=True,
    cursorclass=pymysql.cursors.DictCursor
    )

## hola_pick 커리문 설정하기
cursor = conn.cursor()
sql = 'SELECT NEWS_DATA.TITLE, NEWS_DATA.ARTICLE, NEWS_DATA.URL, NEWS_CLASSIFICATION.SENTIMENT_LABEL,\
    NEWS_CLASSIFICATION.IMPACT_SCORE, STOCK.STOCK_NAME, STOCK.WICS_s FROM NEWS_DATA\
    INNER JOIN NEWS_CLASSIFICATION ON NEWS_DATA.NO = NEWS_CLASSIFICATION.NO\
    LEFT JOIN STOCK ON NEWS_CLASSIFICATION.STOCK_INDEX = STOCK.STOCK_INDEX\
    WHERE NEWS_DATA.DATETIME BETWEEN "2021-07-07 15:30:00" AND "2021-07-08 08:30:00"\
    ORDER BY NEWS_CLASSIFICATION.IMPACT_SCORE DESC LIMIT 13, 4;'
cursor.execute(sql)
result = cursor.fetchall()

## pandas dataframe화 하기
df = pd.DataFrame(result)


### 주식 예측값 함수
def hola_sec_function(name='삼성전자', day=20):

    # 종목명, 종목코드 불러오기
    stock_code = pd.read_csv('./KOSPI_100.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]
    data_st, code_st = st.load_stocks_data(f'{name}', stock_code) # 종목명, 종목코드로 주가데이터 로드
    stocks_st = st.Stocks(data_st)                # stocks 객체 생성
    result_st = stocks_st.predict(code_st, int(day)) # 예측값 딕셔너리 형태로 반환
    
    data_cls, code_cls = cls.load_stocks_data(f'{name}', stock_code) # 종목명, 종목코드로 주가데이터 로드
    stocks_cls = cls.Stocks(data_cls)                # stocks 객체 생성
    stocks_cls.preprocessing()
    sign_data = stocks_cls.stocksign(stocks_cls.data, int(day))                                  # stocks 객체안의 데이터 보조지표 생성 및 전처리
    result_cls = stocks_cls.predict(sign_data, code_cls, int(day))                               # 예측값 딕셔너리 형태로 반환                            

    return result_st, result_cls


### Flask 실행하기
# statick_folder 를 지정했습니다.
app = Flask(__name__, static_url_path="", static_folder="static")
app.config['JSON_AS_ASCII'] = False


### 워드클라우드 페이지
@app.route('/word_cloud')

def word_cloud(): 
    return render_template('index_main.html',image_file_pos="2021-07-08_pos.png", image_file_neg = '2021-07-08_neg.png')
   

### Hola_pick 페이지
@app.route('/hola_pick')

def test():
    ## 뉴스의 제목과 기사를 가져오기
    title_list = []
    article_list = []

    ## 이미 df는 영향력순으로 top4가 나왔으니 for문으로 제목과 기사 리스트 만들어주기
    for i in range(4):
        title_list.append(df['TITLE'][i])
        a = df['ARTICLE'][i]
        # 80자를 기준으로 자르기
        article_list.append(a[:81] + '...')

    # 각 기사에 맞는 이미지 지정해주기
    return render_template('index_news.html', article_list = article_list, title_list = title_list, image_file0="0.jpg", image_file1="1.jpg", image_file2="2.jpg", image_file3='3.jpg')

### News 페이지
@app.route('/news')

def news():
    ## Hola_pick에서 기사를 눌러 index값 받아오기
    index = request.args.get('index', type = int)
    
    # html 리턴값 설정하기
    file_name = str(index) + '.jpg'
    title = df['TITLE'][index]
    a = df['ARTICLE'][index]
    article = a[:251] + '...'
    url = df['URL'][index]
    industry = df['WICS_s'][index]
    sentiment = df['SENTIMENT_LABEL'][index]
    impact = df['IMPACT_SCORE'][index]
    stock = df['STOCK_NAME'][index]

    # html 리턴값들과 이미지 설정하기
    return render_template('index_news_detail.html', file_name = file_name, title = title, article = article, url = url,
    industry = industry, sentiment = int(sentiment), impact = impact, stock = stock, up = "up.png", down = "down.png")

### Stock 페이지
@app.route('/stock')

def stock():
    # 해시태그를 클릭하면 index 받아오기(현재의 문제점은 ""안에 {{변수}}를 집어넣는 것)
    # Input값, /?company=기업명&day=예측기간
    input_company = request.args.get("company") # company(기업명): ex) 삼성전자, SK하이닉스, LG화학, 카카오, NAVER, 현대차
    input_day = request.args.get("day")         # day(예측기간): ex) 5, 20, 60, 120

    msg_st, msg_cls = hola_sec_function(input_company, input_day)

    keys_st = []
    values_st = []
    for key, value in msg_st.items():
        keys_st.append(key)
        values_st.append(np.round(value,0))

    keys_cls = []
    values_cls = []
    for key, value in msg_cls.items():
        keys_cls.append(key)
        values_cls.append(value)
    print(values_st)
    print(values_cls)
    

    ## top7 커리문 설정하기
    cursor = conn.cursor()
    sql = 'SELECT NEWS_DATA.TITLE, STOCK.STOCK_NAME, NEWS_CLASSIFICATION.IMPACT_SCORE FROM NEWS_DATA\
        INNER JOIN NEWS_CLASSIFICATION ON NEWS_DATA.NO = NEWS_CLASSIFICATION.NO\
        LEFT JOIN STOCK ON NEWS_CLASSIFICATION.STOCK_INDEX = STOCK.STOCK_INDEX\
        WHERE STOCK.STOCK_NAME = "{0}" AND (NEWS_DATA.DATETIME BETWEEN "2021-07-07 15:30:00" AND "2021-07-08 08:30:00") \
        ORDER BY NEWS_CLASSIFICATION.IMPACT_SCORE DESC limit 7;'.format(input_company)
    cursor.execute(sql)
    result = cursor.fetchall()

    ## pandas dataframe화 하기
    df = pd.DataFrame(result)

    ## top10 뉴스 출력
    title_list = []
    for i in range(len(df)):
        title_list.append(df['TITLE'][i])
    if len(title_list) == 0: # 예외 처리
        title_list.append('관련 뉴스가 없습니다.')


    ## 신호등 커리문 설정하기
    cursor = conn.cursor()
    sql2 = 'SELECT COUNT(*) FROM NEWS_DATA \
        INNER JOIN NEWS_CLASSIFICATION ON NEWS_DATA.NO = NEWS_CLASSIFICATION.NO \
        LEFT JOIN STOCK ON NEWS_CLASSIFICATION.STOCK_INDEX = STOCK.STOCK_INDEX\
        WHERE STOCK.STOCK_NAME = "{0}" AND (NEWS_DATA.DATETIME BETWEEN "2021-07-07 15:30:00" AND "2021-07-08 08:30:00");'.format(input_company)
    cursor.execute(sql2)
    result2 = cursor.fetchall()

    ## pandas dataframe화 하기
    df2 = pd.DataFrame(result2)

    ## 초록불 커리문 설정하기
    cursor = conn.cursor()
    sql3 = 'SELECT COUNT(*) FROM NEWS_DATA \
        INNER JOIN NEWS_CLASSIFICATION ON NEWS_DATA.NO = NEWS_CLASSIFICATION.NO \
        LEFT JOIN STOCK ON NEWS_CLASSIFICATION.STOCK_INDEX = STOCK.STOCK_INDEX\
        WHERE NEWS_CLASSIFICATION.SENTIMENT_LABEL = 1 AND \
        (NEWS_DATA.DATETIME BETWEEN "2021-07-07 15:30:00" AND "2021-07-08 08:30:00")\
        AND STOCK.STOCK_NAME = "{0}";'.format(input_company)
    cursor.execute(sql3)
    result3 = cursor.fetchall()

    ## pandas dataframe화 하기
    df3 = pd.DataFrame(result3)

    ## 빨간불 커리문 설정하기
    cursor = conn.cursor()
    sql4 = 'SELECT COUNT(*) FROM NEWS_DATA \
        INNER JOIN NEWS_CLASSIFICATION ON NEWS_DATA.NO = NEWS_CLASSIFICATION.NO \
        LEFT JOIN STOCK ON NEWS_CLASSIFICATION.STOCK_INDEX = STOCK.STOCK_INDEX\
        WHERE NEWS_CLASSIFICATION.SENTIMENT_LABEL = 0 AND \
        (NEWS_DATA.DATETIME BETWEEN "2021-07-07 15:30:00" AND "2021-07-08 08:30:00")\
        AND STOCK.STOCK_NAME = "{0}";'.format(input_company)
    cursor.execute(sql4)
    result4 = cursor.fetchall()

    ## pandas dataframe화 하기
    df4 = pd.DataFrame(result4)

    if df2['COUNT(*)'][0] * 0.7 <= df3['COUNT(*)'][0]:
        light = 'green.png' # 초록불
    elif df2['COUNT(*)'][0] * 0.7 <= df4['COUNT(*)'][0]:
        light = 'red.png' # 빨간불
    else:
        light = 'yello.png' # 노란불
        


    return render_template('index_stocks.html', title_list = title_list, com = input_company, da = input_day, 
                            light = light, keys = keys_st, values = values_st, values_cls = values_cls)

@app.route('/stocks', methods = ['POST', 'GET'])

def stocks():
    company = request.form['company']
    day = request.form['day']

    return redirect(url_for('stock', company='%s' % company, day='%s' % day))

@app.route('/architecture')

def architecture():
    return render_template('index_architecture.html')

if __name__ == '__main__':
    # app.run()
        app.run(port=5500)