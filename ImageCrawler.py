#!/usr/bin/env python
# coding: utf-8

## import 하기
import pymysql
import pandas as pd
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
from urllib.request import urlretrieve # 이미지 경로를 파일로 저장

## 이미지 크롤링 함수
def image_get(num):

    ## MySQL 지정하기
    conn = pymysql.connect(
        host = 'localhost',#ip
        port = 3306,   #방번호
        user = 'root', # user 이름
        password = 'root', # 비밀번호
        db = 'db_test', # db 이름
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor
        )

    ## 커리문 설정하기
    cursor = conn.cursor()
    # sql = 'SELECT NEWS_DATA.URL, NEWS_CLASSIFICATION.IMPACT_SCORE FROM NEWS_DATA\
    #  INNER JOIN NEWS_CLASSIFICATION ON NEWS_DATA.NO = NEWS_CLASSIFICATION.NO\
    #  LEFT JOIN STOCK ON NEWS_CLASSIFICATION.STOCK_INDEX = STOCK.STOCK_INDEX\
    #  WHERE NEWS_DATA.DATETIME BETWEEN "2021-07-07 15:30:00" AND "2021-07-08 08:30:00"\
    #  ORDER BY NEWS_CLASSIFICATION.IMPACT_SCORE DESC limit {0};'.format(num)
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

    ## Url header 설정
    headers = requests.utils.default_headers()

    headers.update(
        {
            'User-Agent': 'My User Agent 1.0',
        }
    )

    fileNo = 0

    for i in range(num):

        try:

            print("{0}번째 파일을 크롤링합니다".format(fileNo))

            url = df['URL'][i].replace('\r', '') # URL뒤에 있는 \r 지우기

            print(url)

            html = requests.get(url, headers=headers) # header포함

            soup = BeautifulSoup(html.text, 'html.parser')

            img = soup.find('span', {'class' : 'end_photo_org'}) # photo 찾기

            img = img.find('img') # img 주소만 찾기

            img = img['src'] # 주소만 남기기

            urlretrieve(img, "C:/Users/Admin/2021_PBL_ML/flask/StocksProphet/StocksProphet/image/"+str(fileNo)+".jpg") # 이미지 저장 경로와 이름 설정

            fileNo += 1

        except: # 이미지가 없거나, 동영상인 경우

            print('--{0}번째 기사에 이미지가 없습니다--'.format(fileNo))

            fileNo += 1

    print('*********크롤링 끝*********')

## 불러온 이미지의 갯수를 설정하기
image_get(4)

