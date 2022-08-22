from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
from prophet import Prophet
import pandas as pd
import numpy as np


'''데이터를 로드하는 메소드'''
def load_data():
    df = pd.read_csv("/Users/beom/Desktop/git/data/knps/dl_data/bukhan/bukhan_2.csv")  # curve fitting된 데이터 가져오기
    x, y = [], []  # X: {data}, Y: {EVI}
    info_day = [None, 31, None, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 월별 일 수 정보
    year, month, day = 2003, 1, 1  # 년월일을 알기 위한 변수

    for i in range(len(df)):  # data에 값 채우기
        # 데이터 추가
        x.append(f"{year}-{month}-{day}")
        y.append(df.iloc[i, 1])

        day += 8  # 8일 간격씩 데이터 추가
        if month == 2:  # 2월은 윤년 여부 판단하기
            day_limit = get_Feb_day(year)
        else:  # 2월이 아니면 해당 월의 일 수 가져오기
            day_limit = info_day[month]

        # 다음 월로 넘어가야 한다면,
        if day > day_limit:
            day -= day_limit  # 새로운 일
            month += 1  # 다음 월로 가기

            if month > 12:  # 다음 연도로 넘어가야 한다면,
                year += 1
                # 무조건 1월 1일부터 시작하기 때문에 month와 day를 1로 초기화
                month = 1
                day = 1

    new_df = pd.DataFrame()
    new_df["date"] = pd.to_datetime(x, format="%Y-%m-%d")
    new_df["y"] = y

    return new_df


'''윤년 구하는 메소드'''
def get_Feb_day(year):
    # 4, 100, 400으로 나누어 떨어진다면 윤년
    if year % 4 == 0 or year % 100 == 0 or year % 400 == 0:
        day = 29
    else:
        day = 28

    return day


'''해당하는 모델로 학습하는 메소드'''
def fit_arima():
    pass


def fit_prophet():
    df = load_data()

    print(df)


def fit_random_forest():
    pass


def fit_LSTM():
    pass


'''모델 검증하는 메소드'''
# MSE
def MSE(y, pred_y):
    return mean_squared_error(y, pred_y)


# RMSE
def RMSE(y, pred_y):
    return MSE(y, pred_y) ** 0.5


# R^2
def R2(y, pred_y):
    return r2_score(y, pred_y)