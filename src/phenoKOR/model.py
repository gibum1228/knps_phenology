from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
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
    new_df["ds"] = pd.to_datetime(x, format="%Y-%m-%d")
    new_df["y"] = y
    new_df["index"] = new_df["ds"]
    new_df.set_index("index", inplace=True)

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
    root = '/Users/shjoo/knps_final.csv'
    data = pd.read_csv(root)

    # 북한산의 활엽수를 기준으로 모델링 하겠음.
    # 이후에는 사용자 입력받아서 하는걸로.

    # 데이터 한정짓기

    df = data[(data['code'] == 'bukhan') & (data['class'] == 2)]
    df = df[['date', 'avg']]
    df.columns = ['ds', 'y']
    # 데이터 나누기
    df_train = df[df['ds'] <= '2021-01-01']
    df_test = df[df['ds'] >= '2021-01-01']
    print(df_train)
    print(df_test)

    model = Prophet()
    # model.add_seasonality(period=365,fourier_order=10,mode='additive')
    # model.add_regressor('regressor',mode='additive')
    # model.stan_backend.set_options(newton_fallback=False)
    model.fit(df_train, algorithm='Newton')

    forecast = 365  # forecast만큼 이후를 예측
    df_forecast = model.make_future_dataframe(periods=forecast)  # 예측할 ds 만들기
    df_forecast = model.predict(df_forecast)  # 비어진 ds만큼 예측

    model.plot(df_forecast, xlabel="date", ylabel="evi", figsize=(30, 10))
    model.plot_components(df_forecast)  # 명확한 추세를 발견하였음.
    plt.plot(df_train['y'])
    plt.show()

    forecast_only = df_forecast[df_forecast['ds'] >= '2021-01-01']

    forecast_only_8 = forecast_only[0::8]['yhat']
    print(forecast_only_8)
    print(df_test)


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