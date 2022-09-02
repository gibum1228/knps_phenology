import json
import warnings
from datetime import date

import numpy as np
import pandas as pd

import preprocessing
from fusioncharts import FusionCharts
from fusioncharts import FusionTable
from fusioncharts import TimeSeries

import statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

warnings.filterwarnings('ignore')

# 전역 변수
ROOT, MIDDLE = preprocessing.get_info()


# 연속된 하나의 그래프를 그려주는 메소드
def show_graph(ori_db: pd.DataFrame, option: int = 2):
    value_name = "EVI" if option < 2 else "Gcc"  # 식생지수 이름

    if option == 0:  # 분석
        # df = preprocessing.load_final_data(ori_db['knps'], ori_db['class_num'])  # 데이터 가져오기
        # df, df_sos = pk.curve_fit(df, ori_db)
        keyword = "analysis"
    elif option == 1:  # 예측
        # df = open_model_processing(ori_db)
        keyword = "predict"

    # 시연용
    if option < 2:
        df = pd.read_csv(f"{ROOT}{MIDDLE}data{MIDDLE}knps_final_{keyword}.csv")
        df = df[(df["code"] == ori_db["knps"]) & (df["class"] == int(ori_db["class_num"])) &
                (df['date'].str[:4] >= ori_db["start_year"]) & (df['date'].str[:4] <= ori_db["end_year"])].sort_values(
            'date')
    else:
        df = pd.read_csv(f"{ROOT}{MIDDLE}data{MIDDLE}jiri011_2019_final.csv")
        ori_db['knps'] = 'jiri'
        ori_db['class_num'] = "sungsamjae"
        ori_db['start_year'], ori_db['end_year'] = "2019", "2019"

    data = []  # 그래프를 그리기 위한 데이터
    schema = [{"name": "Time", "type": "date", "format": "%Y-%m-%d"}, {"name": "Type", "type": "string"},
              {"name": "value", "type": "number"}]  # 하나의 data 구조

    for i in range(len(df)):  # data에 값 채우기
        data.append(
            [df['date'].iloc[i], value_name, df['avg' if option < 2 else 'gcc'].iloc[i]])  # schema 형태에 맞게 데이터 추가

        # 고정형 카메라라면 Rcc 값도 추가
        if option >= 2: data.append([df['date'].iloc[i], "Rcc", df['rcc'].iloc[i]])

    fusionTable = FusionTable(json.dumps(schema), json.dumps(data))  # 데이터 테이블 만들기
    timeSeries = TimeSeries(fusionTable)  # 타임시리즈 만들기

    # 그래프 속성 설정하기
    timeSeries.AddAttribute('caption', f'{{"text":"{value_name} of {ori_db["knps"]}"}}')
    timeSeries.AddAttribute('chart',
                            f'{{"exportEnabled": "1", "exportfilename": "{ori_db["knps"]}_{ori_db["class_num"]}_{ori_db["start_year"]}_{ori_db["end_year"]}"}}')
    timeSeries.AddAttribute('subcaption', f'{{"text":"class_num : {ori_db["class_num"]}"}}')
    timeSeries.AddAttribute('series', '"Type"')  # type으로 값 나누기
    timeSeries.AddAttribute('yaxis', [{"plot": {"value": "value"}, "title": "value"}])

    width = 950 if option < 2 else 680
    height = 350 if option < 2 else 250
    # 그래프 그리기
    fcChart = FusionCharts("timeseries", "ex1", width, height, "chart-1", "json", timeSeries)

    # 그래프 정보 넘기기
    return fcChart.render()


# 선택된 만큼의 여러 개의 그래프를 그려주는 메소드
def show_graphs(ori_db: dict, option: int = 2, df: pd.DataFrame = None):
    value_name = 'EVI' if option < 2 else 'Gcc'

    if option == 0:
        # df = pd.read_csv(ROOT + f"{MIDDLE}data{MIDDLE}knps_final.csv")
        # df = df[df['class'] == int(ori_db['class_num'])]
        # df = df[df['code'] == ori_db['knps']]
        #
        # df, df_sos = pk.curve_fit(df, ori_db)
        keyword = "analysis"
    elif option == 1:
        # df = open_model_processing(ori_db) # curve fitting된 데이터 가져오기
        keyword = "predict"

    # 시연용
    if option < 2:
        df = pd.read_csv(f"{ROOT}{MIDDLE}data{MIDDLE}knps_final_{keyword}.csv")
        df = df[(df["code"] == ori_db["knps"]) & (df["class"] == int(ori_db["class_num"])) &
                (df['date'].str[:4] >= ori_db["start_year"]) & (df['date'].str[:4] <= ori_db["end_year"])].sort_values(
            'date')
    else:
        df = pd.read_csv(f"{ROOT}{MIDDLE}data{MIDDLE}jiri011_2019_final.csv")
        ori_db['knps'] = 'jiri'
        ori_db['class_num'] = "sungsamjae"
        ori_db['start_year'], ori_db['end_year'] = "2019", "2019"

    # 그래프 속성 및 데이터를 저장하는 변수
    db = {
        "chart": {  # 그래프 속성
            "exportEnabled": "1",
            "exportfilename": f"{ori_db['knps']}_{ori_db['class_num']}_{ori_db['start_year']}_{ori_db['end_year']}",
            "bgColor": "#FFFFFF",
            "showBorder": "0",
            "showvalues": "0",
            "numvisibleplot": "12",
            "caption": f"{value_name} of {ori_db['knps']}",
            "subcaption": f"class_num : {ori_db['class_num']}",
            "yaxisname": f"{value_name}",
            "drawAnchors": "0",
            "plottooltext": f"<b>$dataValue</b> {value_name} of $label",
        },
        "categories": [{  # X축
            "category": [{"label": str(i)} for i in range(1, 365, 8 if option == 0 else 1)]
            # 분석은 8-day, 예측과 고정형 카메라는 1-day
        }],
        "dataset": []  # Y축
    }

    # 데이터셋에 데이터 넣기
    for now in range(int(ori_db['start_year']), int(ori_db['end_year']) + 1):  # start_year에서 end_year까지
        if option < 2:
            db["dataset"].append({
                "seriesname": f"{now} EVI",  # 레이블 이름
                # 해당 연도에 시작 (1월 1일)부터 (12월 31)일까지의 식생지수 값을 넣기
                "data": [{"value": i} for i in
                         df[df['date'].str[:4] == str(now)]['avg' if option < 2 else 'gcc']]
            })
        else:
            for key in ["gcc", "rcc"]:
                db["dataset"].append({
                    "seriesname": f"{now} {key}",  # 레이블 이름
                    # 해당 연도에 시작 (1월 1일)부터 (12월 31)일까지의 식생지수 값을 넣기
                    "data": [{"value": i} for i in
                             replace_blank(df[df['date'].str[:4] == str(now)], key)]
                })

    width = 950 if option < 2 else 680
    height = 350 if option < 2 else 250
    # 그래프 그리기
    fcChart = FusionCharts('scrollline2d', 'ex1', width, height, 'chart-1', 'json', json.dumps(db))

    return fcChart.render()  # 그래프 정보 넘기기


# 고정형 카메라 정보에서 여러 개의 그래프를 그리기 위한 결측치를 None으로 채우기
def replace_blank(df: pd.DataFrame, key: str) -> list:
    replace_value_list = []  # 365개의 식생지수를 리턴할 리스트
    index, last_date = 0, date(int(df['date'].iloc[0][:4]), 1, 1)

    for i in range(len(df)):  # data에 값 채우기
        focus_date = date.fromisoformat(df['date'].iloc[i])

        # focus와 last 사이에 간격이 있을 경우 결측치가 있는 것이기 때문에 None으로 대체
        for j in range((focus_date - last_date).days - 1):
            replace_value_list.append("None")

        # focus 값을 넣고 last를 업데이트
        replace_value_list.append(df[key].iloc[i])
        last_date = focus_date

    # 12월 31일에 끝나지 않는다면, 미래 결측치를 None으로 채우기
    for i in range((date(int(df['date'].iloc[0][:4]), 12, 31) - last_date).days):
        replace_value_list.append("None")

    return replace_value_list


# 윤년 구하는 메소드
def get_Feb_day(year: int) -> int:
    # 4, 100, 400으로 나누어 떨어진다면 윤년
    if year % 4 == 0 or year % 100 == 0 or year % 400 == 0:
        day = 29
    else:
        day = 28

    return day


# ADF test : 시계열에 단위근이 존재하는지의 여부 검정함으로써 정상 시계열인지 여부 판단함
# H0 : 단위근이 존재한다. 즉, 정상 시계열이 아니다.
# H1 : 단위근이 없다. 즉, 정상 시계열이다.
def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",  # 검정통계량
            "p-value",  # p-value
            "#Lags Used",  # 가능한 시차
            "Number of Observations Used",  # 관측 가능 수
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value  # 검정통계량 기준치
    print(dfoutput)


# kpss 검정 (Kwiatkowski-Phillips-Schmidt-shin test) : 시계열이 정상성인지 판정하는 방법
# H0 : 정상 시계열이다.
# H1 : 정상 시계열이 아니다.
def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic",  # 검정통계량
                              "p-value",  # p-value
                              "Lags Used"]  # 가능한 시차
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value  # 검정통계량 기준치
    print(kpss_output)
