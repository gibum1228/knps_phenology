import json
import warnings
from datetime import date

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

import preprocessing
from fusioncharts import FusionCharts
from fusioncharts import FusionTable
from fusioncharts import TimeSeries

warnings.filterwarnings('ignore')

# 전역 변수
ROOT, MIDDLE = preprocessing.get_info()


# 연속된 하나의 그래프를 그려주는 메소드
def show_graph(ori_db: dict, option: int = 2, df: pd.DataFrame = None):
    value_name = "EVI" if option < 2 else "Gcc"  # 식생지수 이름

    # 시연용
    keyword = "analysis" if option == 0 else "predict"
    if option < 2:
        pass
        # df = pd.read_csv(f"{ROOT}{MIDDLE}data{MIDDLE}knps_final_{keyword}.csv")
        # df = df[(df["code"] == ori_db["knps"]) & (df["class"] == int(ori_db["class_num"])) &
        #         (df['date'].str[:4] >= ori_db["start_year"]) & (df['date'].str[:4] <= ori_db["end_year"])].sort_values(
        #     'date')
    else:
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

    # 시연용
    keyword = "analysis" if option == 0 else "predict"
    if option < 2:
        pass
        # df = pd.read_csv(f"{ROOT}{MIDDLE}data{MIDDLE}knps_final_{keyword}.csv")
        # df = df[(df["code"] == ori_db["knps"]) & (df["class"] == int(ori_db["class_num"])) &
        #         (df['date'].str[:4] >= ori_db["start_year"]) & (df['date'].str[:4] <= ori_db["end_year"])].sort_values(
        #     'date')
    else:
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


def plot_decompose(result):
    '''
    시계열 분석 그래프가 너무 작게 보여서 subplot으로 크게 보이게 하는 함수
    '''
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
    result.observed.plot(legend=False, ax=ax1)
    ax1.set_ylabel('Observed')
    result.trend.plot(legend=False, ax=ax2)
    ax2.set_ylabel('Trend')
    result.seasonal.plot(legend=False, ax=ax3)
    ax3.set_ylabel('Seasonal')
    result.resid.plot(legend=False, ax=ax4)
    ax4.set_ylabel('Residual')


def serial_compose(data_input):
    '''
    시계열 분해를 통해 시계열 데이터의 추세, 계절성, 주기를 확인하고자 한다.
    '''
    df = data_input[['date', 'avg']]
    df.index = df.date
    ts = df.drop("date", axis=1)

    result = seasonal_decompose(ts, model='additive', period=46)

    plot_decompose(result)


# 전체 및 산림별 데이터 분포 확인하는 메서드
def show_data_distribution():
    # 전체 데이터 확인
    data = preprocessing.get_final_data(all=True)
    data['date'] = pd.to_datetime(data['date'])  # date 칼럼을 날짜 형식으로 변환

    # 전체 데이터 기술통계량 확인
    print('Descriptive Statistic of All \n', data['avg'].describe())

    # 전체 데이터에 대한 Boxplot (연도별 EVI 분포 확인)
    plt.figure(figsize=(10, 7))
    p = sns.boxplot(y=data['avg'], x=data['date'].dt.year, palette='Spectral')
    p.set_title('Boxplot of Each Years', fontsize=20)
    p.set_ylabel('EVI', fontsize=15)
    p.set_xlabel('Year', fontsize=15)
    plt.show()

    # 산림별 데이터 특징 확인
    class_name = ['grassland', 'coniferous', 'broadleaved', 'mixed']
    for i in range(4):
        data_class = data[data['class'] == i]  # 특정 산림에 대한 데이터 프레임 생성

        # 특정 산림 기술통계량 확인
        print(f'Descriptive Statistic of ({class_name[i].capitalize()}) \n', data_class['avg'].describe())

        # 특정 산림 데이터에 대한 Boxplot (연도별 EVI 분포 확인)
        plt.figure(figsize=(10, 7))
        p = sns.boxplot(y=data_class['avg'], x=data_class['date'].dt.year, palette='Spectral')
        p.set_title(f'Boxplot of Each Years ({class_name[i].capitalize()})', fontsize=20)
        p.set_ylabel('EVI', fontsize=15)
        p.set_xlabel('Year', fontsize=15)
        plt.show()


# 시계열 데이터에서 ACF와 PACF 확인
# ACF를 통해 정상성 시계열이 아닌 것을 확인 -> 정상 시계열로 만들기 위해 차분 필요성 확인
# PACF를 통해 AR 모형임을 확인, ARIMA의 p값을 2로 설정
def show_acf_pacf_plot():
    # 확인할 특정 국립공원 선정
    parks = ['mudeung', 'wolchul', 'juwang', 'taean', 'halla', 'dadohae']
    for park in parks:
        data = preprocessing.get_final_data(all=True)
        data = data[(data['code'] == f'{park}') & (data['class'] == 2)]  # 각 국립공원 중 활엽수림 확인
        data['date'] = pd.to_datetime(data['date'])  # 시계열 분석을 위해 index를 날짜형으로 변경

        # 2019년 이하 연도를 train 설정
        train = data[data['date'].dt.year <= 2019]
        train.index = train['date']
        train = train.loc[:, 'avg']

        # ACF & PACF Plot
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.5)

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        plot_acf(train, ax=ax1)
        plot_pacf(train, ax=ax2)

        plt.suptitle(f'{park.capitalize()}', fontsize=20)
        plt.show()
