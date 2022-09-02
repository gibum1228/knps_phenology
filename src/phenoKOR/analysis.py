import warnings
import json
from datetime import date, timedelta

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from fusioncharts import FusionCharts
from fusioncharts import FusionTable
from fusioncharts import TimeSeries

import preprocessing

warnings.filterwarnings('ignore')

# 전역 변수
ROOT, MIDDLE = preprocessing.get_info()


# Double Logistic 함수 정의
def Rangers_DL(input_data, start_year, end_year, ori_db):
    '''
    :param Data : input DataFrame
    :param start_year, end_year
    :return: Double Logistic DataFrame, SOS DataFrame
    '''

    data = input_data

    if ori_db['AorP'] == 'A':
        data_final = pd.DataFrame(columns=['code', 'class', 'date', 'avg', 'DOY'])
    else:
        data_final = pd.DataFrame(columns=['date', 'avg', 'DOY'])  # return 할 데이터 프레임
    sos_list = []  # return 할 SOS 값

    each_year_list = []
    for each_date in data['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    data['Year'] = each_year_list  # date 칼럼에서 앞 4개 문자만 추출한 것을 Year 칼럼 추가
    data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]  # 보고 싶은 연도 설정 반영

    data['DOY'] = data['date'].apply(lambda x: int(format(datetime.strptime(x, '%Y-%m-%d'), '%j')))  # DOY 칼럼 추가

    # Normalize 함수
    def Normalize(x, sf):
        return (x - sf[0]) / (sf[1] - sf[0])

    # Backnormalize 함수
    def Backnormalize(x, sf):
        return (x + sf[0] / (sf[1] - sf[0])) * (sf[1] - sf[0])

    # Double Logistic 함수 정의
    def DoubleLogistic(par, t):
        '''
        mn : min EVI, mx : max EVI, sos : start of season, 증가 변곡점, rsp : sos에서 변화율,
        eos : end of season, 감소 변곡점, rau : eos에서 변화율
        '''
        mn, mx, sos, rsp, eos, rau = par[0], par[1], par[2], par[3], par[4], par[5]
        xpred_dl = mn + (mx - mn) * (
                1 / (1 + np.exp(-rsp * (t - sos))) + 1 / (1 + np.exp(rau * (t - eos))) - 1)  # 더블 로지스틱 수식
        return pd.Series(xpred_dl, index=t)

    # SSE 함수 정의
    def SSE(par, x, weights):
        if par[0] > par[1]:  # min EVI가 max EVI 보다 클 경우 99999 반환
            return 99999
        xpred_sse = DoubleLogistic(par, t=t)
        sse = np.sum(((xpred_sse - x) ** 2) * weights)
        return sse

    # 연도별 Double Logistic 적용 위한 반복문
    for each_year in range(start_year, end_year + 1):

        data_re = data[data['Year'] == each_year]
        data_re.index = data_re['DOY']
        if ori_db['AorP'] == 'A':
            data_re = data_re.loc[:, ['code', 'class', 'date', 'avg', 'DOY']]
        else:
            data_re = data_re.loc[:, ['date', 'avg', 'DOY']]

        # 고정 겨울 : 겨울 기간 내 최소 EVI 값이 0보다 작을 경우 0 사용, 0보다 클 경우 최소 EVI 값 사용
        if np.min(data_re['avg']) <= 0:
            data_re.loc[(data_re.index >= 1) & (data_re.index <= 57), 'avg'] = 0
            data_re.loc[(data_re.index >= 336) & (data_re.index <= 365), 'avg'] = 0
        else:
            data_re.loc[(data_re.index >= 1) & (data_re.index <= 57), 'avg'] = np.min(data_re['avg'])
            data_re.loc[(data_re.index >= 336) & (data_re.index <= 365), 'avg'] = np.min(data_re['avg'])

        # Scaling Factor
        sf1 = list(data_re['avg'].quantile(q=(0.05, 0.95)))

        t = list(data_re.DOY)  # DOY t에 저장
        tout = t  # 최종 prediction 에서 쓸 DOY

        x = Normalize(data_re['avg'], sf1)  # 정규화 x에 저장
        mx = np.max(x)  # 최대값
        mn = np.min(x)  # 최소값
        ampl = mx - mn  # 최대값 최소값 차이

        # SSE 최소화할 때 사용할 가중치
        weights = []
        for weight in range(len(x)):
            weight = 1  # 초기 가중치 모두 1로 설정
            weights.append(weight)

        doy = list(pd.Series(t).quantile(q=(0.25, 0.75)))  # 위에서 설정한 t(doy 숫자)에 대하여 25, 75 백분위 설정

        # parameter 초기값 설정
        prior = pd.DataFrame(np.array([[mn, mx, doy[0], 0.5, doy[1], 0.5],
                                       [mn, mx, doy[1], 0.5, doy[0], 0.5],
                                       [mn - ampl / 2, mx + ampl / 2, doy[0], 0.5, doy[1], 0.5],
                                       [mn - ampl / 2, mx + ampl / 2, doy[1], 0.5, doy[0], 0.5]]),
                             columns=['mn', 'mx', 'sos', 'rsp', 'eos', 'rau'])

        iteration = 2

        # 최적 parameter 선정 위한 반복문
        for a in range(iteration):

            opt_df = pd.DataFrame(columns=['cost', 'success', 'param'])  # 최적의 값 선택 위한 데이터 프레임

            for b in range(len(prior)):  # prior 데이터 프레임 한 행씩 적용

                opt_l = minimize(fun=SSE, x0=prior.iloc[b], args=(list(x), weights), method='Nelder-Mead',
                                 options={'maxiter': 1000})
                opt_l_df = pd.DataFrame({'cost': opt_l.fun, 'success': opt_l.success, 'param': [opt_l.x]})
                opt_df = pd.concat([opt_df, opt_l_df], axis=0)

            best = np.argmin(opt_df.cost)  # 비용 함수(cost)가 최소인 지점 best 설정

            if not opt_df.success.iloc[best]:  # success False일 경우

                opt = opt_df.iloc[best, :]
                opt = minimize(fun=SSE, x0=opt.param, args=(list(x), weights), method='Nelder-Mead',
                               options={'maxiter': 1500})  # cost 최소인 지점의 parameter로 다시 최적화

                if a == 1:  # 두 번째 iteration 때도 success가 False가 뜬다면 True인 것만 활용
                    if not opt.success:
                        best = np.argmin(opt_df[opt_df.success].cost)
                        opt = opt_df[opt_df.success].iloc[best, :]
                        opt_param = opt.param
                    else:
                        opt_param = opt.x

                if a == 0:  # 첫 번째 iteration success가 False여도 진행
                    opt_param = opt.x

                df = pd.DataFrame(opt_param).T
                df.columns = ['mn', 'mx', 'sos', 'rsp', 'eos', 'rau']
                prior = pd.concat([prior, df], axis=0)  # prior에 최적 parameter 추가
                xpred = DoubleLogistic(opt_param, t=t)  # 최적 parameter로 Double Logistic 실행

            elif opt_df.success.iloc[best]:  # success 가 True 일 경우
                opt = opt_df.iloc[best, :]
                opt_param = opt.param
                df = pd.DataFrame(opt_param).T
                df.columns = ['mn', 'mx', 'sos', 'rsp', 'eos', 'rau']
                prior = pd.concat([prior, df], axis=0)  # prior에 최적 parameter 추가
                xpred = DoubleLogistic(opt_param, t=t)  # 최적 parameter로 Double Logistic 실행

            mn, mx, sos, rsp, eos, rau = opt_param[0], opt_param[1], opt_param[2], opt_param[3], opt_param[4], \
                                         opt_param[5]

            # 독립변수 sos, eos를 통해 종속변수 0, 100이 나오는 회귀분석과 잔차를 통해 weights 조정
            m = LinearRegression().fit(np.array([[sos], [eos]]), np.array([[0], [100]]))
            tr = m.coef_ * t + m.intercept_  # m.coef_는 기울기, m.intercept_은 intercept
            tr = tr.reshape(-1)  # 1차원 배열로 변형
            tr[tr < 0] = 0  # 예측값 0보다 작으면 0
            tr[tr > 100] = 100  # 예측값 100보다 크면 100
            res = xpred - x  # 잔차
            weights = 1 / ((tr * res + 1) ** 2)  # 기본 weights 설정
            weights[(res > 0) & (res <= 0.01)] = 1  # 잔차가 0 초과 0.01 이하면 가중치 1
            weights[res < 0] = 4  # 잔차가 0보다 작으면 가중치 4
            weights = list(weights)  # 가중치 변경 후 반복문 재실행

        # 최적 parameter로 Double Logistic 함수 적용 후 Backnormalizing을 통해 최종값 도출
        xpred = Backnormalize(DoubleLogistic(opt_param, t=tout), sf=sf1)

        data_re['avg'] = xpred.tolist()  # data_re의 avg 칼럼을 Double Logistic 적용한 값으로 변환

        data_final = pd.concat([data_final, data_re])  # 반환할 Double Logistic 데이터 프레임에 추가
        sos_list.append(np.floor(opt_param[2]))  # 반환할 SOS 데이터 프레임에 추가

    sos_df = pd.DataFrame()
    sos_df['Year'] = [i for i in range(start_year, end_year + 1)]
    sos_df['sos_DOY'] = sos_list  # 누적 DOY를 반환할 데이터 프레임 인덱스로 설정

    return data_final, sos_df


# Savitzky-Golay Function
def Rangers_SG(data_input, start_year, end_year, ori_db):
    '''
    :param data_input : input DataFrame
    :param start_year, end_year
    :return: Savitzky-Golay DataFrame
    '''

    data = data_input  # 입력받은 데이터 data에 저장
    print(data)

    each_year_list = []
    for each_date in data['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    data['Year'] = each_year_list  # date 칼럼에서 앞 4개 문자만 추출한 것을 Year 칼럼 추가
    data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]  # 보고 싶은 연도 설정 반영

    data['DOY'] = data['date'].apply(lambda x: int(format(datetime.strptime(x, '%Y-%m-%d'), '%j')))  # DOY 칼럼 추가

    # Scipy 내부에 있는 savgol_filter 활용
    print(data.avg.shape)
    data['avg'] = savgol_filter(data.avg, window_length=5, polyorder=1)

    if ori_db['AorP'] == 'A':
        data = data.loc[:, ['code', 'class', 'date', 'avg', 'DOY']]
    else:
        data = data.loc[:, ['date', 'avg', 'DOY']]

    sos_df = [0]
    return data, sos_df


# Gaussian Function
def Rangers_GSN(data_input, start_year, end_year, ori_db):
    '''
    :param data_input : input DataFrame
    :param start_year, end_year
    :return: Gaussian DataFrame
    '''

    data = data_input  # 입력받은 데이터 data에 저장

    each_year_list = []
    for each_date in data['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    data['Year'] = each_year_list  # date 칼럼에서 앞 4개 문자만 추출한 것을 Year 칼럼 추가
    data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]  # 보고 싶은 연도 설정 반영

    data['DOY'] = data['date'].apply(lambda x: int(format(datetime.strptime(x, '%Y-%m-%d'), '%j')))  # DOY 칼럼 추가

    # Gaussian Filtering
    kernel1d = cv2.getGaussianKernel(46 * (end_year - start_year + 1), 1)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    data['avg'] = cv2.filter2D(np.array(data.avg), -1, kernel2d).reshape(-1).tolist()  # convolve

    if ori_db['AorP'] == 'A':
        data = data.loc[:, ['code', 'class', 'date', 'avg', 'DOY']]
    else:
        data = data.loc[:, ['date', 'avg', 'DOY']]

    sos_df = [0]
    return data, sos_df


# PPT 시각화

# 고정 카메라 결측치 확인
def pheno_test_plot():
    pheno_test = pd.read_csv('C:/Users/cdbre/Desktop/Project/data/pheno_test.csv')
    pheno_test = pheno_test.groupby('day').mean()  # 일별 평균 데이터 추출

    # 지리산 2020년 1월 데이터 확인 Plot
    plt.figure(figsize=(15, 5))
    plt.scatter(pheno_test.index, pheno_test.rcc, color='red')
    plt.xticks(range(32), range(32))
    plt.ylim(0.2, 0.4)
    plt.gca().axes.yaxis.set_visible(False)  # y축 값 제거
    plt.show()


def see_sos():
    # 개엽일
    SOS_split = pd.read_csv('C:/Users/cdbre/Desktop/Project/data/SOS_final.csv')
    SOS_split.drop('Unnamed: 0', axis=1, inplace=True)
    SOS_split.index = range(2003, 2022)
    SOS_split = SOS_split.iloc[:, SOS_split.columns.str.contains('2')]
    SOS_split.columns = SOS_split.columns.str[:-2]

    sos_test = SOS_split.loc[:, 'bukhan']

    # 4년 씩 묶어서 계산
    test_4 = []
    for i in range(4):
        test_4.append(np.mean(sos_test[(sos_test.index >= 2003 + 4 * i) & (sos_test.index < 2007 + 4 * i)]))
    for i in range(2019, 2022):
        test_4.append(sos_test[sos_test.index == i].iloc[0])

    test_4_df = pd.DataFrame(test_4)
    print(test_4_df)

    # 분석 개엽일 추세 Plot
    plt.figure(figsize=(20, 5))
    plt.rcParams['font.size'] = 12
    plt.plot(test_4, color='red', marker='o', markersize=8)
    plt.xticks([0, 1, 2, 3, 4, 5, 6],
               labels=['2003~2006', '2007~2010', '2011~2014', '2015~2018', '2019', '2020', '2021'])
    plt.ylim(90, 120)
    plt.axis('off')
    plt.show()


# # 회귀 직선 - 개엽일 추세 예측
# X = np.array([0,1,2,3,4,5,6]).reshape(-1,1)
# y = np.array([106.5, 107.5, 110.25, 104.0, 108.0, 107.0, 98.0]).reshape(-1,1)
# m = LinearRegression()
# m.fit(X, y)
# print('2033 SOS prediction :', m.predict([[18]]))  # 2033년의 개엽일은 3월 31일

# 개엽일과 봄철 기온 상관관계 함수
def corr_sos_temp():
    '''
    Example : 북한산의 개엽일과 봄철 기온의 상관관계 (2003년 ~ 2021년)
    북한산 기온 데이터 (기상청 기상자료개방포털 활용)
    '''
    data = pd.read_csv(f'C:/Users/cdbre/Desktop/Project/data/weather/bukhan_tem.csv', encoding='CP949')
    data.columns = ['num', 'site', 'date', 'temp', 'precip']
    data = data.iloc[:, 2:]

    data['Year'] = data['date'].apply(lambda x: format(datetime.strptime(x, '%b-%y'), '%Y'))
    data['Month'] = data['date'].apply(lambda x: format(datetime.strptime(x, '%b-%y'), '%m'))

    data_spring = data[(data['Month'] == '03') | (data['Month'] == '04')]  # 봄철인 3월과 4월만 고려
    data_spring = data_spring.groupby(['Year']).mean()  # 각 연도별 봄철 평균 기온

    SOS_df = pd.read_csv('C:/Users/cdbre/Desktop/Project/data/SOS_final.csv')
    SOS_df.drop('Unnamed: 0', axis=1, inplace=True)
    SOS_df = SOS_df.iloc[:, SOS_df.columns.str.contains('2')]  # 활엽수만 비교하기 위해 2 포함한 열만 추출
    SOS_df.columns = SOS_df.columns.str[:-2]  # 칼럼명에서 마지막 _2 제거
    SOS_df.index = range(2003, 2022)
    data_sos = SOS_df.loc[:, ['bukhan']]  # 개엽일 데이터 프레임에서 '북한산' 국립 공원 선택
    data_sos = data_sos.astype('int')  # 개엽일 정수형으로 변환

    # 봄철 기온의 변화와 개엽일 변화 추이
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 5)
    plt.rcParams['font.size'] = 10

    fig, ax1 = plt.subplots()
    ax1.plot(data_sos.index, data_sos, color='green', lw=4)

    ax2 = ax1.twinx()
    ax2.plot(data_sos.index, data_spring.temp, color='red', lw=4)
    plt.axis('off')
    plt.show()


# 식생지수의 최대값과 최소값 변화 추이 비교 함수
def sos_minmax(park_input):
    CV_df = pd.read_csv('C:/Users/cdbre/Desktop/Project/data/DL_Final.csv')  # Double Logistic 적용된 데이터
    ori_data = pd.read_csv('C:/Users/cdbre/Desktop/Project/data/knps_final.csv')  # 원본 데이터

    CV_df.columns = ['CUM_DOY', 'code', 'class', 'date', 'avg', 'DOY']

    each_year_list = []
    for each_date in CV_df['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    CV_df.loc[:, 'Year'] = each_year_list  # Year 칼럼 추가

    # 국립 공원과 산림 지역 선택
    data = CV_df[(CV_df['code'] == park_input) & (CV_df['class'] == 2)]  # DL 적용 데이터에서 원하는 국립 공원의 활엽수림 선택

    # 비교할 원본 데이터
    OR_df = ori_data[(ori_data['code'] == park_input) & (ori_data['class'] == 2)]  # 원본 데이터에서 선택
    OR_df.index = data['CUM_DOY']

    each_year_list = []
    for each_date in OR_df['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    OR_df.loc[:, 'Year'] = each_year_list  # Year 칼럼 추가

    # 원본 데이터에서 min, max 추출
    OR_minmax = []
    for i in range(2003, 2022):
        year_max = np.max(OR_df[OR_df['Year'] == i].avg)  # 각 연도별 최대값 추가
        year_min = np.min(OR_df[OR_df['Year'] == i].avg)  # 각 연도별 최소값 추가
        OR_minmax.append([year_min, year_max])

    OR_minmax = pd.DataFrame(OR_minmax, columns=['min', 'max'], index=range(2003, 2022))  # 각 연도별 min, max 데이터 프레임

    # DL 데이터에서 min, max 추출
    CV_minmax = []
    for i in range(2003, 2022):
        year_max = np.max(data[data['Year'] == i].avg)
        year_min = np.min(data[data['Year'] == i].avg)
        CV_minmax.append([year_min, year_max])

    CV_minmax = pd.DataFrame(CV_minmax, columns=['min', 'max'], index=range(2003, 2022))

    print('max \n', CV_minmax['max'])
    print('min \n', CV_minmax['min'])

    # min, max 값 서로 비교 Plot
    plt.figure(figsize=(20, 5))
    plt.title(f'Comparing {park_input} Broadleaved MIN & Max EVI')
    plt.plot(OR_minmax.index, OR_minmax['min'], color='black', label='Original Min', linestyle='dashed', lw=2)
    plt.plot(OR_minmax.index, OR_minmax['max'], color='black', label='Original Max', linestyle='dashed', lw=2)
    plt.plot(CV_minmax.index, CV_minmax['min'], color='red', label='Curve Fitting Min', lw=3)
    plt.plot(CV_minmax.index, CV_minmax['max'], color='red', label='Curve Fitting Max', lw=3)
    plt.xticks(range(2002, 2023), range(2002, 2023))
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.show()


# 1년치 원본값 Plot 확인 (2021년 지리산 활엽수림)
def org_bl():
    hello = pd.read_csv('C:/Users/cdbre/Desktop/Project/data/knps_final.csv')

    hello = hello[(hello['code'] == 'jiri') & (hello['class'] == 2)]

    each_year_list = []
    for each_date in hello['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    hello['Year'] = each_year_list

    hello = hello[hello['Year'] == 2021]
    hello['DOY'] = hello['date'].apply(lambda x: int(format(datetime.strptime(x, '%Y-%m-%d'), '%j')))

    # 원본 데이터 Plot
    plt.figure(figsize=(15, 5))
    plt.plot(hello.DOY, hello.avg, color='blue', lw=4)
    plt.axis('off')
    plt.show()


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
def show_graphs(ori_db: dict, option: int = 2):
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
