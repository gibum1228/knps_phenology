import os
import platform
import phenoKOR as pk
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# 전역변수
middle = "/" if platform.system() == "Darwin" else "\\" # 운영체제에 따라 슬래쉬 설정
root = os.getcwd() + middle


def check_stationarity(path):
    x, *y = pk.load_csv(path)


def DL():
    ## 국립공원 22개 리스트
    # parks = ['bukhan','byeonsan','chiak','dadohae','deogyu','gaya','gyeongju','gyeryong','halla','hallyeo','juwang',
    #          'mudeung','naejang','odae','seorak','sobaek','songni','taean','taebaek','wolchul','worak']

    ## ISSUE : jiri(지리산) 데이터 4년치 누락되어 리스트에서 잠시 제외함(2004,2006,2007,2009)

    parks = ['deogyu']  ## 덕유까지 완료

    ## 모든 국립공원 한번에 Double Logistic 처리
    for park in parks:

        path = f'C:/Users/cdbre/Desktop/Project/data/{park}/'  ## 데이터 저장된 파일 경로
        file_list = os.listdir(path)  ## 저장된 파일들 리스트화하여 저장
        file_list = [file for file in file_list if file.endswith('csv')]

        ## class 별 dataframe, wEVI, sos list 설정
        grassland = pd.DataFrame(columns=['Datetime', 'avg'])  ## class 0 : 초지
        coniferous = pd.DataFrame(columns=['Datetime', 'avg'])  ## class 1 : 침엽수
        broadleaved = pd.DataFrame(columns=['Datetime', 'avg'])  ## class 2 : 활엽수
        mixed = pd.DataFrame(columns=['Datetime', 'avg'])  ## class 3 : 혼효림

        wEVI_gl = []  ## 초지 wEVI
        wEVI_cf = []  ## 침엽수 wEVI
        wEVI_bl = []  ## 활엽수 wEVI
        wEVI_mx = []  ## 혼효림 wEVI

        sos_gl = []  ## 초지 SOS
        sos_cf = []  ## 침엽수 SOS
        sos_bl = []  ## 활엽수 SOS
        sos_mx = []  ## 혼효림 SOS

        for file in file_list:

            data = pd.read_csv(path + file)
            # what = file.split('_')[2][:-4]  ## 파일 이름에서 어떤 종류인지 추출

            for j in range(0, 4):

                data_class = data[data['class'] == j]
                data_class = data_class.loc[:, ['date', 'avg']]  ## date, avg 칼럼 추출

                ## DOY 추출 후, index 로 설정
                date = []
                for d in list(data_class.date):
                    d = str(d)
                    date.append(int(d[-3:]))

                data_class.index = date
                data_class.drop('date', axis=1, inplace=True)

                ## 줄리안 데이 : 10월 16일 - 290, 11월 17일 - 322, 그 사이 값의 기하 평균을 wEVI로 설정
                wEVI_df = data_class[(data_class.index >= 290) & (data_class.index <= 322)]

                mul_wEVI = 1
                for item in list(wEVI_df.avg):  ## winterEVI 안에 10월 16일 ~ 11월 17일 사이 값 저장
                    mul_wEVI *= item

                wEVI = mul_wEVI ** (1 / len(wEVI_df.avg))  ## 기하 평균

                if j == 0:
                    wEVI_gl.append(wEVI)
                elif j == 1:
                    wEVI_cf.append(wEVI)
                elif j == 2:
                    wEVI_bl.append(wEVI)
                else:
                    wEVI_mx.append(wEVI)

        ## 각 class 별 wEVI 구하기
        wEVI_list = [wEVI_gl, wEVI_cf, wEVI_bl, wEVI_mx]

        wEVI_values = []  ## wEVI 값 저장할 리스트

        for i in wEVI_list:
            mul_wEVI = 1
            for i2 in i:
                mul_wEVI *= i2
            i = mul_wEVI ** (1 / len(i))
            wEVI_values.append(i)

        ## 최종 커브 피팅 적용
        for file in file_list:

            data = pd.read_csv(path + file)
            year = int(file.split('_')[0])

            for j in range(0, 4):

                data_class = data[data['class'] == j]
                data_class['Datetime'] = data_class['date'].apply(
                    lambda x: format(datetime.strptime(x, '%Y-%j'), '%Y-%m-%d'))

                data_sub = pd.DataFrame(columns=['Datetime', 'avg', 'DOY'])
                data_sub['Datetime'] = data_class['Datetime']  ## 출력될 데이터 프레임의 날짜 형식 칼럼

                ## date, avg, Datetime 칼럼 추출
                data_class = data_class.loc[:, ['date', 'avg']]

                ## DOY 추출 후, index 로 설정
                date = []
                doy_each = []
                for d in list(data_class['date']):
                    d = str(d)
                    date.append(int(d[-3:]) + 365 * (int(year) - 2003))  ## 연도 DOY 반영하여 index 설정
                    doy_each.append(int(d[-3:]))  ## 각각 DOY 반영하여 data_sub 칼럼에 추가

                data_class.index = date
                data_sub.index = date
                data_sub['DOY'] = doy_each  ## 추후 연도별 Plot에 용이
                data_class.drop('date', axis=1, inplace=True)

                ## wEVI보다 작은 값들은 wEVI 값으로 대체
                data_class[data_class['avg'] < wEVI_values[j]] = wEVI_values[j]

                ## Normalize 함수
                def Normalize(x, sf):
                    return (x - sf[0]) / (sf[1] - sf[0])

                ## Backnormalize 함수
                def Backnormalize(x, sf):
                    return (x + sf[0] / (sf[1] - sf[0])) * (sf[1] - sf[0])

                ## Scaling Factor
                sf1 = list(data_class['avg'].quantile(q=(0.05, 0.95)))

                t = list(data_class.index)  ## DOY t에 저장
                tout = t  ## 최종 prediction 에서 쓸 DOY

                x = Normalize(data_class['avg'], sf1)  ## 정규화 x에 저장
                n = len(x)  ## n에 x의 길이 저장
                avg = np.mean(x)  ## 평균
                mx = np.max(x)  ## 최대값
                mn = np.min(x)  ## 최소값
                ampl = mx - mn  ## 최대값과 최소값의 차이

                ## Double Logistic 함수 정의
                def DoubleLogistic(par, t):
                    '''
                    mn : min EVI, mx : max EVI, sos : start of season, 증가 변곡점, rsp : sos일 때 변화율,
                    eos : end of season, 감소 변곡점, rau : eos일 때 변화율
                    '''
                    mn, mx, sos, rsp, eos, rau = par[0], par[1], par[2], par[3], par[4], par[5]
                    xpred = mn + (mx - mn) * (1 / (1 + np.exp(-rsp * (t - sos))) + 1 / (1 + np.exp(rau * (t - eos))))
                    df = pd.Series(xpred)
                    df.index = t
                    return df

                ## SSE 함수 정의
                def SSE(par, x, weights):
                    if par[0] > par[1]:
                        return 99999
                    x_pred = DoubleLogistic(par, t=t)
                    sse = np.sum(((x_pred - x) ** 2) * weights)
                    return sse

                ## 가중치 전부 1로 하여 저장
                weights = []
                for weight in range(len(x)):
                    weight = 1
                    weights.append(weight)

                doy = list(pd.Series(t).quantile(q=(0.25, 0.75)))  ## 위에서 설정한 t(doy 숫자)에 대하여 25, 75 백분위 설정

                ## parameter 초기값 설정
                prior = pd.DataFrame(np.array([[mn, mx, doy[0], 0.5, doy[1], 0.5],
                                               [mn, mx, doy[1], 0.5, doy[0], 0.5],
                                               [mn - ampl / 2, mx + ampl / 2, doy[0], 0.5, doy[1], 0.5],
                                               [mn - ampl / 2, mx + ampl / 2, doy[1], 0.5, doy[0], 0.5]]),
                                     columns=['mn', 'mx', 'sos', 'rsp', 'eos', 'rau'])

                iter = 3

                ## 최적의 값 선택 위한 데이터 프레임 설정
                opt_df = pd.DataFrame(columns=['cost', 'success', 'param'])

                ## 최적 parameter 선정 위한 반복문
                for a in range(iter):
                    for b in range(len(prior)):
                        opt_l = minimize(fun=SSE, x0=prior.iloc[b], args=(list(x), weights), method='BFGS',
                                         options={'maxiter': 1000})
                        opt_l_df = pd.DataFrame({'cost': opt_l.fun, 'success': opt_l.success, 'param': [opt_l.x]})
                        opt_df = pd.concat([opt_df, opt_l_df])

                    best = np.argmin(opt_df.cost)  ## 비용 함수(cost)가 최소인 지점을 best로 설정
                    if not opt_df.success.iloc[best]:  ## success False일 경우
                        opt = opt_df.iloc[best, :]
                        opt = minimize(fun=SSE, x0=opt.param, args=(list(x), weights), method='BFGS',
                                       options={'maxiter': 1000})
                        opt_param = opt.x
                        ## cost 최소인 지점에서 parameter를 통해 다시 최적화
                        df = pd.DataFrame(opt_param).T
                        df.columns = ['mn', 'mx', 'sos', 'rsp', 'eos', 'rau']
                        prior = pd.concat([prior, df], axis=0)  ## prior에 최적 parameter 추가
                        xpred = DoubleLogistic(opt_param, t=t)  ## 최적 parameter로 Double Logistic 실행
                    elif opt_df.success.iloc[best]:  ## success 가 True 일 경우
                        opt = opt_df.iloc[best, :]
                        opt_param = opt.param
                        df = pd.DataFrame(opt_param).T
                        df.columns = ['mn', 'mx', 'sos', 'rsp', 'eos', 'rau']
                        prior = pd.concat([prior, df], axis=0)  ## prior에 최적 parameter 추가
                        xpred = DoubleLogistic(opt_param, t=t)  ## 최적 parameter로 Double Logistic 실행

                    par_init = opt_param
                    mn, mx, sos, rsp, eos, rau = opt_param[0], opt_param[1], opt_param[2], opt_param[3], opt_param[4], \
                                                 opt_param[5]

                    ## 독립변수 sos, eos를 통해 종속변수 0, 100이 나오는 회귀분석 WHY!!
                    m = LinearRegression().fit(np.array([sos, eos]).reshape(-1, 1), np.array([0, 100]).reshape(-1, 1))
                    tr = m.coef_ * t + m.intercept_  ## m.coef_는 기울기, m.intercept_은 intercept 이다.
                    tr = tr.reshape(-1)  ## 1차원 배열로 변형
                    tr[tr < 0] = 0  ## 예측값 0보다 작으면 0
                    tr[tr > 100] = 100  ## 예측값 100보다 크면 100
                    res = xpred - x  ## 잔차
                    weights = 1 / ((tr * res + 1) ** 2)
                    weights[(res > 0) & (res <= 0.01)] = 1  ## 잔차가 0 초과 0.01 이하면 가중치 1
                    weights[res < 0] = 4  ## 잔차가 0보다 작으면 가중치 4
                    weights = list(weights)  ## 가중치 변경 후 반복문
                '''
                ## 최적 opt 의 success False 일 경우
                if not opt.success:
                    opt_param[:] = 'NA'  ## parameter 설정한거 결측값 처리
                    xpred = []
                    for c in range(len(tout)):  ## xpred에 NA를 tout 길이만큼 저장
                        c = 'NA'
                        xpred.append(c)
                '''
                # else:  ## 0 이면 더블 로지스틱 함수 써서 xpred 예측값 도출
                xpred = DoubleLogistic(opt_param, t=tout)

                ## 정규화 해준거 원래대로 돌려놓기
                xpred = Backnormalize(xpred, sf=sf1)

                data_sub['avg'] = xpred

                if j == 0:
                    grassland = pd.concat([grassland, data_sub])
                    sos_gl.append(opt_param[2])
                elif j == 1:
                    coniferous = pd.concat([coniferous, data_sub])
                    sos_cf.append(opt_param[2])
                elif j == 2:
                    broadleaved = pd.concat([broadleaved, data_sub])
                    sos_bl.append(opt_param[2])
                else:
                    mixed = pd.concat([mixed, data_sub])
                    sos_mx.append(opt_param[2])

        ## SOS, Double Logistic 적용 값 csv 저장
        sos_all = [sos_gl, sos_cf, sos_bl, sos_mx]
        for i, sos_list in enumerate(sos_all):
            for j, sos in enumerate(sos_list):
                sos = round(sos)  ## 값이 소수점으로 나와서 round으로 반올림
                sos_list[j] = sos
            sos_all[i] = pd.DataFrame(sos_list, index=range(2003, 2022), columns=['sos'])
            if i == 0:
                sos_all[i].to_csv(f'C:/Users/cdbre/Desktop/Project/data/pred/sos/{park}_sos_grassland.csv')
                grassland.to_csv(f'C:/Users/cdbre/Desktop/Project/data/pred/{park}/{park}_DL_grassland_datetime.csv')
            elif i == 1:
                sos_all[i].to_csv(f'C:/Users/cdbre/Desktop/Project/data/pred/sos/{park}_sos_coniferous.csv')
                coniferous.to_csv(f'C:/Users/cdbre/Desktop/Project/data/pred/{park}/{park}_DL_coniferous_datetime.csv')
            elif i == 2:
                sos_all[i].to_csv(f'C:/Users/cdbre/Desktop/Project/data/pred/sos/{park}_sos_broadleaved.csv')
                broadleaved.to_csv(
                    f'C:/Users/cdbre/Desktop/Project/data/pred/{park}/{park}_DL_broadleaved_datetime.csv')
            else:
                sos_all[i].to_csv(f'C:/Users/cdbre/Desktop/Project/data/pred/sos/{park}_sos_mixed.csv')
                mixed.to_csv(f'C:/Users/cdbre/Desktop/Project/data/pred/{park}/{park}_DL_mixed_datetime.csv')