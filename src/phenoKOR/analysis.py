# 메서드
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import cv2
import warnings
warnings.filterwarnings('ignore')

## Double Logistic 함수 정의
def Rangers_DL(path, park, class_input, start_year, end_year):
    '''
    :param path: write your path
    :param park: which park you are interested in
    :param class_input : must be 0(grassland), 1(coniferous), 2(broadleaved), 3(mixed)
    :param start_year, end_year
    :return:
    '''

    data = pd.read_csv(path + 'knps_final.csv')

    data_final = pd.DataFrame(columns=['code', 'class', 'date', 'avg', 'DOY'])  # 최종적으로 return 할 데이터 프레임
    sos_list = []  # 최종적으로 return 할 sos 데이터

    data = data[(data['code'] == park) & (data['class'] == class_input)]  # 원하는 국립 공원, 산림 지정

    each_year_list = []
    for each_date in data['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    data['Year'] = each_year_list
    data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]  # 보고 싶은 연도 설정 반영

    data['DOY'] = data['date'].apply(lambda x: int(format(datetime.strptime(x, '%Y-%m-%d'), '%j')))
    data['DOY_CUM'] = data['DOY'] + 365 * (data['Year'] - data['Year'].iloc[0])  # 첫 번째 연도 기준으로 DOY 누적 게산

    # Normalize 함수
    def Normalize(x, sf):
        return (x - sf[0]) / (sf[1] - sf[0])

    # Backnormalize 함수
    def Backnormalize(x, sf):
        return (x + sf[0] / (sf[1] - sf[0])) * (sf[1] - sf[0])

    # Double Logistic 함수 정의
    def DoubleLogistic(par, t):
        '''
        mn : min EVI, mx : max EVI, sos : start of season, 증가 변곡점, rsp : sos일 때 변화율,
        eos : end of season, 감소 변곡점, rau : eos일 때 변화율
        '''
        mn, mx, sos, rsp, eos, rau = par[0], par[1], par[2], par[3], par[4], par[5]
        xpred = mn + (mx - mn) * (1 / (1 + np.exp(-rsp * (t - sos))) + 1 / (1 + np.exp(rau * (t - eos))) - 1)
        df = pd.Series(xpred, index=t)
        return df

    # SSE 함수 정의
    def SSE(par, x, weights):
        if par[0] > par[1]:
            return 99999
        x_pred = DoubleLogistic(par, t=t)
        sse = np.sum(((x_pred - x) ** 2) * weights)
        return sse

    for each_year in range(start_year, end_year+1):

        data_re = data[data['Year'] == each_year]
        data_re.index = data_re['DOY']
        data_re = data_re.loc[:, ['code', 'class', 'date', 'avg', 'DOY']]

        # 고정 겨울 : 최소 EVI값이 0보다 작을 0 사용
        if np.min(data_re['avg']) <= 0:
            data_re.loc[(data_re.index >= 1) & (data_re.index <= 57), 'avg'] = 0
            data_re.loc[(data_re.index >= 336) & (data_re.index <= 365), 'avg'] = 0  # 겨울의 경우 식생지수를 최소값으로 고정
        else:
            data_re.loc[(data_re.index >= 1) & (data_re.index <= 57), 'avg'] = np.min(data_re['avg'])
            data_re.loc[(data_re.index >= 336) & (data_re.index <= 365), 'avg'] = np.min(data_re['avg'])

        # Scaling Factor
        sf1 = list(data_re['avg'].quantile(q=(0.05, 0.95)))

        t = list(data_re.DOY)  # DOY t에 저장
        tout = t  # 최종 prediction 에서 쓸 DOY

        x = Normalize(data_re['avg'], sf1)  # 정규화 x에 저장
        n = len(x)  # n에 x의 길이 저장
        avg = np.mean(x)  # 평균
        mx = np.max(x)  # 최대값
        mn = np.min(x)  # 최소값
        ampl = mx - mn  # 최대값과 최소값의 차이

        # 가중치 전부 1로 하여 저장
        weights = []
        for weight in range(len(x)):
            weight = 1
            weights.append(weight)

        doy = list(pd.Series(t).quantile(q=(0.25, 0.75)))  # 위에서 설정한 t(doy 숫자)에 대하여 25, 75 백분위 설정

        # parameter 초기값 설정
        prior = pd.DataFrame(np.array([[mn, mx, doy[0], 0.5, doy[1], 0.5],
                                       [mn, mx, doy[1], 0.5, doy[0], 0.5],
                                       [mn - ampl / 2, mx + ampl / 2, doy[0], 0.5, doy[1], 0.5],
                                       [mn - ampl / 2, mx + ampl / 2, doy[1], 0.5, doy[0], 0.5]]),
                             columns=['mn', 'mx', 'sos', 'rsp', 'eos', 'rau'])

        iter = 2

        # 최적 parameter 선정 위한 반복문
        for a in range(iter):

            # 최적의 값 선택 위한 데이터 프레임 설정
            opt_df = pd.DataFrame(columns=['cost', 'success', 'param'])

            for b in range(len(prior)):
                opt_l = minimize(fun=SSE, x0=prior.iloc[b], args=(list(x), weights), method='Nelder-Mead',
                                 options={'maxiter': 1000})
                opt_l_df = pd.DataFrame({'cost': opt_l.fun, 'success': opt_l.success, 'param': [opt_l.x]})
                opt_df = pd.concat([opt_df, opt_l_df], axis=0)

            best = np.argmin(opt_df.cost)  # 비용 함수(cost)가 최소인 지점을 best로 설정

            if not opt_df.success.iloc[best]:  # success False일 경우
                opt = opt_df.iloc[best, :]
                print('be4 :', opt.success)
                opt = minimize(fun=SSE, x0=opt.param, args=(list(x), weights), method='Nelder-Mead',
                               options={'maxiter': 1500})  # cost 최소인 지점에서 parameter를 통해 다시 최적화

                if a == 1:  # 두 번째 iteration 때 그마저도 False가 뜬다면
                    if not opt.success:
                        best = np.argmin(opt_df[opt_df.success].cost)  # True 인 것만 활용
                        opt = opt_df[opt_df.success].iloc[best, :]
                        opt_param = opt.param
                    else:
                        opt_param = opt.x

                if a == 0:  # 첫 번째 iteration 은 그냥 False 여도 진행 가능
                    opt_param = opt.x

                print('after :', opt.success)
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

            par_init = opt_param
            mn, mx, sos, rsp, eos, rau = opt_param[0], opt_param[1], opt_param[2], opt_param[3], opt_param[4], \
                                         opt_param[5]

            # 독립변수 sos, eos를 통해 종속변수 0, 100이 나오는 회귀분석 WHY!!
            m = LinearRegression().fit(np.array([[sos], [eos]]), np.array([[0], [100]]))
            tr = m.coef_ * t + m.intercept_  # m.coef_는 기울기, m.intercept_은 intercept 이다.
            tr = tr.reshape(-1)  # 1차원 배열로 변형
            tr[tr < 0] = 0  # 예측값 0보다 작으면 0
            tr[tr > 100] = 100  # 예측값 100보다 크면 100
            res = xpred - x  # 잔차
            weights = 1 / ((tr * res + 1) ** 2)
            weights[(res > 0) & (res <= 0.01)] = 1  # 잔차가 0 초과 0.01 이하면 가중치 1
            weights[res < 0] = 4  # 잔차가 0보다 작으면 가중치 4
            weights = list(weights)  # 가중치 변경 후 반복문

        # 최적 opt 의 success False 일 경우
        if not opt.success:
            print('Finally Failed :', each_year)
            for num_NA in range(len(opt_param)):
                opt_param[num_NA] = np.nan  # parameter 설정한 것 결측값 처리
            xpred = []
            for c in range(len(tout)):  # xpred에 결측값을 tout 길이 만큼 저장
                c = np.nan
                xpred.append(c)

        else:  # 0 이면 더블 로지스틱 함수 써서 xpred 예측값 도출
            xpred = DoubleLogistic(opt_param, t=tout)
            xpred = Backnormalize(xpred, sf=sf1)

        if type(xpred) == list:  # xpred의 타입이 list일 때와 Series일 때로 출력되어 조건문 설정
            data_re['avg'] = xpred
        else:
            data_re['avg'] = xpred.tolist()

        data_final = pd.concat([data_final, data_re])
        sos_list.append(np.floor(opt_param[2]))

        print(park, class_input, each_year)

    data_final.index = data['DOY_CUM']

    return data_final, sos_list


# # 국립공원 22개 리스트
# parks = ['bukhan','byeonsan','chiak','dadohae','deogyu','gaya','gyeongju','gyeryong','halla','hallyeo','jiri',
#          'juwang','mudeung','naejang','odae','seorak','sobaek','songni','taean','taebaek','wolchul','worak']
#
#
# Rangers_df = pd.DataFrame(columns=['code','class','date','avg','DOY'])
# Rangers_sos = pd.DataFrame(index=range(2003,2022))
#
# for park in parks:
#     for class_input in range(4):
#         park_df, sos_df = Rangers_DL(path='C:/Users/cdbre/Desktop/Project/data/', park=park, class_input=class_input,
#                                      start_year=2003, end_year=2021)
#         Rangers_df = pd.concat([Rangers_df, park_df], axis=0)
#
#         sos_df = pd.DataFrame(sos_df, index=range(2003,2022), columns=[f'{park}_{class_input}'])
#         Rangers_sos = pd.concat([Rangers_sos, sos_df], axis=1)
#
# Rangers_df.to_csv('C:/Users/cdbre/Desktop/Project/data/FuckYou_LAST.csv')
# Rangers_sos.to_csv('C:/Users/cdbre/Desktop/Project/data/FuckYouSOS_LAST.csv')



# Savitzky-Golay Function
def Rangers_SG(path, park, class_input, start_year, end_year):

    data = pd.read_csv(path + 'knps_final.csv')  # 데이터 불러오기
    data = data[(data['code'] == park) & (data['class'] == class_input)]  # 국립 공원과 산림 선정

    each_year_list = []
    for each_date in data['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    data['Year'] = each_year_list

    data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]  # 보고 싶은 연도 설정 반영

    data['DOY'] = data['date'].apply(lambda x: int(format(datetime.strptime(x, '%Y-%m-%d'), '%j')))
    data['DOY_CUM'] = data['DOY'] + 365 * (data['Year'] - data['Year'].iloc[0])  # 첫 번째 연도 기준으로 DOY 누적 게산
    data.index = data['DOY_CUM']

    # Scipy 내부에 있는 savgol_filter 활용
    data['smoothed_1dg'] = savgol_filter(data.avg, window_length=5, polyorder=1)

    data = data.loc[:, ['code', 'class', 'date', 'avg', 'DOY', 'smoothed_1dg']]

    return data.smoothed_1dg


# Gaussian Function
def Rangers_GSN(path, park, class_input, start_year, end_year):

    data = pd.read_csv(path + 'knps_final.csv')  # 데이터 불러오기
    data = data[(data['code'] == park) & (data['class'] == class_input)]  # 국립 공원과 산림 선정

    each_year_list = []
    for each_date in data['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    data['Year'] = each_year_list

    data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]  # 보고 싶은 연도 설정 반영

    data['DOY'] = data['date'].apply(lambda x: int(format(datetime.strptime(x, '%Y-%m-%d'), '%j')))
    data['DOY_CUM'] = data['DOY'] + 365 * (data['Year'] - data['Year'].iloc[0])  # 첫 번째 연도 기준으로 DOY 누적 게산
    data.index = data['DOY_CUM']

    # Gaussian Filtering
    kernel1d = cv2.getGaussianKernel(46*(end_year - start_year + 1), 1)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    data['Gaussian'] = cv2.filter2D(np.array(data.avg), -1, kernel2d).reshape(-1).tolist() # convolve

    data = data.loc[:, ['code', 'class', 'date', 'avg', 'DOY', 'Gaussian']]

    return data.Gaussian






