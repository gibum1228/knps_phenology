# 메서드
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import cv2
import warnings
warnings.filterwarnings('ignore')

# Double Logistic 함수 정의
def Rangers_DL(input_data, start_year, end_year):
    '''
    :param Data : input DataFrame
    :param start_year, end_year
    :return: Double Logistic DataFrame, SOS DataFrame
    '''

    data = input_data


    data_final = pd.DataFrame(columns=['code', 'class', 'date', 'avg', 'DOY'])  # return 할 데이터 프레임
    sos_list = []  # return 할 SOS 값

    each_year_list = []
    for each_date in data['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    data['Year'] = each_year_list  # date 칼럼에서 앞 4개 문자만 추출한 것을 Year 칼럼 추가
    data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]  # 보고 싶은 연도 설정 반영

    data['DOY'] = data['date'].apply(lambda x: int(format(datetime.strptime(x, '%Y-%m-%d'), '%j')))  # DOY 칼럼 추가
    data['DOY_CUM'] = data['DOY'] + 365 * (data['Year'] - data['Year'].iloc[0])  # 시작 연도 기준 DOY 누적 게산


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
        xpred_dl = mn + (mx - mn) * (1 / (1 + np.exp(-rsp * (t - sos))) + 1 / (1 + np.exp(rau * (t - eos))) - 1)  # 더블 로지스틱 수식
        return pd.Series(xpred_dl, index=t)

    # SSE 함수 정의
    def SSE(par, x, weights):
        if par[0] > par[1]:  # min EVI가 max EVI 보다 클 경우 99999 반환
            return 99999
        xpred_sse = DoubleLogistic(par, t=t)
        sse = np.sum(((xpred_sse - x) ** 2) * weights)
        return sse

    # 연도별 Double Logistic 적용 위한 반복문
    for each_year in range(start_year, end_year+1):

        data_re = data[data['Year'] == each_year]
        data_re.index = data_re['DOY']
        data_re = data_re.loc[:, ['code', 'class', 'date', 'avg', 'DOY']]

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
    sos_df['Year'] = [i for i in range(start_year, end_year+1)]
    sos_df['sos_DOY'] = sos_list

      # 누적 DOY를 반환할 데이터 프레임 인덱스로 설정

    return data_final, sos_df


# Savitzky-Golay Function
def Rangers_SG(data_input, start_year, end_year):
    '''
    :param data_input : input DataFrame
    :param start_year, end_year
    :return: Savitzky-Golay DataFrame
    '''

    data = data_input  # 입력받은 데이터 data에 저장

    each_year_list = []
    for each_date in data['date']:
        each_date = int(each_date[:4])
        each_year_list.append(each_date)
    data['Year'] = each_year_list  # date 칼럼에서 앞 4개 문자만 추출한 것을 Year 칼럼 추가
    data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]  # 보고 싶은 연도 설정 반영

    data['DOY'] = data['date'].apply(lambda x: int(format(datetime.strptime(x, '%Y-%m-%d'), '%j')))  # DOY 칼럼 추가
    data['DOY_CUM'] = data['DOY'] + 365 * (data['Year'] - data['Year'].iloc[0])  # 시작 연도 기준 DOY 누적 게산
    data.index = data['DOY_CUM']  # 누적 DOY를 반환할 데이터 프레임 인덱스로 설정

    # Scipy 내부에 있는 savgol_filter 활용
    data['avg'] = savgol_filter(data.avg, window_length=5, polyorder=1)

    data = data.loc[:, ['code', 'class', 'date', 'avg', 'DOY']]

    sos_df = [0]
    return data, sos_df


# Gaussian Function
def Rangers_GSN(data_input, start_year, end_year):
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
    data['DOY_CUM'] = data['DOY'] + 365 * (data['Year'] - data['Year'].iloc[0])  # 시작 연도 기준 DOY 누적 게산
    data.index = data['DOY_CUM']  # 누적 DOY를 반환할 데이터 프레임 인덱스로 설정

    # Gaussian Filtering
    kernel1d = cv2.getGaussianKernel(46*(end_year - start_year + 1), 1)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    data['avg'] = cv2.filter2D(np.array(data.avg), -1, kernel2d).reshape(-1).tolist() # convolve

    data = data.loc[:, ['code', 'class', 'date', 'avg', 'DOY']]
    sos_df = [0]
    return data, sos_df