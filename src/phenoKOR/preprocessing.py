import io
import os
import platform

import cv2 as cv
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io
from PIL import Image
from datetime import datetime
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression


# ROOT 정보와 슬래쉬(MIDDLE) 정보 가져오기
def get_info() -> (str, str):
    middle = "/" if platform.system() != "Windows" else "\\"  # 윈도우: "\\" and 리눅스,유닉스: "/"
    index = os.getcwd().find('knps_phenology')  # knps_phenology(root 프로젝트 폴더)가 있는 인덱스 찾기
    root = os.getcwd()[:index + 14]  # root = "*/knps_phenology"

    return root, middle


# 최종 통합 데이터 파일에서 전체 데이터 또는 원하는 국립공원 산림의 데이터 로드하기
def get_final_data(db: dict = None, all: bool = False) -> pd.DataFrame:
    df = pd.read_csv(f"{ROOT}{MIDDLE}data{MIDDLE}knps_final.csv")  # csv 파일 로드

    # 전체 데이터를 추출할지 여부 판단
    if all:
        return df  # 전체 데이터 반환
    else:
        # 조건에 맞는 데이터만 반환
        return df[(df["code"] == db['knps']) & (df["class"] == int(db['class_num']))
                  & (df['date'].str[:4] >= db["start_year"]) & (df['date'].str[:4] <= db["end_year"])].sort_values(
            'date')


# mat 파일에서 마스크 이미지 생성하기
def mat2image(path: str, filename: str) -> None:
    mask = get_mask_for_mat(path, filename)  # mat 파일 읽기
    new_img = np.zeros((mask.shape[0], mask.shape[1]))  # 검정색 밑바탕 이미지 생성

    # 마스크 개수만큼 (3차원 크기만큼)
    for channel in range(mask.shape[2]):
        new_img = np.sum([new_img, np.where(mask[:, :, channel] > 0, 255, 0)], axis=0)  # 값 합치기

    cv.imwrite(f"{ROOT}{MIDDLE}data{MIDDLE}{filename.split('.')[0]}_mask.png", new_img)  # data 폳더에 저장


# mat 파일 읽기 [기존 분석 프로그램(PhenoCam GUI)에서는 ROI 정보를 mat 파일로 저장함]
def get_mask_for_mat(path: str, filename: str) -> npt.NDArray:
    mask = scipy.io.loadmat(path + MIDDLE + filename)  # mat 파일 로드

    return mask['roiimg']  # mat 파일을 호출하면 dictionary인데 마스크 정보를 담고 있는 'roiimg'를 반환


# 이미지 데이터에서 관심영역 추출하기
def get_roi(img: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    new_img = np.where(mask == 255, img, 0)  # 첫번째 매개변수 조건에 따라 참이면 값 유지, 거짓이면 0으로 변경

    return new_img


# 로컬 경로에서 이미지 데이터 가져오기
def get_image_for_local(path: str) -> (pd.DataFrame, npt.NDArray):
    info = {}  # 정보가 담긴 딕셔너리
    ori_df = pd.DataFrame()  # 정보를 담을 데이터프레임
    columns = ["date", "year", "month", "day", "code", "time", "imgs", "rcc", "gcc"]

    for key in columns:  # 속성명으로 딕셔너리 초기화
        info[f"{key}"] = []

    for month in sorted(os.listdir(path)):  # 월 정보가 있는 폴더에서
        if month.count(".") != 0: continue  # 디렉토리가 아니라면 skip

        for img_name in sorted(os.listdir(path + "/" + month)):  # 월 폴더에 진입
            c, y, m, dd, t, *_ = img_name.replace(".", "_").split("_")  # 정보 추출(코드, 년도, 월, 일, 시간)
            img = cv.imread(path + "/" + month + "/" + img_name, cv.IMREAD_COLOR)  # img_name인 이미지 호출
            for k, v in [("date", "-".join([y, m, dd])), ("year", y), ("month", m), ("day", dd), ("code", c),
                         ("time", t), ("imgs", img)]:  # (키, 값)
                info[f"{k}"].append(v)

    for key in columns:  # 데이터프레임에 값 넣기
        ori_df[f"{key}"] = info[f"{key}"]

    return ori_df, np.array(info["imgs"])  # 데이터 정보가 담긴 데이터프레임과 이미지 데이터 ndarray 반환


# 웹에서 이미지 데이터 가져오기
def get_image_for_web(folder: dict) -> (pd.DataFrame, npt.NDArray):
    columns = ['date', 'code', 'year', 'month', 'day']  # 파일 정보 저장에 사용할 key
    db = {f"{key}": [] for key in columns}  # list를 데이터프레임에 넣기 위해 key로 정보를 저장하는 db 생성
    db["imgs"] = []  # 이미지 데이터 정보

    # 입력 받은 파일 갯수만큼
    for file in folder:  # 파일 하나씩 연산하기
        filename = file.name  # 파일 이름 가져오기
        filename_split = filename.split('_')  # 파일 이름에서 정보 추출

        # db에 정보 추가
        for i in range(4):
            db[columns[i + 1]].append(filename_split[i])  # code, year, month, day 순으로 삽입

        # datetime 넣기
        db["date"].append(
            pd.to_datetime(f"{filename_split[1]}-{filename_split[2]}-{filename_split[3]}", format="%Y-%m-%d"))
        db["imgs"].append(byte2img(file.read()))  # 바이트 파일을 이미지로 변환해 리스트에 저장

    df = pd.DataFrame(columns=columns)  # 데이터 프레임 생성
    for key in columns:  # 데이터를 열로 넣기
        df[f"{key}"] = db[f"{key}"]

    return df, np.array(db["imgs"])


# 바이트 파일에서 이미지 파일로 변환
def byte2img(byte: bytes) -> npt.NDArray:
    decoding_data = io.BytesIO(byte)  # 디코딩 된 파일
    img_pil = Image.open(decoding_data)  # 이미지 데이터로 만들기
    img_numpy = np.array(img_pil)  # ndarray로 형변환
    img = cv.cvtColor(img_numpy, cv.COLOR_RGB2BGR)  # opencv에 맞게 RGB -> BGR로 변경

    return img  # 이미지 파일 반환


# Chromatic Coordinate 값 연산
def get_cc(img: npt.NDArray) -> (float, float):
    # b, g, r 순서로 채널별 값 구하기 (dn: digital number)
    red_dn = np.sum(img[:, :, 2])
    blue_dn = np.sum(img[:, :, 0])
    green_dn = np.sum(img[:, :, 1])

    # 분모에 해당하는 레드 + 블루 + 그린 색상값 더하기
    bunmo = red_dn + blue_dn + green_dn

    try:
        # rcc, gcc 값 구하기
        red_cc = red_dn / bunmo
        green_cc = green_dn / bunmo
    except ZeroDivisionError:  # 분모가 0일 때 에러 처리하기 위해 try-except 작성
        print("변수 bunmo가 0입니다.")
        return 0., 0.

    return red_cc, green_cc


# 다양한 curve fitting 알고리즘을 선택해 데이터 전처리하기
def curve_fit(df: pd.DataFrame, ori_db: dict) -> (pd.DataFrame, pd.DataFrame):
    start_year = int(ori_db['start_year'])
    end_year = int(ori_db['end_year'])

    if ori_db['curve_fit'] == '1':
        after_df, df_sos = double_logistic_func(df, start_year, end_year, ori_db)
    elif ori_db['curve_fit'] == '2':
        after_df, df_sos = savitzky_golay_func(df, start_year, end_year, ori_db)
    elif ori_db['curve_fit'] == '3':
        after_df, df_sos = gaussian_func(df, start_year, end_year, ori_db)

    return after_df, df_sos


# Double Logistic 함수 정의
def double_logistic_func(input_data, start_year, end_year, ori_db):
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
def savitzky_golay_func(data_input, start_year, end_year, ori_db):
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
def gaussian_func(data_input, start_year, end_year, ori_db):
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
    kernel1d = cv.getGaussianKernel(46 * (end_year - start_year + 1), 1)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    data['avg'] = cv.filter2D(np.array(data.avg), -1, kernel2d).reshape(-1).tolist()  # convolve

    if ori_db['AorP'] == 'A':
        data = data.loc[:, ['code', 'class', 'date', 'avg', 'DOY']]
    else:
        data = data.loc[:, ['date', 'avg', 'DOY']]

    sos_df = [0]
    return data, sos_df


# 국립공원 영어 이름을 가져오기
def get_knps_name_en() -> list:
    name = ["gaya", "gyeongju", "gyeryong", "naejang", "dadohae", "deogyu", "mudeung", "byeonsan", "bukhan", "seorak",
            "sobaek", "songni", "odae", "worak", "wolchul", "juwang", "jiri", "chiak", "taebaek", "taean", "halla",
            "hallyeo"]

    return name


# 국립공원 한글 이름을 가져오기
def get_knps_name_kr() -> list:
    name = ["가야산", "경주", "계룡산", "내장산", "다도해상", "덕유산", "무등산", "변신반도", "북한산", "설악산", "소백산", "속리산", "오대산",
            "월악산", "월출산", "주왕산", "지리산", "치악산", "태백산", "태안해상", "한라산", "한려해상"]

    return name


# 국립 공원 위도, 경도 가져오기
def get_knps_position() -> list:
    # 국립공원명과 대치되는 (위도, 경도)
    position = [(35.779385, 128.122559), (35.867430, 129.222565), (36.356057, 127.212067), (35.483333, 126.883333),
                (34.708203, 125.901489), (35.824494, 127.787476),
                (35.134127, 126.988756), (35.680893, 126.531392), (37.70338, 127.032166), (38.1573652, 128.4355274),
                (36.909725, 128.459374), (36.533333, 127.9),
                (37.724030, 128.599777), (36.852005, 128.197261), (34.757310, 126.680823), (36.402218, 129.187889),
                (35.333333, 127.716667), (37.37169, 128.050509),
                (37.0545, 128.916666), (36.78712, 126.143475), (33.366667, 126.533333), (34.75882, 127.97596)]

    return position


# 전역변수
ROOT, MIDDLE = get_info()
