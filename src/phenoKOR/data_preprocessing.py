import os
import platform
import io
import cv2 as cv
import numpy as np
import pandas as pd
import scipy.io
from PIL import Image
import phenoKOR as pk


# root 정보와 슬래쉬 정보 가져오기
def get_info():
    middle = "/" if platform.system() != "Windows" else "\\"  # 운영체제에 따라 슬래쉬 설정
    index = os.getcwd().find('knps_phenology') # root 디렉토리
    root = os.getcwd()[:index + 15] # root 디렉토리까지 자르기

    return root, middle


# 최종 통합 데이터 파일에서 원하는 국립공원 산림 데이터 로드하기
def load_final_data(knps, class_num):
    df = pd.read_csv(f"{root}data{middle}knps_final.csv")  # 데이터 가져오기

    # 조건에 맞는 데이터 추출
    return df[(df["code"] == knps) & (df["class"] == class_num)].sort_values('date')


# 마스크 정보가 담긴 mat 파일 가져오기
def load_mask(path, filename):
    mask = scipy.io.loadmat(path + filename) # scipy.io

    return mask['roiimg'] # mat 파일을 호출하면 dict type인데 마스크 정보를 담고 있는 'roiimg'를 return


# 이미지 데이터에서 관심영역 추출하기
def load_roi(img, mask):
    new_img = np.zeros_like(img) # 검은 배경만 가지는 이미지 데이터 생성

    new_img[mask > 0] = img[mask > 0] # 관심영역 부분의 픽셀만 가져와 적용

    return new_img


# 로컬 경로에서 이미지 데이터 가져오기
def load_image_for_local(path):
    info = {} # 정보가 담긴 딕셔너리
    ori_df = pd.DataFrame() # 정보를 담을 데이터프레임
    columns = ["date", "year", "month", "day", "code", "time", "imgs", "rcc", "gcc"]

    for key in columns: # 속성명으로 딕셔너리 초기화
        info[f"{key}"] = []

    for month in sorted(os.listdir(path)):
        if month.count(".") != 0: continue
        print(f"start {month}")
        for img_name in sorted(os.listdir(path + "/" + month)):
            c, y, m, dd, t, *_ = img_name.replace(".", "_").split("_") # 정보 추출(코드, 년도, 월, 일, 시간)
            img = cv.imread(path + "/" + month + "/" + img_name, cv.IMREAD_COLOR) # img_name인 이미지 호출
            for k, v in [("date", "-".join([y, m, dd])), ("year", y), ("month", m), ("day", dd), ("code", c), ("time", t), ("imgs", img)]: # (키, 값)
                info[f"{k}"].append(v)

    for key in columns: # 데이터프레임에 값 넣기
        ori_df[f"{key}"] = info[f"{key}"]

    return ori_df, np.array(info["imgs"])


# 웹에서 이미지 데이터 가져오기
def load_image_for_web(folder, img_mask): # folder: dictionary, img_mask: ndarray
    # 입력 받은 이미지 정보를 저장할 데이터 프레임
    df = pd.DataFrame(columns=["date", "year", "month", "day", "code", "rcc", "gcc"])

    # 입력 받은 파일 갯수만큼
    for i in range(len(folder)):
        img_info = folder[i] # i번째 이미지 정보 받기

        filename = img_info.open().split('.')[0] # 파일 이름
        code, year, month, day, *_ = filename.split('_') # 파일 이름에서 정보 추출

        img_byte = img_info.read() # byte 파일 받기
        data = io.BytesIO(img_byte) # 디코딩 적용
        img_pil = Image.open(data) # 이미지 데이터로 만들기
        img_numpy = np.array(img_pil) # ndarray로 형변환
        img = cv.cvtColor(img_numpy, cv.COLOR_RGB2BGR) # opencv에 맞게 RGB -> BGR로 변경

        # 이미지와 마스크로 관심영역만 값이 남아 있는(활성화 된) 새로운 이미지 추출
        img_roi = load_roi(img, img_mask)

        # rcc, gcc 값 구하기
        rcc, gcc = pk.get_cc(img_roi)

        # 데이터프레임에 행 추가
        df.loc[len(df)] = [pd.to_datetime(f"{year}-{month}-{day}", format="%Y-%m-%d"),
                           year, month, day, code, rcc, gcc]

    return df


# csv 데이터 가져오기
def load_csv(array_buffer):
    pass


# 윤년 구하는 메소드
def get_Feb_day(year):
    # 4, 100, 400으로 나누어 떨어진다면 윤년
    if year % 4 == 0 or year % 100 == 0 or year % 400 == 0:
        day = 29
    else:
        day = 28

    return day


# 국립공원 영어 이름을 로드하기
def get_knps_name_EN():
    name = ["gaya", "gyeongju", "gyeryong", "naejang", "dadohae", "deogyu", "mudeung", "byeonsan", "bukhan", "seorak",
            "sobaek", "songni", "odae", "worak", "wolchul", "juwang", "jiri", "chiak", "taebaek", "taean", "halla", "hallyeo"]

    return name


# 국립공원 한글 이름을 로드하기
def get_knps_name_KR():
    name = ["가야산", "경주", "계룡산", "내장산", "다도해상", "덕유산", "무등산", "변신반도", "북한산", "설악산", "소백산", "속리산", "오대산",
            "월악산", "월출산", "주왕산", "지리산", "치악산", "태백산", "태안해상", "한라산", "한려해상"]

    return name


# 국립 공원 위도, 경도 로드하기
def get_knps_position():
    # 국립공원명과 대치되는 (위도, 경도)
    position = [(35.779385, 128.122559), (35.867430, 129.222565), (36.356057, 127.212067), (35.483333, 126.883333), (34.708203, 125.901489), (35.824494, 127.787476),
                (35.134127, 126.988756), (35.680893, 126.531392), (37.70338, 127.032166), (38.1573652, 128.4355274), (36.909725, 128.459374), (36.533333, 127.9),
                (37.724030, 128.599777), (36.852005, 128.197261), (34.757310, 126.680823), (36.402218, 129.187889), (35.333333, 127.716667), (37.37169, 128.050509),
                (37.0545, 128.916666), (36.78712, 126.143475), (33.366667, 126.533333), (34.75882, 127.97596)]

    return position

# 전역변수
root, middle = get_info()
