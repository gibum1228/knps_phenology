import io
import os
import platform

import cv2 as cv
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io
from PIL import Image


# ROOT 정보와 슬래쉬(MIDDLE) 정보 가져오기
def get_info() -> (str, str):
    middle = "/" if platform.system() != "Windows" else "\\"  # 윈도우: "\\" and 리눅스,유닉스: "/"
    index = os.getcwd().find('knps_phenology')  # knps_phenology(root 프로젝트 폴더)가 있는 인덱스 찾기
    root = os.getcwd()[:index + 14]  # root = "*/knps_phenology"

    return root, middle


# 최종 통합 데이터 파일에서 전체 데이터 또는 원하는 국립공원 산림의 데이터 로드하기
def get_final_data(knps: str = "", class_num: str = "", all: bool = False) -> pd.DataFrame:
    df = pd.read_csv(f"{ROOT}{MIDDLE}data{MIDDLE}knps_final.csv")  # csv 파일 로드

    # 전체 데이터를 추출할지 여부 판단
    if all:
        return df  # 전체 데이터 반환
    else:
        # 조건에 맞는 데이터만 반환
        return df[(df["code"] == knps) & (df["class"] == int(class_num))].sort_values('date')


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


# 다양한 curve fitting 알고리즘을 key으로 선택 가능
def curve_fit(y, ori_db: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    start_year = int(ori_db['start_year'])
    end_year = int(ori_db['end_year'])

    if ori_db['curve_fit'] == '1':
        data_final, sos_df = analysis.Rangers_DL(y, start_year, end_year, ori_db)
    elif ori_db['curve_fit'] == '2':
        data_final, sos_df = analysis.Rangers_SG(y, start_year, end_year, ori_db)
    elif ori_db['curve_fit'] == '3':
        data_final, sos_df = analysis.Rangers_GSN(y, start_year, end_year, ori_db)

    return data_final, sos_df


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
