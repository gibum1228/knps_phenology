import os
import platform
import cv2 as cv
import numpy as np
import pandas as pd
import scipy.io as io


# root 정보와 슬래쉬 정보 가져오기
def get_info():
    middle = "/" if platform.system() != "Windows" else "\\"  # 운영체제에 따라 슬래쉬 설정
    root = os.getcwd() + middle # 현재 프로젝트 폴더 위치

    return root, middle


# 최종 통합 데이터 파일에서 원하는 국립공원 산림 데이터 로드하기
def load_final_data(knps, class_num):
    df = pd.read_csv(f"{root}data{middle}knps_final.csv")  # 데이터 가져오기

    # 조건에 맞는 데이터 추출
    return df[(df["code"] == knps) & (df["class"] == class_num)].sort_values('date')


# 마스크 정보가 담긴 mat 파일 가져오기
def load_mask(path, filename):
    mask = io.loadmat(path + filename) # scipy.io

    return mask['roiimg'] # mat 파일을 호출하면 dict type인데 마스크 정보를 담고 있는 'roiimg'를 return


# 이미지 데이터에서 관심영역 추출하기
def load_roi(imgs, mask):
    new_imgs = np.zeros_like(imgs) # 검은 배경만 가지는 이미지 데이터 생성

    new_imgs[mask > 0] = imgs[mask > 0] # 관심영역 부분의 픽셀만 가져와 적용

    return new_imgs


# 이미지 데이터 가져오기
def load_image(path):
    info = {} # 정보가 담긴 딕셔너리
    ori_df = pd.DataFrame() # 정보를 담을 데이터프레임

    for key in ["date", "year", "month", "day", "code", "time", "imgs"]: # 속성명
        info[f"{key}"] = []

    for month in sorted(os.listdir(path)):
        if month.count(".") != 0: continue
        print(f"start {month}")
        for img_name in sorted(os.listdir(path + "/" + month)):
            c, y, m, dd, t, *_ = img_name.replace(".", "_").split("_") # 정보 추출(코드, 년도, 월, 일, 시간)
            img = cv.imread(path + "/" + month + "/" + img_name, cv.IMREAD_COLOR) # img_name인 이미지 호출
            for k, v in [("date", "-".join([y, m, dd])), ("year", y), ("month", m), ("day", dd), ("code", c), ("time", t), ("imgs", img)]: # (키, 값)
                info[f"{k}"].append(v)

    for key in ["date", "year", "month", "day", "code", "time"]: # 속성명
        ori_df[f"{key}"] = info[f"{key}"]

    return ori_df, np.array(info["imgs"])


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
            "sobaek", "songni", "odae", "worak", "wolchul", "juwang", "jiri", "chiak", "taebaek", "taean", "halla",
            "hallyeo"]

    return name


# 국립공원 한글 이름을 로드하기
def get_knps_name_KR():
    name = ["가야산", "경주", "계룡산", "내장산", "다도해상", "덕유산", "무등산", "변신반도", "북한산", "설악산", "소백산", "속리산", "오대산",
            "월악산", "월출산", "주왕산", "지리산", "치악산", "태백산", "태안해상", "한라산", "한려해상"]

    return name


# 전역변수
root, middle = get_info()
