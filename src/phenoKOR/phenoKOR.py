'''
phenoKOR 패키지
'''
import os
import cv2 as cv
import pandas as pd
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

'''
불러오는 메소드 모음
'''
# 관심영역 정보가 담긴 파일 가져오기
def load_mask(path, filename):
    mask = io.loadmat(path + filename) # scipy.io

    return mask['roiimg'] # mat 파일을 호출하면 dict type인데 마스크 정보를 담고 있는 'roiimg'를 return


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


# 이미지 데이터에서 관심영역 추출하기
def load_roi(imgs, mask):
    new_imgs = np.zeros_like(imgs) # 검은 배경만 가지는 이미지 데이터 생성

    new_imgs[mask > 0] = imgs[mask > 0] # 관심영역 부분의 픽셀만 가져와 적용

    return new_imgs


# 연도 데이터 가져오기
def load_csv(path):
    df = pd.read_csv(path)

    x = [i for i in range(9, 365, 16)] # 아쿠아는 9일부터 16일씩
    # 클래스별 일일 평균값 불러오기
    y0 = df[df["class"] == 0]["avg"].to_list()
    y1 = df[df["class"] == 1]["avg"].to_list()
    y2 = df[df["class"] == 2]["avg"].to_list()
    y3 = df[df["class"] == 3]["avg"].to_list()

    return x, y0, y1, y2, y3


# csv 파일에서 이미지 가져오기
def load_image_for_csv():
    pass


'''
저장 메소드 모음
'''
# 이미지 데이터를 csv 파일로 저장
def save_img_to_csv(path, filename, imgs):
    pass


# 관심영역 이미지 데이터를 csv 파일로 저장
def save_roi_to_csv(path, filename, imgs):
    pass


# 정제된 gcc 값이 담긴 DataFrame을 csv 파일로 저장
def save_gcc_to_csv(path, filename, df):
    pass


'''
연산 메소드 모음
'''
def get_cc(imgs, key):
    # b, g, r 순서로 채널별 값 구하기
    red_dn = imgs[:, :, 2]
    blue_dn = imgs[:, :, 0]
    green_dn = imgs[:, :, 1]

    # 분모에 해당하는 레드 + 블루 + 그린 색상값 더하기
    bunmo = red_dn + blue_dn + green_dn

    # 각각의 Chromatic Coordinate 값 구하기
    red_cc = red_dn / bunmo
    green_cc = green_dn / bunmo
    blue_cc = blue_dn / bunmo

    '''
    원하는 값만 뽑고 싶으면 key를 r, g, b 중 지정,
    전부 다 뽑고 싶다면 key를 지정하지 않음
    '''
    if key == "r": # r_cc
        return red_cc
    elif key == "g": # g_cc
        return green_cc
    elif key == "b": # b_cc
        return blue_cc
    else: # all
        return red_cc, green_cc, blue_cc


# 다양한 curve fitting 알고리즘을 key으로 선택 가능
def curve_fit(y, key):
    pass


'''
그래프 시각화 메소드 모음
'''
# plot 그리기
def show_plot(y, x = []):
    plt.figure(figsize=(10, 10))
    plt.plot(x, y) if x else plt.plot(y)
    plt.show()