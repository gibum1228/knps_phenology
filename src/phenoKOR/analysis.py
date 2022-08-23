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


def C_Phenophase(path, park, class_input, year, how):
    '''
    :param path: where the csv comes from
    :param park: must be string type
    :param class_input: must be string type
    :param year : year you want to see. type : int
    :param how : which method you're gonna use, it should be in ['sos', 'diff50%', 'diff40%']
    :return: Phenophase which and when you are interested in
    '''

    park_type = ['bukhan', 'byeonsan', 'chiak', 'dadohae', 'deogyu', 'gaya', 'gyeongju', 'gyeryong', 'halla',
                 'hallyeo', 'jiri', 'juwang', 'mudeung', 'naejang', 'odae', 'seorak', 'sobaek', 'songni',
                 'taean', 'taebaek', 'wolchul', 'worak']
    class_type = ['grassland', 'coniferous', 'broadleaved', 'mixed']
    years = list(range(2003, 2022))

    if park in park_type:
        if class_input in class_type:
            if year in years:
                data = pd.read_csv(path + f'{park}/{park}_DL_{class_input}_datetime.csv')
                sos_df = pd.read_csv(path + f'sos/{park}_sos_{class_input}_last.csv')
            else:
                print(f'{year} is unavailable')  # 해당 연도 판별
        else:
            print(f'{class_input} is not correct')  # 해당 산림 있는지 판별
    else:
        print(f'{park} is not correct')  # 해당 국립공원 판별

    if how == 'sos':
        sos_df.columns = ['year', 'sos']

        phenophase = sos_df[sos_df['year'] == year].iloc[0,1]  # sos 스칼라 값
        phenophase = format(datetime.datetime.strptime(str(year)+str(phenophase), '%Y%j'), '%Y-%m-%d')
        phenophase_betw = ""  # 다른 조건들은 사이값 추출하니까 이 경우 빈칸으로 설정

    if how == 'diff40%':
        data = data[data['Datetime'].str[:4] == str(year)]

        thresh = np.min(data['avg']) + (np.max(data['avg']) - np.min(data['avg'])) * 0.4 ## 개엽일 EVI 값

        ## 개엽일 사이값 찾기
        high = data[data['avg'] >= thresh]['Datetime'].iloc[0]
        low = data.Datetime[[data[data['avg'] >= thresh]['Datetime'].index[0] - 1]].iloc[0]
        high_value = data[data['Datetime'] == high].iloc[0, 2]  ## high avg 값만 추출
        low_value = data[data['Datetime'] == low].iloc[0, 2]  ## low avg 값만 추출
        div_add = (high_value - low_value) / 8

        for a in range(8):
            if low_value > thresh:
                break
            else:
                low_value += div_add

        phenophase = format(pd.to_datetime(low) + datetime.timedelta(days=a-1), '%Y-%m-%d')
        phenophase_betw = f'Phenophase is between {low} and {high}'

    if how == 'diff50%':
        data = data[data['Datetime'].str[:4] == str(year)]

        thresh = np.min(data['avg']) + (np.max(data['avg']) - np.min(data['avg'])) * 0.5 ## 개엽일 EVI 값

        ## 개엽일 사이값 찾기
        high = data[data['avg'] >= thresh]['Datetime'].iloc[0]
        low = data.Datetime[[data[data['avg'] >= thresh]['Datetime'].index[0] - 1]].iloc[0]
        high_value = data[data['Datetime'] == high].iloc[0, 2]  ## high avg 값만 추출
        low_value = data[data['Datetime'] == low].iloc[0, 2]  ## low avg 값만 추출
        div_add = (high_value - low_value) / 8

        for a in range(8):
            if low_value > thresh:
                break
            else:
                low_value += div_add

        phenophase = format(pd.to_datetime(low) + datetime.timedelta(days=a-1), '%Y-%m-%d')
        phenophase_betw = f'Phenophase is between {low} and {high}'

    return phenophase, phenophase_betw


if __name__ == "__main__":
    root = "/Users/beom/Desktop/git/data/knps/"

    check_stationarity(root + "day_8_data/2021_jiri_final.csv")

