from django.views.decorators.clickjacking import xframe_options_exempt
from django.shortcuts import render
from folium import plugins
import folium
import base64
import pandas as pd
import datetime
from pathlib import Path
import sys

from django.views.decorators.csrf import csrf_exempt
from prophet.serialize import model_from_json
import cv2 as cv
import numpy as np

# 다른 패키지에 있는 모듈 가져오기
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ../../knps_phenology/src
if sys.path.count(f"{BASE_DIR}/phenoKOR") == 0: sys.path.append(f"{BASE_DIR}/phenoKOR")
if sys.path.count(f"{BASE_DIR}\\phenoKOR") == 0: sys.path.append(f"{BASE_DIR}\\phenoKOR")
import preprocessing
import analysis as als

# 전역변수
ROOT, MIDDLE = preprocessing.get_info()
knps_final = preprocessing.get_final_data("", "", True)
property_list = ["knps", "curve_fit", "start_year", "end_year", "class_num", "threshold", "shape",
                 "AorP"]  # 리퀘스트 메소드로 주고 받을 속성명들


# 홈 페이지
@xframe_options_exempt  # iframe 허용하기 위한 태그
def index(request):
    m = folium.Map(location=[36.684273, 128.068635], zoom_start=6.5, width="100%", height="100%")  # 기본이 되는 지도 정보 가져오기

    name = preprocessing.get_knps_name_en()
    position = preprocessing.get_knps_position()

    # 국립공원 수만큼 반복
    for i in range(len(name)):
        # 국립공원 대표 이미지 가져오기
        pic = base64.b64encode(open(f'../resource/{name[i]}.png', 'rb').read()).decode()
        # 국립공원 대표 이미지 클릭시 해당 분석 페이지로 이동하는 html 태그 설정
        image_tag = f'<div style="text-align:center; "><a href="http://127.0.0.1:8000/analysis/?knps={name[i]}&start_year=2003&end_year=2003&class_num=0&curve_fit=1&shape=1&threshold=0.4" target="_top"><img src="data:image/png;base64,{pic}" width="200" height="150"></a></div>'
        # iframe 생성
        iframe = folium.IFrame(image_tag, width=220, height=170)
        # html 띄울 popup 객체 생성
        popup = folium.Popup(iframe)

        # 지도에 마커 찍기
        folium.Marker(position[i],
                      popup=popup,
                      icon=folium.Icon(color='green', icon='fa-tree', prefix='fa')).add_to(m)

    plugins.LocateControl().add_to(m)  # 위치 컨트롤러 추가

    maps = m._repr_html_()  # 지도를 템플릿에 삽입하기위해 iframe이 있는 문자열로 반환 (folium)

    return render(request, 'map/index.html', {'map': maps})  # index.html에 map 변수에 maps 값 전달하기


# 분석 페이지
def analysis(request):
    db = {}  # 데이터를 저장하고 페이지에 넘겨 줄 딕셔너리

    if request.method == 'GET':  # GET 메소드로 값이 넘어 왔다면,
        for key in property_list:
            # 값이 넘어 오지 않았다면 "", 값이 넘어 왔다면 해당하는 값을 db에 넣어줌
            db[f"{key}"] = request.GET[f"{key}"] if request.GET.get(f"{key}") else ""  # 삼항 연산자

    # shape 값(연속, 연도)에 따라 그래프를 그려줌
    db['graph'] = als.show_graph(db, 0) if db['shape'] == "1" else als.show_graphs(db, 0)
    # *shape는 default: 1임
    db['dataframe'] = export_doy(db)

    # *threshold는 default : 50

    return render(request, 'map/analysis.html', db)  # 웹 페이지에 값들 뿌려주기


# 예측 페이지
def predict(request):
    db = {}

    if request.method == 'GET':
        for key in property_list:
            db[f"{key}"] = request.GET[f"{key}"] if request.GET.get(f"{key}") else ""

    db['graph'] = als.show_graph(db, 1) if db['shape'] == "1" else als.show_graphs(db, 1)
    db['dataframe'] = predict_export_doy(db)

    return render(request, 'map/predict.html', db)


# 페노캠 이미지 분석하는 페이지
@csrf_exempt
def phenocam(request):
    db = {}

    if request.method == 'GET':
        for key in property_list:
            db[f"{key}"] = request.GET[f"{key}"] if request.GET.get(f"{key}") else ""

    if request.method == 'POST':
        for key in property_list:
            db[f"{key}"] = request.POST[f"{key}"] if request.POST.get(f"{key}") else ""
        # if request.FILES:  # input[type=file]로 값이 넘어 왔다면,
        #     request_dict = dict(request.FILES)  # FILES 객체를 딕셔너리로 변환
        #
        #     # 이미지 가져오기
        #     if request_dict['imgs']:
        #         # 바이트 데이터에서 파일 정보를 추출하고 이미지 데이터로 변환
        #         df, imgs = preprocessing.get_image_for_web(request_dict['imgs'])
        #
        #         # 마스크 이미지 가져오기
        #         if request_dict['img_mask']:
        #             img_mask = preprocessing.byte2img(request_dict['img_mask'][0].read())
        #             img_mask = cv.resize(img_mask, (imgs[0].shape[1], imgs[0].shape[0]))  # 캔버스 그릴 때 축소해서 원본 크기로 맞춤
        #             new_mask = np.where(img_mask == 255, img_mask, 0)  # 직접 그린 관심 영역 제외하고 검정색(0)으로 만들기
        #
        #     # 관심 영역 이미지 리스트
        #     imgs_roi = []
        #     for img in imgs:
        #         img_roi = preprocessing.get_roi(img, new_mask)  # 관심 영역 구하기
        #         imgs_roi.append(img_roi)
        #     imgs_roi = np.array(imgs_roi)
        #
        #     # 관심 영역 이미지에 대한 rcc, gcc 값 구하기
        #     rcc_list, gcc_list = [], []
        #     for img_roi in imgs_roi:
        #         rcc, gcc = preprocessing.get_cc(img_roi)
        #
        #         rcc_list.append(rcc)
        #         gcc_list.append(gcc)
        #     # list로 열 추가
        #     df['rcc'] = rcc_list
        #     df['gcc'] = gcc_list

        db['graph'] = als.show_graph(db) if db['shape'] == "1" else als.show_graphs(db)

    return render(request, 'map/phenocam.html', db)


def export_doy(ori_db):
    df = preprocessing.get_final_data(ori_db['knps'], ori_db['class_num'])
    df_sos = pd.read_csv(ROOT + f"{MIDDLE}data{MIDDLE}knps_final_sos.csv")
    df_sos = df_sos[['year', ori_db['knps'] + '_' + ori_db['class_num']]]
    df_sos.columns = ['year', 'sos']

    phenophase_date = ''
    phenophase_betw = ''

    sos = []
    doy = []
    betwn = []

    # sos 기준으로 개엽일 추출
    if ori_db['curve_fit'] == '1':
        # df, df_sos = pk.curve_fit(df, ori_db)
        # df_sos.columns = ['year', 'sos']
        df_sos = pd.read_csv(f"{ROOT}{MIDDLE}data{MIDDLE}knps_final_sos.csv")

        for year in range(int(ori_db['start_year']), int(ori_db['end_year']) + 1):
            # phenophase_doy = df_sos[df_sos['year'] == year]['sos'].to_list()[0]  # sos 스칼라 값
            phenophase_doy = df_sos[f"{ori_db['knps']}_{ori_db['class_num']}"].iloc[year - 2003]
            phenophase_date = (f'{year}년 : {phenophase_doy}일')
            sos.append(phenophase_date)

    else:
        df, df_sos = analysis.curve_fit(df, ori_db)

    for year in range(int(ori_db['start_year']), int(ori_db['end_year']) + 1):
        data = df[df['date'].str[:4] == str(year)]
        thresh = np.min(data['avg']) + ((np.max(data['avg']) - np.min(data['avg'])) * (
            float(ori_db["threshold"])))  ##개엽일의 EVI 값

        ## 개엽일 사이값 찾기
        high = data[data['avg'] >= thresh]['date'].iloc[0]
        low = data.date[[data[data['avg'] >= thresh].index[0] - 8]].to_list()[0]
        high_value = data.avg[data['date'] == high].to_list()[0]  ## high avg 값만 추출
        low_value = data.avg[data['date'] == low].to_list()[0]  ## low avg 값만 추출
        div_add = (high_value - low_value) / 8

        for a in range(8):
            if low_value > thresh:
                break
            else:
                low_value += div_add

        phenophase_doy = format(pd.to_datetime(low) + datetime.timedelta(days=a - 1), '%Y-%m-%d')
        phenophase_date = format(datetime.datetime.strptime(phenophase_doy, '%Y-%m-%d'), '%j') + '일,' + phenophase_doy
        phenophase_betw = (f'{low} ~ {high}')
        doy.append(phenophase_date)
        betwn.append(phenophase_betw)

    if ori_db['curve_fit'] == '1':
        total_DataFrame = pd.DataFrame(columns=['SOS기준 개엽일', '임계치 개엽일', '임계치 오차범위'])
        for i in range(len(doy)):
            total_DataFrame.loc[i] = [sos[i], doy[i], betwn[i]]
    else:
        total_DataFrame = pd.DataFrame(columns=['임계치 개엽일', '임계치 오차범위'])
        for i in range(len(doy)):
            total_DataFrame.loc[i] = [doy[i], betwn[i]]

    html_DataFrame = total_DataFrame.to_html(justify='center', index=False, table_id='mytable')

    return html_DataFrame


def open_model_processing(ori_db):
    with open(ROOT + f"{MIDDLE}data{MIDDLE}model{MIDDLE}{ori_db['knps']}_{ori_db['class_num']}", 'r') as fin:
        m = model_from_json(fin.read())

    periods = 4
    for i in range(int(ori_db['start_year']), int(ori_db['end_year']) + 1):

        if i % 4 == 1:
            periods += 366

        else:
            periods += 365

    future = m.make_future_dataframe(periods)
    forecast = m.predict(future)

    df = forecast[['ds', 'yhat']]
    df.columns = ['date', 'avg']
    df['date'] = df['date'].astype('str')
    doy_list = []
    for i in range(len(df)):
        date = df.loc[i, 'date']
        calculate_doy = datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:10])).strftime("%j")
        doy_list.append(calculate_doy)

    df['DOY'] = doy_list

    # df, df_sos = phenoKOR.curve_fit(df, ori_db)

    df = df[(df['date'].str[:4] >= ori_db['start_year'])]

    df = df.reset_index(drop=True)

    return (df)


def predict_export_doy(ori_db):
    with open(ROOT + f"{MIDDLE}data{MIDDLE}model{MIDDLE}{ori_db['knps']}_{ori_db['class_num']}", 'r') as fin:
        m = model_from_json(fin.read())

    periods = 0
    for i in range(int(ori_db['start_year']), int(ori_db['end_year']) + 1):
        if i % 4 == 0:
            periods += 370
        else:
            periods += 369

    future = m.make_future_dataframe(periods)
    forecast = m.predict(future)

    df = forecast[['ds', 'yhat']]
    df.columns = ['date', 'avg']
    df['date'] = df['date'].astype('str')
    doy_list = []
    for i in range(len(df)):
        date = df.loc[i, 'date']
        calculate_doy = datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:10])).strftime("%j")
        doy_list.append(calculate_doy)

    df['DOY'] = doy_list

    phenophase_date = ''
    phenophase_betw = ''

    sos = []
    doy = []
    betwn = []

    # sos 기준으로 개엽일 추출
    if ori_db['curve_fit'] == '1':
        # df, df_sos = pk.curve_fit(df, ori_db)
        # df_sos.columns = ['year', 'sos']
        df_sos = pd.read_csv(f"{ROOT}{MIDDLE}data{MIDDLE}knps_final_predict_sos.csv")

        for year in range(int(ori_db['start_year']), int(ori_db['end_year']) + 1):
            # phenophase_doy = df_sos[df_sos['year'] == year]['sos'].to_list()[0]  # sos 스칼라 값
            phenophase_doy = df_sos[f"{ori_db['knps']}_{ori_db['class_num']}"].iloc[year - 2022]
            phenophase_date = (f'{year}년 : {phenophase_doy}일')
            sos.append(phenophase_date)

    else:
        df, df_sos = analysis.curve_fit(df, ori_db)

    for year in range(int(ori_db['start_year']), int(ori_db['end_year']) + 1):
        data = df[df['date'].str[:4] == str(year)]
        thresh = np.min(data['avg']) + ((np.max(data['avg']) - np.min(data['avg'])) * (
            float(ori_db["threshold"])))  ##개엽일의 EVI 값

        ## 개엽일 사이값 찾기
        high = data[data['avg'] >= thresh]['date'].iloc[0]
        low = data.date[[data[data['avg'] >= thresh].index[0] - 8]].to_list()[0]
        high_value = data.avg[data['date'] == high].to_list()[0]  ## high avg 값만 추출
        low_value = data.avg[data['date'] == low].to_list()[0]  ## low avg 값만 추출
        div_add = (high_value - low_value) / 8

        for a in range(8):
            if low_value > thresh:
                break
            else:
                low_value += div_add

        phenophase_doy = format(pd.to_datetime(low) + datetime.timedelta(days=a - 1), '%Y-%m-%d')
        phenophase_date = format(datetime.datetime.strptime(phenophase_doy, '%Y-%m-%d'),
                                 '%j') + '일,' + phenophase_doy
        phenophase_betw = (f'{low} ~ {high}')
        doy.append(phenophase_date)
        betwn.append(phenophase_betw)

    if ori_db['curve_fit'] == '1':
        total_DataFrame = pd.DataFrame(columns=['SOS기준 개엽일', '임계치 개엽일', '임계치 오차범위'])
        for i in range(len(doy)):
            total_DataFrame.loc[i] = [sos[i], doy[i], betwn[i]]
    else:
        total_DataFrame = pd.DataFrame(columns=['임계치 개엽일', '임계치 오차범위'])
        for i in range(len(doy)):
            total_DataFrame.loc[i] = [doy[i], betwn[i]]

    html_DataFrame = total_DataFrame.to_html(justify='center', index=False, table_id='mytable')
    return html_DataFrame
