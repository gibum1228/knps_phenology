import io
from PIL import Image
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_exempt
import django
from django.shortcuts import render
from folium import plugins
import folium
import base64
from .fusioncharts import FusionCharts
from .fusioncharts import FusionTable
from .fusioncharts import TimeSeries
import json
import pandas as pd
import numpy as np
import datetime
import sys

sys.path.append("/Users/beom/Desktop/git/knps_phenology/src/phenoKOR")
sys.path.append("C:\\github\\bigleader\\knps_phenology\\src\\phenoKOR")
import cv2 as cv
import numpy as np
import data_preprocessing as dp
import matplotlib.pyplot as plt

# 전역변수
root, middle = dp.get_info()


# 홈 페이지
@xframe_options_exempt  # iframe 허용하기 위한 태그
def index(request):
    m = folium.Map(location=[36.684273, 128.068635], zoom_start=6.5, width="100%", height="100%")  # 기본이 되는 지도 정보 가져오기

    name = dp.get_knps_name_EN()
    position = dp.get_knps_position()

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
    property_list = ["knps", "curve_fit", "start_year", "end_year", "class_num", "threshold",
                     "shape"]  # GET 메소드로 주고 받을 변수 이름들
    db = {}  # 데이터를 저장하고 페이지에 넘겨 줄 딕셔너리

    if request.method == 'GET':  # GET 메소드로 값이 넘어 왔다면,
        for key in property_list:
            # 값이 넘어 오지 않았다면 "", 값이 넘어 왔다면 해당하는 값을 db에 넣어줌
            db[f"{key}"] = request.GET[f"{key}"] if request.GET.get(f"{key}") else ""  # 삼항 연산자

    db['graph'] = get_chart(db) if db['shape'] == "1" else get_multi_plot(db)  # shape 값(연속, 연도)에 따라 그래프를 그려줌
    # *shape는 default: 1임
    db['dataframe'] = export_doy(db)

    # *threshold는 default : 50

    return render(request, 'map/analysis.html', db)  # 웹 페이지에 값들 뿌려주기


# 예측 페이지
def predict(request):
    property_list = ["knps", "curve_fit", "start_year", "end_year", "class_num", "threshold"]
    db = {}

    if request.method == 'GET':
        for key in property_list:
            db[f"{key}"] = request.GET[f"{key}"] if request.GET.get(f"{key}") else ""

    db['graph'] = get_chart(db)  # 예측은 연속 그래프 고정

    return render(request, 'map/predict.html', db)


# 페노캠 이미지 분석하는 페이지
def phenocam(request):
    db = {}

    if request.method == 'POST' and request.FILES['folder']:
        image = dict(request.FILES)['folder']  # 폴더 내에 있는 모든 파일 가져오기
        images = image[0]

        print("===== images name =====")
        # print('myfile read:', images.read())  # file 읽기
        print('myfile size:', images.size)  # file 읽기
        print('myfile content_type:', images.content_type)
        print('myfile open:', images.open())
        myfile_read = images.read()
        print('myfile read type:', type(myfile_read))

        data_io = io.BytesIO(myfile_read)
        img_pil = Image.open(data_io)
        print(img_pil)
        numpy_image = np.array(img_pil)
        open_cv_image = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)
        print(open_cv_image.shape)

        print("=====")
        print(len(image))

        # encoding_img = np.fromstring(str(myfile_read), dtype=np.uint8)
        # print(encoding_img)
        # print(len(encoding_img))
        # print(type(encoding_img))
        # img = cv.imdecode(encoding_img, cv.IMREAD_COLOR)
        # print(img)

    return render(request, 'map/phenocam.html', db)


# 연속된 그래프를 그려주는 메소드
def get_chart(ori_db):
    df = pd.read_csv(root + f"data{middle}knps_final.csv")  # 데이터 가져오기
    df = df[df['class'] == int(ori_db['class_num'])]
    df = df[df['code'] == ori_db['knps']]

    data = []  # 그래프를 그리기 위한 데이터
    schema = [{"name": "Time", "type": "date", "format": "%Y-%m-%d"}, {"name": "EVI", "type": "number"}]  # 하나의 data 구조
    info_day = [None, 31, None, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 월별 일 수 정보
    year, month, day = 2003, 1, 1  # 년월일을 알기 위한 변수

    for _ in range(len(df)):  # data에 값 채우기
        data.append([f"{year}-{month}-{day}", df.iloc[len(data), 4]])  # schema 형태에 맞게 데이터 추가

        day += 8  # 8일 간격씩 데이터 추가
        if month == 2:  # 2월은 윤년 여부 판단하기
            day_limit = dp.get_Feb_day(year)
        else:  # 2월이 아니면 해당 월의 일 수 가져오기
            day_limit = info_day[month]

        # 다음 월로 넘어가야 한다면,
        if day > day_limit:
            day -= day_limit  # 새로운 일
            month += 1  # 다음 월로 가기

            if month > 12:  # 다음 연도로 넘어가야 한다면,
                year += 1
                # 무조건 1월 1일부터 시작하기 때문에 month와 day를 1로 초기화
                month = 1
                day = 1

    fusionTable = FusionTable(json.dumps(schema), json.dumps(data))  # 데이터 테이블 만들기
    timeSeries = TimeSeries(fusionTable)  # 타임시리즈 만들기

    # 그래프 속성 설정하기
    timeSeries.AddAttribute('caption', f'{{"text":"EVI of {ori_db["knps"]}"}}')
    timeSeries.AddAttribute('chart', f'{{"theme":"candy", "exportEnabled": "1", "exportfilename": "{ori_db["knps"]}_{ori_db["class_num"]}_{ori_db["start_year"]}_{ori_db["end_year"]}"}}')
    timeSeries.AddAttribute('subcaption', f'{{"text":"class_num : {ori_db["class_num"]}"}}')
    timeSeries.AddAttribute('yaxis', '[{"plot":{"value":"EVI"},"format":{"prefix":""},"title":"EVI"}]')

    # 그래프 그리기
    fcChart = FusionCharts("timeseries", "ex1", 960, 400, "chart-1", "json", timeSeries)

    # 그래프 정보 넘기기
    return fcChart.render()


# 연도별 그래프를 그려주는 메소드
def get_multi_plot(ori_db):
    df = pd.read_csv(root + f"data{middle}knps_final.csv")
    df = df[df['class'] == int(ori_db['class_num'])]
    df = df[df['code'] == ori_db['knps']]
    print(df)
    # curve fitting된 데이터 가져오기

    # 그래프 속성 및 데이터를 저장하는 변수
    db = {
        "chart": {  # 그래프 속성
            "exportEnabled": "1",
            "exportfilename" : f"{ori_db['knps']}_{ori_db['class_num']}_{ori_db['start_year']}_{ori_db['end_year']}",
            "bgColor": "#262A33",
            "bgAlpha": "100",
            "showBorder": "0",
            "showvalues": "0",
            "numvisibleplot": "12",
            "caption": f"EVI of {ori_db['knps']}",
            "subcaption": f"class_num : {ori_db['class_num']}",
            "yaxisname": "EVI",
            "theme": "candy",
            "drawAnchors": "0",
            "plottooltext": "<b>$dataValue</b> EVI of $label",

        },
        "categories": [{  # X축
            "category": [{"label": str(i)} for i in range(1, 365, 8)]
        }],
        "dataset": []  # Y축
    }

    # 데이터셋에 데이터 넣기
    for now in range(int(ori_db['start_year']), int(ori_db['end_year']) + 1):  # start_year에서 end_year까지
        db["dataset"].append({
            "seriesname": str(now),  # 레이블 이름
            # 해당 연도에 시작 (1월 1일)부터 (12월 31)일까지의 EVI 값을 넣기
            "data": [{"value": i} for i in
                     df[df['date'].str[:4] == str(now)].avg]
        })

    # 그래프 그리기
    chartObj = FusionCharts('scrollline2d', 'ex1', 960, 400, 'chart-1', 'json', json.dumps(db))

    return chartObj.render()  # 그래프 정보 넘기기


def export_doy(ori_db):
    df = pd.read_csv(root + f"data{middle}knps_final.csv")
    df_sos = pd.read_csv(root + f"data{middle}knps_sos.csv")
    df_sos = df_sos[['year',ori_db['knps']+'_'+ori_db['class_num']]]
    df_sos.columns = ['year', 'sos']

    phenophase_date = ''
    phenophase_betw = ''

    sos = []
    doy = []
    betwn = []
        
    # sos 기준으로 개엽일 추출
    for year in range(int(ori_db['start_year']), int(ori_db['end_year']) + 1):
        phenophase_doy = df_sos[df_sos['year'] == year]['sos'].to_list()[0]  # sos 스칼라 값
        phenophase_date = (f'{year}년 : {phenophase_doy}일')
        sos.append(phenophase_date)

        data = df[df['date'].str[:4] == str(year)]
        thresh = np.min(data['avg']) + ((np.max(data['avg']) - np.min(data['avg'])) * (
            float(ori_db["threshold"])))  ##개엽일의 EVI 값

        ## 개엽일 사이값 찾기
        high = data[data['avg'] >= thresh]['date'].iloc[0]
        low = data.date[[data[data['avg'] >= thresh].index[0] - 1]].to_list()[0]
        high_value = data.avg[data['date'] == high].to_list()[0]  ## high avg 값만 추출
        low_value = data.avg[data['date'] == low].to_list()[0]  ## low avg 값만 추출
        div_add = (high_value - low_value) / 8

        for a in range(8):
            if low_value > thresh:
                break
            else:
                low_value += div_add

        phenophase_doy = format(pd.to_datetime(low) + datetime.timedelta(days=a - 1), '%Y-%m-%d')
        phenophase_date= format(datetime.datetime.strptime(phenophase_doy, '%Y-%m-%d'), '%j')+'일,'+phenophase_doy
        phenophase_betw= (f'{low} ~ {high}')
        doy.append(phenophase_date)
        betwn.append(phenophase_betw)

    total_DataFrame = pd.DataFrame(columns=['SOS기준 개엽일', '임계치 개엽일', '임계치 오차범위'])

    for i in range(len(doy)):
        total_DataFrame.loc[i] = [sos[i], doy[i], betwn[i]]




    html_DataFrame = total_DataFrame.to_html(justify='center', index=False, table_id ='mytable')



    return html_DataFrame
