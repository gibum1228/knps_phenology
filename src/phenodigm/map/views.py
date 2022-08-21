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
import requests
import pandas as pd


# 전역변수
root = "/Users/beom/Desktop/git/data/knps/dl_data/"


@xframe_options_exempt
def index(request):
    m = folium.Map(location=[36.684273, 128.068635], zoom_start=6.5, width="100%", height="100%")  # 기본이 되는 지도 정보 가져오기

    name = ["gaya", "gyeongju", "gyeryong", "naejang", "dadohae", "deogyu", "mudeung", "byeonsan", "bukhan", "seorak",
            # 산 이름
            "sobaek", "songni", "odae", "worak", "wolchul", "juwang", "jiri", "chiak", "taebaek", "taean", "halla",
            "hallyeo"]
    position = [(35.779385, 128.122559), (35.867430, 129.222565), (36.356057, 127.212067), (35.483333, 126.883333),
                # 국립공원명과 대치되는 (위도, 경도)
                (34.708203, 125.901489), (35.824494, 127.787476), (35.134127, 126.988756), (35.680893, 126.531392),
                (37.70338, 127.032166), (38.1573652, 128.4355274), (36.909725, 128.459374), (36.533333, 127.9),
                (37.724030, 128.599777),
                (36.852005, 128.197261), (34.757310, 126.680823), (36.402218, 129.187889), (35.333333, 127.716667),
                (37.37169, 128.050509),
                (37.0545, 128.916666), (36.78712, 126.143475), (33.366667, 126.533333), (34.75882, 127.97596)]

    # 국립공원 수만큼 반복
    for i in range(len(name)):
        # 이미지 가져오기
        pic = base64.b64encode(open(f'../resource/{name[i]}.png', 'rb').read()).decode()
        image_tag = f'<div style="text-align:center; "><a href="http://127.0.0.1:8000/analysis/?knps={name[i]}&start_year=2003&end_year=2003&class_num=0&curve_fit=1&shape=1" target="_top"><img src="data:image/png;base64,{pic}" width="200" height="150"></a></div>'
        # iframe 생성
        iframe = folium.IFrame(image_tag, width=220, height=170)
        # html 띄울 popup 객체 생성
        popup = folium.Popup(iframe)

        # 지도에 마커 찍기
        folium.Marker(position[i],
                      popup=popup,
                      icon=folium.Icon(color='green', icon='fa-tree', prefix='fa')).add_to(m)

    plugins.LocateControl().add_to(m)

    maps = m._repr_html_()  # 지도를 템플릿에 삽입하기위해 iframe이 있는 문자열로 반환 (folium)

    return render(request, 'map/index.html', {'map': maps})


def analysis(request):
    property_list = ["knps", "curve_fit", "start_year", "end_year", "class_num", "threshold", "shape"]

    db = {}

    if request.method == 'GET':
        for key in property_list:
            db[f"{key}"] = request.GET[f"{key}"] if request.GET.get(f"{key}") else ""

    db['graph'] = get_chart(db['knps'], db['class_num'])

    return render(request, 'map/analysis.html', db)


def predict(request):
    pass


def get_Feb_day(year):
    if year % 4 == 0 or year % 100 == 0 or year % 400 == 0:
        day = 29
    else:
        day = 28

    return day


def get_chart(knps, class_num):
    df = pd.read_csv(root + "bukhan/bukhan_2.csv") # curve fitting된 데이터 가져오기
    data = [] # 그래프를 그리기 위한 데이터
    schema = [{"name": "Time", "type": "date", "format": "%Y-%m-%d"}, {"name": "EVI", "type": "number"}]
    info_day = [None, 31, None, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # 월별 일 수 정보
    year, month, day = 2003, 1, 1 # 년월일을 알기 위한 변수

    for doy in df.iloc[:, 0].to_list(): # 첫번째 열 정보 가져오기(day 정보)
        data.append([f"{year}-{month}-{day}", df.iloc[len(data), 1]])

        day += 8
        if month == 2:
            day_limit = get_Feb_day(year)
        else:
            day_limit = info_day[month]

        if day > day_limit:
            day -= day_limit
            month += 1

            if month > 12:
                year += 1
                month = 1
                day = 1

    fusionTable = FusionTable(json.dumps(schema), json.dumps(data))
    timeSeries = TimeSeries(fusionTable)

    timeSeries.AddAttribute('chart', '{}')
    timeSeries.AddAttribute('caption', '{"text":"식생지수 분석"}')
    timeSeries.AddAttribute('subcaption', '{"text":"국립공원공단 레인저스"}')
    timeSeries.AddAttribute('yaxis',
                            '[{"plot":{"value":"EVI"},"format":{"prefix":""},"title":"EVI"}]')

    # Create an object for the chart using the FusionCharts class constructor
    fcChart = FusionCharts("timeseries", "ex1", 700, 450, "chart-1", "json", timeSeries)
    print(len(df), len(data))
    # returning complete JavaScript and HTML code, which is used to generate chart in the browsers.
    return  fcChart.render()