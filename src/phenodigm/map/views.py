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


# 전역변수
root = "/Users/ho/Desktop/dl/"

'''홈 페이지'''
@xframe_options_exempt # iframe 허용하기 위한 태그
def index(request):
    m = folium.Map(location=[36.684273, 128.068635], zoom_start=6.5, width="100%", height="100%")  # 기본이 되는 지도 정보 가져오기

    name = ["gaya", "gyeongju", "gyeryong", "naejang", "dadohae", "deogyu", "mudeung", "byeonsan", "bukhan", "seorak", # 산 이름
            "sobaek", "songni", "odae", "worak", "wolchul", "juwang", "jiri", "chiak", "taebaek", "taean", "halla", "hallyeo"]
    position = [(35.779385, 128.122559), (35.867430, 129.222565), (36.356057, 127.212067), (35.483333, 126.883333), # 국립공원명과 대치되는 (위도, 경도)
                (34.708203, 125.901489), (35.824494, 127.787476), (35.134127, 126.988756), (35.680893, 126.531392), (37.70338, 127.032166),
                (38.1573652, 128.4355274), (36.909725, 128.459374), (36.533333, 127.9), (37.724030, 128.599777), (36.852005, 128.197261),
                (34.757310, 126.680823), (36.402218, 129.187889), (35.333333, 127.716667), (37.37169, 128.050509), (37.0545, 128.916666),
                (36.78712, 126.143475), (33.366667, 126.533333), (34.75882, 127.97596)]

    # 국립공원 수만큼 반복
    for i in range(len(name)):
        # 국립공원 대표 이미지 가져오기
        pic = base64.b64encode(open(f'../resource/{name[i]}.png', 'rb').read()).decode()
        # 국립공원 대표 이미지 클릭시 해당 분석 페이지로 이동하는 html 태그 설정
        image_tag = f'<div style="text-align:center; "><a href="http://127.0.0.1:8000/analysis/?knps={name[i]}&start_year=2003&end_year=2003&class_num=0&curve_fit=1&shape=1" target="_top"><img src="data:image/png;base64,{pic}" width="200" height="150"></a></div>'
        # iframe 생성
        iframe = folium.IFrame(image_tag, width=220, height=170)
        # html 띄울 popup 객체 생성
        popup = folium.Popup(iframe)

        # 지도에 마커 찍기
        folium.Marker(position[i],
                      popup=popup,
                      icon=folium.Icon(color='green', icon='fa-tree', prefix='fa')).add_to(m)

    plugins.LocateControl().add_to(m) # 위치 컨트롤러 추가

    maps = m._repr_html_()  # 지도를 템플릿에 삽입하기위해 iframe이 있는 문자열로 반환 (folium)

    return render(request, 'map/index.html', {'map': maps}) # index.html에 map 변수에 maps 값 전달하기


'''분석 페이지'''
def analysis(request):
    property_list = ["knps", "curve_fit", "start_year", "end_year", "class_num", "threshold", "shape"] # GET 메소드로 주고 받을 변수 이름들
    db = {} # 데이터를 저장하고 페이지에 넘겨 줄 딕셔너리

    if request.method == 'GET': # GET 메소드로 값이 넘어 왔다면,
        for key in property_list:
            # 값이 넘어 오지 않았다면 "", 값이 넘어 왔다면 해당하는 값을 db에 넣어줌
            db[f"{key}"] = request.GET[f"{key}"] if request.GET.get(f"{key}") else "" # 삼항 연산자

    db['graph'] = get_chart(db) if db['shape'] == "1" else get_multi_plot(db) # shape 값(연속, 연도)에 따라 그래프를 그려줌
    # *shape는 default: 1임

    return render(request, 'map/analysis.html', db) # 웹 페이지에 값들 뿌려주기


'''예측 페이지'''
def predict(request):
    property_list = ["mountain", "curve_fit", "start_year", "end_year", "class_num", "threshold"]

    db = {}

    if request.method == 'GET':
        for key in property_list:
            db[f"{key}"] = request.GET[f"{key}"] if request.GET.get(f"{key}") else ""
    print(request.GET)
    return  render(request, 'map/predict.html', db)


'''윤년 구하는 메소드'''
def get_Feb_day(year):
    # 4, 100, 400으로 나누어 떨어진다면 윤년
    if year % 4 == 0 or year % 100 == 0 or year % 400 == 0:
        day = 29
    else:
        day = 28

    return day


'''연속된 그래프를 그려주는 메소드'''
def get_chart(ori_db):
    df = pd.read_csv(root + "bukhan/bukhan_2.csv") # curve fitting된 데이터 가져오기
    data = [] # 그래프를 그리기 위한 데이터
    schema = [{"name": "Time", "type": "date", "format": "%Y-%m-%d"}, {"name": "EVI", "type": "number"}] # 하나의 data 구조
    info_day = [None, 31, None, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # 월별 일 수 정보
    year, month, day = 2003, 1, 1 # 년월일을 알기 위한 변수

    for _ in range(len(df)): # data에 값 채우기
        data.append([f"{year}-{month}-{day}", df.iloc[len(data), 1]]) # schema 형태에 맞게 데이터 추가

        day += 8 # 8일 간격씩 데이터 추가
        if month == 2: # 2월은 윤년 여부 판단하기
            day_limit = get_Feb_day(year)
        else: # 2월이 아니면 해당 월의 일 수 가져오기
            day_limit = info_day[month]

        # 다음 월로 넘어가야 한다면,
        if day > day_limit:
            day -= day_limit # 새로운 일
            month += 1 # 다음 월로 가기

            if month > 12: # 다음 연도로 넘어가야 한다면,
                year += 1
                # 무조건 1월 1일부터 시작하기 때문에 month와 day를 1로 초기화
                month = 1
                day = 1

    fusionTable = FusionTable(json.dumps(schema), json.dumps(data)) # 데이터 테이블 만들기
    timeSeries = TimeSeries(fusionTable) # 타임시리즈 만들기

    # 그래프 속성 설정하기
    timeSeries.AddAttribute('caption', '{"text":"식생지수 분석"}')
    timeSeries.AddAttribute('chart', '{"theme":"candy", "exportEnabled": "1"}')
    timeSeries.AddAttribute('subcaption', '{"text":"국립공원공단 레인저스"}')
    timeSeries.AddAttribute('yaxis', '[{"plot":{"value":"EVI"},"format":{"prefix":""},"title":"EVI"}]')


    # 그래프 그리기
    fcChart = FusionCharts("timeseries", "ex1", 1180, 450, "chart-1", "json", timeSeries)

    # 그래프 정보 넘기기
    return  fcChart.render()


'''연도별 그래프를 그려주는 메소드'''
def get_multi_plot(ori_db):
    df = pd.read_csv(root + "bukhan/bukhan_2.csv")  # curve fitting된 데이터 가져오기

    # 그래프 속성 및 데이터를 저장하는 변수
    db = {
        "chart": { # 그래프 속성
            "exportEnabled": "1",
            "bgColor": "#262A33",
            "bgAlpha": "100",
            "showBorder": "0",
            "showvalues": "0",
            "numvisibleplot": "12",
            "caption": "식생지수 분석",
            "subcaption": "국립공원공단 레인저스",
            "yaxisname": "EVI",
            "theme": "candy",
            "drawAnchors": "0",
            "plottooltext": "<b>$dataValue</b> EVI of $label",

        },
        "categories": [{ # X축
            "category": [{"label": str(i)} for i in range(1, 365, 8)]
        }],
        "dataset": [] # Y축
    }

    # 데이터셋에 데이터 넣기
    for now in range(int(ori_db['start_year']), int(ori_db['end_year']) + 1): # start_year에서 end_year까지
        db["dataset"].append({
            "seriesname": str(now), # 레이블 이름
            # 해당 연도에 시작 (1월 1일)부터 (12월 31)일까지의 EVI 값을 넣기
            "data": [{ "value": i } for i in df[(df[df.columns[0]] >= 365 * (now - 2003)) & (df[df.columns[0]] <= 365 * (now - 2002))][df.columns[1]]]
        })

    # 그래프 그리기
    chartObj = FusionCharts('scrollline2d', 'ex1', 1180, 450, 'chart-1', 'json', json.dumps(db))

    return chartObj.render() # 그래프 정보 넘기기

