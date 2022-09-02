# analysis.py

## 목차

1. [show_graph()](#show_graph)
2. [show_graphs()](#show_graphs)
3. [replace_blank()](#replace_blank)
4. [get_Feb_day()](#get_Feb_day)
5. [adf_test()](#adf_test)
6. [kpss_test()](#kpss_test)
7. [show_data_distribution()](#show_data_distribution)
8. [show_acf_pacf_plot()](#show_acf_pacf_plot)

## show_graph

`매개변수`: ori_db:dict, option:int, df:pandas.DataFrame   
`기능`: fusionCharts를 사용해 option(분석, 예측, 고정형 카메라)에 따라 한 개의 시계열 데이터 그래프를 그리는 메소드   
`리턴값`: fcChart.render():fusionCharts html   

## show_graphs

`매개변수`: ori_db:dict, option:int, df:pandas,DataFrame    
`기능`: fusionCharts를 사용해 option(분석, 예측, 고정형 카메라)에 따라 여러 개의 연도별 시계열 데이터 그래프를 그리는 메소드    
`리턴값`: fcChart.render():fusionCharts html   

## replace_blank

`매개변수`: df: pandas.DataFrame, key:str   
`기능`: 고정형 카메라 정보에서 그래프를 그리기 위해 결측치를 "None"으로 채우기 위한 메소드   
`리턴값`: replace_value_list:list   

## get_Feb_day

`매개변수`: year:int   
`기능`: 윤년일 때 2월달 일 수를 구하는 메소드   
`리턴값`: day:int   

## adf_test

`매개변수`: timeseries:pandas.DataFrame    
`기능`: 시계열에 단위근이 존재하는지의 여부 검정함으로써 정상 시계열인지 여부 판단 메소드   
`리턴값`: int, int, int, int   

## kpss_test

`매개변수`: timeseries:pandas.DataFrame   
`기능`:  시계열이 정상성인지 판정하는 메소드   
`리턴값`: int, int, int, int   

## show_data_distribution

`매개변수`: 없음     
`기능`: 전체 및 산림별 데이터에 대한 기술통계량 확인 및 Boxplot을 통한 연도별 EVI 분포 확인하는 메소드  
`리턴값`: 없음  

## show_acf_pacf_plot

`매개변수`: 없음   
`기능`: 특정 국립공원의 활엽수림 시계열 데이터 대한 ACF과 PACF 확인하는 메소드. 이를 통해 정상성 여부 및 ARIMA 파라미터인 p,d,q 값 추론   
`리턴값`: 없음  
