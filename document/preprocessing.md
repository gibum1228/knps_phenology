# preprocessing.py

## 목차
1. [get_info()](#get_info)
2. [get_final_data()](#get_final_data)
3. [get_mask_for_mat](#get_mask_for_mat)
4. [mat2image()](#mat2image)
5. [get_image_for_local](#get_image_for_local)
6. [get_image_for_web](#get_image_for_web)
7. [byte2img()](#byte2img)
8. [get_roi()](#get_roi)
9. [get_cc()](#get_cc)
10. [curve_fit()](#curve_fit)
11. [get_knps_name_en()](#get_knps_name_en)
12. [get_knps_name_kr()](#get_knps_name_kr)
13. [get_knps_position()](#get_knps_position)
14. [curve_fit()](#curve_fit)
15. [double_logistic_func()](#double_logistic_func)
16. [savitzky_golay_func()](#savitzky_golay_func)
17. [gaussian_func()](#gaussian_func)

## get_info
`매개변수` : 없음   
`기능` : 프로젝트 폴더 경로 탐색 메소드   
`리턴값` : str, str  

## get_final_data
`매개변수` : knps:str, class_num:str, all:bool    
`기능` : 분석, 예측에 필요한 데이터 프레임을 추출하여 리턴하는 메소드   
`리턴값` : pandas.DataFrame   

## get_mask_for_mat
`매개변수` : path:str, filename:str   
`기능` : 저장되어 있는 mat파일을 읽어오는 메소드   
`리턴값` : 없음   

## mat2image
`매개변수` : path:str, filename:str   
`기능` : mat 파일에서 마스크 이미지를 생성하는 메소드   
`리턴값` : 없음   


## get_image_for_local
`매개변수` : path:str   
`기능` : 로컬 경로에 존재하는 이미지 데이터 가져오는 메소드   
`리턴값` : pandas.DataFrame, npt.NDArray   

## get_image_for_web
`매개변수` : folder:dict   
`기능` : 웹상에 이미지 데이터 가져오는 메소드   
`리턴값` : pandas.DataFrame, npt.NDArray   

## byte2img
`매개변수` : byte:bytes   
`기능` : 바이트파일 이미지파일로 변환 메소드   
`리턴값` : npt.NDArray   

## get_roi
`매개변수` : img:npt.NDArray, mask:npt.NDArray   
`기능` : ROI(관심영역)정보 추출 메소드   
`리턴값` : npt.NDArray   

## get_cc
`매개변수` : img:npt.NDArray   
`기능` : Chromatic Coordinate  연산 메소드   
`리턴값` : float, float      

## get_knps_name_en
`매개변수` : 없음   
`기능` : 국립공원 영어이름 불러오기   
`리턴값` :  list   

## get_knps_name_kr
`매개변수` : 없음   
`기능` : 국립공원 한글이름 불러오기   
`리턴값` : list   

## get_knps_position
`매개변수` : 없음   
`기능` : 국립공원 위도,경도 정보 불러오기   
`리턴값` : list   

## curve_fit
`매개변수` : df:pandas.DataFrame, ori_db:dict    
`기능` : 적용할 Curve Fitting 방법으로 연산 후 결과 리턴하는 메소드   
`리턴값` : pandas.DataFrame, pandas.DataFrame   

## double_logistic_func
`매개변수` : input_data:pandas.DataFrame, start_year:int, end_year:int, ori_db:dict)    
`기능` : Double Logistic 연산 메소드 (Double Logistic 수식 : mn + (mx - mn) * (1 / (1 + np.exp(-rsp * (t - sos))) + 1 / (1 + np.exp(rau * (t - eos))) - 1) ,
mn : min EVI, mx : max EVI, sos(start of season) : 증가 변곡점, rsp : sos에서 변화율, eos(end of season) : 감소 변곡점, rau : eos에서 변화율 )
`리턴값` : pandas.DataFrame(Double Logistic 적용된 EVI 값), pandas.DataFrame(연도별 sos)

## savitzky_golay_func
`매개변수` : data_input:pandas.DataFrame, start_year:int, end_year:int, ori_db:dict   
`기능` : Savitzky Golay 연산 메소드   
`리턴값` : pandas.DataFrame, list   

## gaussian_func
`매개변수` : data_input:pandas.DataFrame, start_year:int, end_year:int, ori_db:dict   
`기능` : Gaussian Filtering 연산 메소드    
`리턴값` : pandas.DataFrame, list   
