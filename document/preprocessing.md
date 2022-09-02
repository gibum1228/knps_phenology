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

## get_info
매개변수 : 없음
기능 : 프로젝트 폴더 경로 탐색 메소드
리턴값 : (str, str)  

## get_final_data
매개변수 : (knps: str, class_num: str, all: bool) 
기능 : 분석, 예측에 필요한 데이터 프레임을 추출하여 리턴하는 메소드
리턴값 : pd.DataFrame

## get_mask_for_mat
매개변수 : (path: str, filename: str) 
기능 : 저장되어 있는 mat파일을 읽어오는 메소드
리턴값 : 없음

## mat2image
매개변수 : (path: str, filename: str) 
기능 : mat 파일에서 마스크 이미지를 생성하는 메소드
리턴값 : 없음


## get_image_for_local
매개변수 : (path: str)
기능 : 로컬 경로에 존재하는 이미지 데이터 가져오는 메소드
리턴값 : (pd.DataFrame, npt.NDArray)

## get_image_for_web
매개변수 : (folder: dict)
기능 : 웹상에 이미지 데이터 가져오는 메소드
리턴값 : (pd.DataFrame, npt.NDArray)

## byte2img
매개변수 : (byte: bytes)
기능 : 바이트파일 이미지파일로 변환 메소드
리턴값 : (img:이미지 파일)

## get_roi
매개변수 : (img: npt.NDArray, mask: npt.NDArray)
기능 : ROI(관심영역)정보 추출 메소드
리턴값 : (npt.NDArray)

## get_cc
매개변수 : (img: npt.NDArray)
기능 : Chromatic Coordinate  연산 메소드
리턴값 : (float, float)

## curve_fit
매개변수 : (y, ori_db: pd.DataFrame)
기능 : 다양한 커브피팅 선택 메소드
리턴값 : (pd.DataFrame, pd.DataFrame)

## get_knps_name_en
매개변수 : 없음
기능 : 국립공원 영어이름 불러오기
리턴값 :  (list)

## get_knps_name_kr
매개변수 : 없음
기능 : 국립공원 한글이름 불러오기
리턴값 : (list)

## get_knps_position
매개변수 : 없음
기능 : 국립공원 위도,경도 정보 불러오기
리턴값 : (list)
