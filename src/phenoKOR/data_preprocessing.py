import os
import platform
import pandas as pd


'''전역변수'''
middle = "/" if platform.system() == "Darwin" else "\\" # 운영체제에 따라 슬래쉬 설정
root = os.getcwd() + middle


'''최종 통합 데이터 파일에서 원하는 국립공원 산림 데이터 로드하기'''
def load_final_data(knps, class_num):
    df = pd.read_csv(f"{root}data{middle}knps_final.csv")  # 데이터 가져오기

    # 조건에 맞는 데이터 추출
    return df[(df["code"] == knps) & (df["class"] == class_num)].sort_values('date')


'''윤년 구하는 메소드'''
def get_Feb_day(year):
    # 4, 100, 400으로 나누어 떨어진다면 윤년
    if year % 4 == 0 or year % 100 == 0 or year % 400 == 0:
        day = 29
    else:
        day = 28

    return day


'''국립공원 영어 이름을 로드하기'''
def get_knps_name_EN():
    name = ["gaya", "gyeongju", "gyeryong", "naejang", "dadohae", "deogyu", "mudeung", "byeonsan", "bukhan", "seorak",
            "sobaek", "songni", "odae", "worak", "wolchul", "juwang", "jiri", "chiak", "taebaek", "taean", "halla",
            "hallyeo"]

    return name


'''국립공원 한글 이름을 로드하기'''
def get_knps_name_KR():
    name = ["가야산", "경주", "계룡산", "내장산", "다도해상", "덕유산", "무등산", "변신반도", "북한산", "설악산", "소백산", "속리산", "오대산",
            "월악산", "월출산", "주왕산", "지리산", "치악산", "태백산", "태안해상", "한라산", "한려해상"]

    return name