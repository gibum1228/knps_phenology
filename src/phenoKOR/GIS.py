import os
import platform
import pandas as pd


'''DataFrame에 저장할 정보를 담은 데이터베이스 초기화 메소드'''
def init_db(class_num, columns):
    db = {}

    # 클래스(0-3) 별 갯수, 총합 저장
    for key in range(class_num):
        db[f'{key}'] = [0, 0] # [count, sum]

    # 행의 기본 정보를 저장
    for key in column:
        db[f'{key}'] = [] # columns

    return db


'''일별 데이터를 시계열로 만든 연도 데이터로 저장'''
def save_sequence_csv(db, info):
    df = pd.DataFrame(columns=column)

    year, code, modis = info.split("_")  # 저장할 파일 정보를 추출

    for key in column:  # 일별 데이터를 DataFrame으로 변환
        df[f"{key}"] = db[f"{key}"]
    print(df.head())
    # 파일 저장
    df.to_csv(root + "ori_data" + middle + f"{'_'.join([year, code, modis[:3]])}.csv")
    # 일별 데이터 초기화
    db = init_db(4, column)

    return db


'''전역 변수'''
root = "/Users/beom/Desktop/git/data/knps/" # csv 파일 위치
middle = "/" if platform.system() == "Darwin" else "\\" # 운영체제에 따라 슬래쉬 설정
column = ["date", "code", "class", "avg"] # 최종 파일 컬럼

'''
일별 데이터 == db
db.shape == 일(23) * 클래스 수(4) by 4
'''
if __name__ == "__main__":
    # 변수 초기화
    last_info= None # 계산중인 '연도_코드_위성'
    db = init_db(4, column) # 일별 데이터
    dir_list = sorted(os.listdir(root + "csv")) # 파일 목록

    for n in range(len(dir_list)): # 파일 개수만큼 반복
        filename = dir_list[n] # 파일 이름들 가져오기
        if filename == ".DS_Store": continue # 맥북 기본 설정 파일은 스킵

        '''현재 파일에 대한 정보 가져오기'''
        info, day, *_ = filename.split(".")
        year, code, modis = info.split("_")

        if last_info is None: last_info = info # 처음 실행할 경우

        '''연도 혹은 코드 혹은 위성이 바뀌었을 경우, 파일 저장해야 함'''
        if last_info != info:
            db = save_sequence_csv(db, last_info)

        '''연도, 코드, 위성이 같은 파일이라면, 일별 데이터 추가해야 함'''
        df = pd.read_csv(root + "csv" + middle + filename) # 전처리가 되지 않은 일일 데이터 가져오기

        info_class = df["FRTP_CD"].to_list() # 클래스 코드 [0-4], but [0-3]만 필요
        info_count = df["_count"].to_list() # 면적의 픽셀 갯수
        info_sum = df["_sum"].to_list() # 면적의 픽셀(EVI) 총합

        # 클래스별 count, sum 구하기
        for i in range(len(df)):
            if info_class[i] == 4: break # 죽림일 경우 반복문 종료

            db[f"{info_class[i]}"][0] += info_count[i]
            db[f"{info_class[i]}"][1] += info_sum[i]

        # 일일 클래스별 평균값 구하기
        for i in range(4):
            db["date"].append(year + "-" + day[-3:])
            db["code"].append(code)
            db["class"].append(i)
            db["avg"].append(db[f"{i}"][1] / db[f"{i}"][0])
            # 초기화
            db[f"{i}"] = [0, 0]

        # 마지막 파일이라면 그대로 저장
        if n == len(dir_list) - 1:
            save_sequence_csv(db, info)