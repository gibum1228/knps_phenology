import os
import platform
import pandas as pd


'''DataFrame에 저장할 정보를 담은 데이터베이스 초기화 메소드'''
def init_db(class_num):
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

    # 파일 저장
    df.to_csv(root + "ori_data" + middle + f"{'_'.join([year, code, modis[:3]])}.csv")
    # 일별 데이터 초기화
    db = init_db(4, column)

    return db


'''
QGIS에서 뽑아낸 csv 파일에 있는 클래스별 일일 데이터에서 EVI 평균값으로 일일 대표 데이터를 만들어 연도별로 정리한 연도_코드_위성.csv 파일을 생성 
'''
def data_preprocessing(dir_name):
    # 변수 초기화
    last_info = None  # 계산중인 '연도_코드_위성'
    db = init_db(4, column)  # 일별 데이터
    dir_list = sorted(os.listdir(root + dir_name))  # 파일 목록
    if middle == "/": dir_list.pop(0) # 맥북이라면 .DS_Store를 삭제

    # 파일 개수만큼 반복
    for n in range(len(dir_list)):
        if n // 100 % 0: print(f"{len(dir_list)}개 중 {n}개 작업중.......")
        filename = dir_list[n]  # 파일 이름 가져오기
        after_filename = filename.replace("_", ".")  # 파일명 전처리 후

        '''현재 파일에 대한 정보 가져오기'''
        year, code, modis, day, *_ = after_filename.split(".")
        info = "_".join([year, code, modis])

        if last_info is None: last_info = info  # 처음 실행할 경우

        '''(연도 혹은 코드 혹은 위성)이 바뀌었을 경우, 파일 저장해야 함'''
        if last_info != info:
            db = save_sequence_csv(db, last_info)
            last_info = info

        '''연도, 코드, 위성이 같은 파일이라면, 일별 데이터 추가해야 함'''
        df = pd.read_csv(root + dir_name + middle + filename)  # 전처리가 되지 않은 일일 데이터 가져오기
        # 결측치 제거
        df.drop(index=df[df["FRTP_CD"] == 4].index, inplace=True)  # 죽림(4)는 삭제
        df.dropna(subset=['_mean'], inplace=True)  # 평균값이 없는(_count, _sum이 0) 행을 삭제 -> EVI 값 없음

        info_class = df["FRTP_CD"].to_list()  # 클래스 코드
        info_count = df["_count"].to_list()  # 면적의 픽셀 갯수
        info_sum = df["_sum"].to_list()  # 면적의 픽셀(EVI) 총합

        # 클래스별 count, sum 구하기
        for i in range(len(df)):
            db[f"{info_class[i]}"][0] += info_count[i]
            db[f"{info_class[i]}"][1] += info_sum[i]

        # 일일 클래스별 평균값 구하기
        for i in range(4):
            if db[f"{i}"][0] == 0:  # count가 0이면 클래스가 없음
                continue

            db["date"].append(year + "-" + day[-3:])
            db["code"].append(code)
            db["class"].append(i)
            db["avg"].append(db[f"{i}"][1] / db[f"{i}"][0])
            # 초기화
            db[f"{i}"] = [0, 0]

        # 마지막 파일이라면 그대로 저장
        if n == len(dir_list) - 1:
            save_sequence_csv(db, info)


'''MODIS로 촬영한 16일 간격을 가지는 두 개(Terra, Aqua)의 데이터를 8일 간격으로 병합해주는 메소드'''
def merge_8day(dir_name):
    path = root + dir_name + middle
    dir_list = sorted(os.listdir(root + dir_name)) # 파일 목록 가져오기

    for n in range(0, len(dir_list), 2): # 2개가 한 세트이기 떄문에 반복문이 2씩 증가
        # 테라, 아쿠아 파일 이름 가져오기
        terra_filename = dir_list[n]
        aqua_filename = dir_list[n+1]

        # 파일 정보 가져오기
        year, code, *_ = terra_filename.split("_")

        # 연도 정보가 담긴 csv 파일 가져오기
        terra_df = pd.read_csv(path + terra_filename)
        aqua_df = pd.read_csv(path + aqua_filename)

        df = pd.concat([terra_df, aqua_df], ignore_index=True) # 두 개의 csv 파일을 합치기
        df.drop(df.columns[0], axis=1, inplace=True) # 불필요한 열 삭제
        df.sort_values(['date', 'class'], inplace=True) # date, class 순으로 정렬
        df.reset_index(drop=True, inplace=True) # 인덱스 초기화

        # 알맞은 파일명으로 저장
        df.to_csv(f"{root}8_day_data{middle}{year}_{code}_final.csv")


'''전역 변수'''
root = "/Users/beom/Desktop/git/data/knps/" # csv 파일 위치
middle = "/" if platform.system() == "Darwin" else "\\" # 운영체제에 따라 슬래쉬 설정
column = ["date", "code", "class", "avg"] # 최종 파일 컬럼

'''
일별 데이터 == db
db.shape == 일(23) * 클래스 수(4) by 4
'''
if __name__ == "__main__":
    # 일일 데이터 연도별 데이터로 만들기
    data_preprocessing("csv")
    # 16일 간격의 두 개의 데이터를 병합해 8일 간격으로 만들기
    merge_8day("ori_data")

