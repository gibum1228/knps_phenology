import os

import pandas as pd

import phenoKOR as pk
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import model

# plot 한글 깨짐 현상 방지
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    root = "/Users/beom/Desktop/git/data/knps/"
    order = 4

    if order == 1:
        print("load mask()")
        mask = pk.load_mask(root, "roi.mat")
        print("load image()")
        df, imgs = pk.load_image(root + "sungsamjae/2020/sample")
        print("load_roi()")
        rois = pk.load_roi(imgs, mask)

        print("get cc()")
        r_cc, g_cc, b_cc = pk.get_cc(rois)

        print(g_cc)

        plt.plot(g_cc)
        plt.show()
    elif order == 2:
        x, *y = pk.load_csv(root + "day_8_data/2021_halla_final.csv") # 그래프 그릴 파일 가져오기
        title = ["비산림", "침엽수림", "활엽수림", "혼효림"] # 클래스 정보
        color = ["red", "blue", "green", "violet"] # 컬러 정보

        plt.figure(figsize=(15, 15)) # 창 크기 설정
        for i in range(4): # 클래스별 반복
            fp = np.polyfit(x, y[i], 15) # 커브 피팅

            plt.subplot(220 + (i+1))
            plt.title("2021년 한라산의 " + title[i])
            plt.ylim(0.1, 1.0) # y축 범위
            plt.plot(x, y[i], c=color[i])
            # plt.plot(x, np.poly1d(fp)(x), c=color[i])
            plt.xlabel("DOY")
            plt.ylabel("EVI")

        plt.show()
    elif order == 3:
        df = model.load_data()

        model.fit_prophet()
    elif order == 4:
        final_df = pd.DataFrame(columns=["date", "code", "class", "avg"])

        '''윤년 구하는 메소드'''
        def get_Feb_day(year):
            # 4, 100, 400으로 나누어 떨어진다면 윤년
            if year % 4 == 0 or year % 100 == 0 or year % 400 == 0:
                day = 29
            else:
                day = 28

            return day

        for year in range(2003, 2022):
            info_day = [None, 31, get_Feb_day(year), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 월별 일 수 정보

            for knps in pk.get_knps_name():
                df = pd.read_csv(f"/Users/beom/Desktop/git/knps_phenology/data/{year}_{knps}_final.csv")
                month, day = 1, 1
                x = []

                for i in range(46):  # data에 값 채우기
                    # 데이터 추가
                    for _ in range(4):
                        x.append(f"{year}-{month}-{day}")

                    day += 8  # 8일 간격씩 데이터 추가
                    # 다음 월로 넘어가야 한다면,

                    if day > info_day[month]:
                        day -= info_day[month]  # 새로운 일
                        month += 1  # 다음 월로 가기

                df["date"] = pd.to_datetime(x, format="%Y-%m-%d")

                final_df = pd.concat([final_df, df])

        final_df.sort_values("code")
        final_df.reset_index(inplace=True, drop=True)
        del final_df['Unnamed: 0']
        print(final_df)
        final_df.to_csv(f"/Users/beom/Desktop/git/knps_phenology/data/knps_final.csv")