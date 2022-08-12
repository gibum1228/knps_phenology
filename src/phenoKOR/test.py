import phenoKOR as pk
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# plot 한글 깨짐 현상 방지
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    root = "/Users/beom/Desktop/git/data/knps/"
    order = 2

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
        x, *y = pk.load_csv(root + "ori_data/2021_jiri_MYD.csv") # 그래프 그릴 파일 가져오기
        title = ["비산림", "침엽수림", "활엽수림", "혼효림"] # 클래스 정보
        color = ["red", "blue", "green", "violet"] # 컬러 정보

        plt.figure(figsize=(15, 15)) # 창 크기 설정
        for i in range(4): # 클래스별 반복
            fp = np.polyfit(x, y[i], 10) # 커브 피팅

            plt.subplot(220 + (i+1))
            plt.title("2021년 지리산의 " + title[i])
            plt.ylim(0.1, 0.7) # y축 범위
            plt.scatter(x, y[i], c=color[i])
            plt.plot(x, np.poly1d(fp)(x), c=color[i])
            plt.xlabel("DOY")
            plt.ylabel("EVI")

        plt.show()