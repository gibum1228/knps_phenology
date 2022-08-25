import matplotlib.pyplot as plt


# Chromatic Coordinate 값 연산
def get_cc(img):
    # b, g, r 순서로 채널별 값 구하기
    red_dn = img[:, :, 2]
    blue_dn = img[:, :, 0]
    green_dn = img[:, :, 1]

    # 분모에 해당하는 레드 + 블루 + 그린 색상값 더하기
    bunmo = red_dn + blue_dn + green_dn

    # 각각의 Chromatic Coordinate 값 구하기
    red_cc = red_dn / bunmo
    green_cc = green_dn / bunmo

    return red_cc, green_cc


# 다양한 curve fitting 알고리즘을 key으로 선택 가능
def curve_fit(y, key):
    pass


# plot 그리기
def show_plot(y, x = []):
    plt.figure(figsize=(10, 10))
    plt.plot(x, y) if x else plt.plot(y)
    plt.show()