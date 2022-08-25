import matplotlib.pyplot as plt
import cv2 as cv
import imutils
import numpy as np
import json


# Chromatic Coordinate 값 연산
def get_cc(imgs):
    # b, g, r 순서로 채널별 값 구하기
    red_dn = imgs[:, :, 2]
    blue_dn = imgs[:, :, 0]
    green_dn = imgs[:, :, 1]

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


# ROI 마스크 PNG 파일 추출
def draw_mask():
    pts = []  # 마우스로 클릭한 포인트 저장
    mask_list = []  # 마스크 리스트 저장

    # 마스크 추출을 위한 마우스 클릭 이벤트 리스너
    def draw_mask_eventListener(event, x, y, flags, param):
        global pts
        img2 = img.copy()

        if event == cv.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭 시 pts에 (x,y)좌표를 추가
            pts.append((x, y))

        if event == cv.EVENT_RBUTTONDOWN:  # 마우스 오른쪽 버튼 클릭 시 클릭 했던 포인트를 삭제
            pts.pop()

        if event == cv.EVENT_LBUTTONDBLCLK:  # 마우스 왼쪽 버튼 더블 클릭 시 좌표들을 리스트에 추가
            # 초기화
            mask_list.append(pts)
            pts = []

        if event == cv.EVENT_MBUTTONDOWN:  # 마우스 중앙(휠)버튼 클릭 시 ROI 선택 종료
            result_roi = np.zeros(img.shape, np.uint8)  # 최종 마스크 이미지

            for point in mask_list:
                if not point: continue
                mask = np.zeros(img.shape, np.uint8)
                points = np.array(point, np.int32)
                points = points.reshape((-1, 1, 2))  # pts 2차원을 이미지와 동일하게 3차원으로 재배열
                mask = cv.polylines(mask, [points], True, (255, 255, 255), 2)  # 포인트를 연결하는 라인을 설정 후 마스크 생성
                mask2 = cv.fillPoly(mask.copy(), [points], (255, 255, 255))  # 채워진 다각형 마스크 생성

                ROI = cv.bitwise_and(mask2, img)  # img와 mask2에 중첩된 부분을 추출

                result_roi = cv.add(result_roi, ROI)  # 마스크 이미지끼리 더하기

            result_roi = np.where(result_roi == 0, result_roi, 255)  # 첫번째 매개변수 조건에 따라 참이면 유지, 거짓이면 255으로 변경
            cv.imwrite('result_roi.png', result_roi) # 저장
            cv.destroyAllWindows()  # 열린 창 닫기
            cv.waitKey(0)

        try:
            if len(pts) > 0:  # 마우스 포인트 원으로 지정
                cv.circle(img2, pts[-1], 3, (0, 0, 255), -1)
        except:
            pts = []

        if len(pts) > 1:  # 마우스 포인트 연결 라인 생성
            for i in range(len(pts) - 1):
                cv.circle(img2, pts[i], 5, (0, 0, 255), -1)
                cv.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

        if len(mask_list) > 0:  # 마스크 여러 개일때 포인트 연결 라인 생성
            for m in mask_list:
                for i in range(len(m) - 1):
                    cv.circle(img2, m[i], 5, (0, 0, 255), -1)
                    cv.line(img=img2, pt1=m[i], pt2=m[i + 1], color=(255, 0, 0), thickness=2)

        cv.imshow('image', img2)  # 이미지 화면 출력

    img = cv.imread("C:/Users/kub84/Desktop/jir031_2021_06_01_132807.JPG")  # 저장된 이미지 읽어 오기
    img = cv.resize(img, (600, 400))
    cv.namedWindow('image')  # 새로운 윈도우 창 이름 설정
    cv.setMouseCallback('image', draw_mask_eventListener)  # 마우스 이벤트가 발생했을 때 전달할 함수

    while True:
        key = cv.waitKey(1) & 0xFF  # SOH
        if key == 27:  # ESC
            break
    cv.destroyAllWindows()  # 열린 창 닫기
