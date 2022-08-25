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





### ROI 영역 지정
pts = []  # 마우스로 클릭한 포인트를 저장
resultforJSON = []  # pts에 저장된 포인트를 json형태로 저장
file_path = './ROI.json'  # json으로 저장하기 위한 파일경로
mask_list = []


def draw_roi(event, x, y, flags, param):  # roi검출을 위한 함수 정의
    global pts
    img2 = img.copy()

    if event == cv.EVENT_LBUTTONDOWN:  # 마우스 왼쪽버튼을 클릭하면
        pts.append((x, y))  # pts에 (x,y)좌표를 추가한다
        print('포인트 #%d 좌표값(%d,%d)' % (len(pts), x, y))  # 정상적으로 추가되는지 출력으로 확인

    #         resultforJSON.append({'point':[len(pts)],
    #                               'coordinate':[[int(x),int(y)]]})
    #                               # 포인트 순서와 좌표값을 딕셔너리 형태로 추가해준다

    if event == cv.EVENT_RBUTTONDOWN:  # 마우스 오른쪽버튼을 클릭하면
        pts.pop()  # 클릭했던 포인트를 삭제한다

    if event == cv.EVENT_LBUTTONDBLCLK:
        print("in db")
        pts.pop()
        mask_list.append(pts)
        pts = []

    if event == cv.EVENT_MBUTTONDOWN:  # 마우스 중앙(휠)버튼을 클릭하면
        #         print('총 %d개의 포인트 설정' % len(pts))
        #         mask = np.zeros(img.shape, np.uint8) #컬러를 다루기 때문에 np로 형변환
        #         points = np.array(pts, np.int32)
        #         points = points.reshape((-1,1,2)) #pts 2차원을 이미지와 동일하게 3차원으로 재배열
        #         mask = cv.polylines(mask, [points], True, (255,255,255), 2) #포인트와 포인트를 연결하는 라인을 설정
        #         mask2 = cv.fillPoly(mask.copy(), [points], (255,255,255)) #폴리곤 내부 색상 설정

        #         ROI = cv.bitwise_and(mask2,img) # mask와 mask2에 중첩된 부분을 추출
        print("in middle")
        result_roi = np.zeros(img.shape, np.uint8)

        for point in mask_list:
            print("for.....")
            mask = np.zeros(img.shape, np.uint8)  # 컬러를 다루기 때문에 np로 형변환
            points = np.array(point, np.int32)
            points = points.reshape((-1, 1, 2))  # pts 2차원을 이미지와 동일하게 3차원으로 재배열
            mask = cv.polylines(mask, [points], True, (255, 255, 255), 2)  # 포인트와 포인트를 연결하는 라인을 설정
            mask2 = cv.fillPoly(mask.copy(), [points], (255, 255, 255))  # 폴리곤 내부 색상 설정

            ROI = cv.bitwise_and(mask2, img)  # mask와 mask2에 중첩된 부분을 추출

            result_roi = cv.add(result_roi, ROI)

        #         with open(file_path,'w') as outfile: #resultforJSON에 저장된 내용을 json파일로 추출
        #             json.dump(resultforJSON,outfile,indent=4)

        result_roi = np.where(result_roi == 0, result_roi, 255)
        print("save")
        cv.imwrite('result_roi.png', result_roi)
        cv.imshow('ROI', result_roi)
        #         cv.waitKey(0)
        exit()

    if len(pts) > 0:  # 포인트를 '원'으로 표시
        cv.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv.circle(img2, pts[i], 5, (0, 0, 255), -1)
            cv.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv.imshow('image', img2)


img = cv.imread("C:/Users/kub84/Desktop/jir031_2021_06_01_132807.JPG")
img = cv.resize(img, (600, 400))
cv.namedWindow('image')
cv.setMouseCallback('image', draw_roi)

while True:
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord('s'):
        saved_data = {'ROI': pts}
        break
cv.destroyAllWindows()