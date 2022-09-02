<img src="./rangers_logo.png" width="200" height="200" />

# 레인저스(Rangers)

안녕하세요. 2022년 `데이터청년캠퍼스`에 참여한 경남대학교 빅리더 AI 아카데미의 `레인저스`입니다.   
> 저희 팀은 국립공원공단에서 기후변화를 확인하기 위해 위성 데이터와 고정 카메라에서 찍힌 고해상도 이미지 데이터로 개엽, 낙엽, 착엽 기간을 쉽게 `분석` 할 수 있도록 도와주고 `예측`까지 해주는 웹 서비스를 개발하였습니다.
   
사용한 데이터로는 [Terra](https://search.earthdata.nasa.gov/search?q=C1621383370-LPDAAC_ECS&sp[0]=127%2C38&qt=2003-01-01T00%3A00%3A00.000Z%2C2021-12-31T23%3A59%3A59.999Z&lat=37.99951171875&long=127.001953125&zoom=7), [Aqua](https://search.earthdata.nasa.gov/search?q=MYD13Q1%20V061&sp[0]=127%2C38&qt=2003-01-01T00%3A00%3A00.000Z%2C2021-12-31T23%3A59%3A59.999Z&lat=37.99951171875&long=127.001953125&zoom=7), [임상도](http://data.nsdi.go.kr/dataset/20190716ds00001), [국립공원 보호지역](http://www.kdpa.kr/), 고정 카메라 이미지가 있습니다.
<br><br>
## 목차
1. [설치](#1-설치)
1. [시작](#2-시작)
1. [데이터 미리보기](#3-데이터-미리보기)
1. [문서](#4-문서)
<br><br>
## 1. 설치
1-1. 깃허브 레파지토리 복사 후 폴더 이동
```
git clone https://github.com/gibum1228/knps_phenology.git
cd knps_phenology
```
1-2. 패키지 다운로드
```
pip install -r requirements.txt
```
설치 조건: 
`python >= 3.6.0`
<br><br>
## 2. 시작
2-1. 서버 시작(**레파지토리 폴더 내 root 기준**)
```
cd src/phenodigm
python manage.py runserver
```
2-2. [여기](http://127.0.0.1:8000)를 눌러 로컬 서버에 접속 
<br><br>
## 3. 데이터 미리보기
- 데이터 구조
```bash
├── 데이터청년캠퍼스 본선 제출
│   ├── 최종 데이터
│   │   └── 5_8일_간격_데이터
│   │
│   ├── 전처리중인 데이터
│   │   ├── 1_국립공원_보호지역
│   │   ├── 1_전국_임상도_(1:5000)
│   │   ├── 2_국립공원_보호지역의_임상도
│   │   ├── 2_인공위성_원본_데이터
│   │   ├── 3_인공위성과_임상도
│   │   ├── 4_16일_간격_데이터
│   │   └── 고정형_카메라_원본_데이터
│   │
│   └── 분석을 위한 데이터
│       ├── 모델 비교
│       └── 기온, 강수량 데이터
``` 
[여기](https://drive.google.com/drive/folders/18wJPKxjIfvqn2lseMWll9AnG_dIO0vnh?usp=sharing)를 눌러 전체 데이터가 있는 구글 드라이브에 바로가기
<br><br>
## 4. 문서

- [preprocessing.py](https://github.com/gibum1228/knps_phenology/blob/main/document/preprocessing.md)
- [analysis.py](https://github.com/gibum1228/knps_phenology/blob/main/document/analysis.md)
- [model.py](https://github.com/gibum1228/knps_phenology/blob/main/document/model.md)
