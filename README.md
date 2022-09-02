<img src="./rangers_logo.png" width="200" height="200" />

# 레인저스(Rangers)

안녕하세요. 2022년 `데이터청년캠퍼스`에 참여한 경남대학교 빅리더 AI 아카데미의 `레인저스`입니다.   
저희 팀은 국립공원공단에서 기후변화를 확인하기 위해 위성 데이터와 고정 카메라에서 찍힌 고해상도 이미지 데이터로 개엽, 낙엽, 착엽 기간을 쉽게 `분석` 할 수 있도록 도와주고 `예측`까지 해주는 웹 서비스를 개발하였습니다. 팀명이 `레인저스`인 이유는 국립공원을 지키는 사람을 파크 레인저(Park Ranger)라고 하는데 저희는 국립공원뿐만 아닌 더 나아가 지구를 대상으로 식생지수인 초록색을 지키고자 하는 마음이 있기 때문입니다.
   
데이터 전처리 과정에서 `QGIS` 프로그램과 `Python`을 사용했고, 웹 서비스 구축을 위해 `Django`를 사용했습니다.   
사용한 데이터로는 [Terra](https://search.earthdata.nasa.gov/search?q=C1621383370-LPDAAC_ECS&sp[0]=127%2C38&qt=2003-01-01T00%3A00%3A00.000Z%2C2021-12-31T23%3A59%3A59.999Z&lat=37.99951171875&long=127.001953125&zoom=7), [Aqua](https://search.earthdata.nasa.gov/search?q=MYD13Q1%20V061&sp[0]=127%2C38&qt=2003-01-01T00%3A00%3A00.000Z%2C2021-12-31T23%3A59%3A59.999Z&lat=37.99951171875&long=127.001953125&zoom=7), [임상도](http://data.nsdi.go.kr/dataset/20190716ds00001), [국립공원 보호지역](http://www.kdpa.kr/), 고정 카메라 이미지가 있습니다.
> 공개된 데이터인 경우에 데이터명을 클릭시 데이터를 다운 받을 수 있는 홈페이지로 이동합니다.
<br><br>
## 목차
1. [설치](#1-설치)
1. [시작](#2-시작)
1. [데이터 미리보기](#3-데이터-미리보기)
1. [문서](#4-문서)
1. [위대한 레인저들](#5-위대한-레인저들)
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
├── 데이터청년캠퍼스 - 레인저스
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
> 디렉토리 맨 앞에 있는 숫자는 지리정보 데이터에 대해 어려움이 있는 사람을 위해 이해를 돕고자 전처리 순서를 나타낸 것입니다.   
> ex1) 1_... + 1_... = 2_...&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ex2) 2_... + 2_... = 3_...

*[여기](https://drive.google.com/drive/folders/18wJPKxjIfvqn2lseMWll9AnG_dIO0vnh?usp=sharing)를 눌러 전체 데이터가 있는 구글 드라이브에 바로가기
<br><br>
## 4. 문서

- [preprocessing.py](https://github.com/gibum1228/knps_phenology/blob/main/document/preprocessing.md)
- [analysis.py](https://github.com/gibum1228/knps_phenology/blob/main/document/analysis.md)
- [model.py](https://github.com/gibum1228/knps_phenology/blob/main/document/model.md)
<br><br>
## 5. 위대한 레인저들
- 레인저스(Rangers)
  - [강유빈](https://github.com/yubbi4)
  - [김기범](https://github.com/gibum1228)
  - [안형주](https://github.com/HyungjooAhn1)
  - [정호찬](https://github.com/Eumgill98)
  - [주세현](https://github.com/shjoo0407)
   
- 경남대학교 빅리더 AI 아카데미
  - 전종식 교수님
  - 손영기 대표님

- 국립공원공단(KNPS)
   - 유병혁 과장님
   - 김진원 박사님
   - 박은하 박사님
   - 김지윤 박사님
