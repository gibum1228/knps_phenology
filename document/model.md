# model.py

## 목차

1. [fit_prophet()](#fit_prophet)
2. [LSTM](#LSTM)
3. [train_LSTM()](#train_LSTM)
4. [split_data()](#split_data)
5. [fit_LSTM()](#fit_LSTM)
6. [model_compare()](#model_compare)
7. [arima()](#arima)
8. [MSE()](#MSE)
9. [RMSE()](#RMSE)
10. [R2()](#R^2)
11. [MAPE](#MAPE)


## fit_prophet

`매개변수`: 없음   
`기능` : prophet 모델을 사용해 'Newton' 알고리즘을 적용하여 'forecast' 에 명시된 일 수 만큼 예측을 진행하는 메소드.   
`리턴값` : 없음. 단, plot_components는 추세와 주별,연별 주기 그래프를 보여줌.   

## LSTM

`속성` : hiddnen_dim, step, output_dim, layers, lstm, fc   
`기능` : LSTM 네트워크 구조를 설정해주는 클래스      

## train_LSTM

`매개변수` : model, train_df:pandas.DataFrame, num_epochs:int, verbose:int, patience:int   
`기능` : 에포크마다 모델을 학습하는 메소드   
`리턴값` : model, train_hist   

## split_data

`매개변수` : df:pandas.DataFrame, steop:int, y_count:int   
`기능` : 데이터를 스텝별로 나눠 학습 데이터와 테스트 데이터를 저장하는 메소드    
`리턴값` : numpy.ndarray   

## fit_LSTM

`매개변수` : 없음   
`기능` : LSTM을 학습하는 과정을 자동화한 메소드    
`리턴값` : 없음   

## model_compare

`매개변수` : 없음   
`기능` : 전체 모델별 r2, rmse, mape 비교하여 최적의 모델 선정하는 메소드   
`리턴값` : 없음   

## arima

`매개변수` : 없음    
`기능` : ARIMA 모델을 사용하여 데이터를 학습 후 예측한 뒤 r2, rmse, mape 계산하는 메소드   
`리턴값` : 없음   

## MSE

`매개변수` : y:numpy.ndarray, pred_y.ndarray    
`기능` : 평균 제곱 오차 구하는 메소드    
`리턴값` : ndarray     

## RMSE

`매개변수` : y:numpy.ndarray, pred_y.ndarray    
`기능` : 평균 제곱근 편차 구하는 메소드   
`리턴값` : ndarray   

## R^2

`매개변수` : y:numpy.ndarray, pred_y.ndarray    
`기능` : 결정계수 구하는 메소드   
`리턴값` : ndarray   

## MAPE

`매개변수` : y:numpy.ndarray, pred_y.ndarray    
`기능` : 평균 절대비오차 구하는    
`리턴값` : ndarray   
