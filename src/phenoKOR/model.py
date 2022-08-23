import math

import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima_model import ARIMA
from prophet import Prophet
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더


'''데이터를 로드하는 메소드'''
def load_data(knps, class_num):
    df = pd.read_csv(os.getcwd() + "/data/knps_final.csv")  # 데이터 가져오기

    # 조건에 맞는 데이터 추출
    return df[(df["code"] == knps) & (df["class"] == class_num)].sort_values('date')


'''윤년 구하는 메소드'''
def get_Feb_day(year):
    # 4, 100, 400으로 나누어 떨어진다면 윤년
    if year % 4 == 0 or year % 100 == 0 or year % 400 == 0:
        day = 29
    else:
        day = 28

    return day


'''해당하는 모델로 학습하는 메소드'''
def fit_arima():
    pass


def fit_prophet():
    df = load_data()

    model = Prophet()
    # model.stan_backend.set_options(newton_fallback=False)
    model.fit(df, algorithm='Newton')

    forecast = 730 # forecast만큼 이후를 예측
    df_forecast = model.make_future_dataframe(periods=forecast) # 예측할 ds 만들기
    df_forecast = model.predict(df_forecast) # 비어진 ds만큼 예측

    # 시각화
    # model.plot(df_forecast, xlabel="date", ylabel="evi", figsize=(25, 15))
    # plt.show()


def fit_random_forest():
    pass


def fit_LSTM():
    df = load_data("gaya", 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # time_steps개의 데이터로 for_periods가 나온다고 학습 -> 후에는 이와 같은 방법으로 예측
    time_steps, for_periods = 7, 1
    # 학습 데이터와 테스트 데이터 분리
    ts_train = (df[df['date'] < "2020-01-01"])['avg'].values
    ts_test = (df[df['date'] >= "2020-01-01"])['avg'].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # 스텝별 학습 데이터 저장하기
    X_train, y_train = [], []
    for i in range(time_steps, ts_train_len):
        X_train.append(ts_train[i - time_steps:i])
        y_train.append(ts_train[i:i + for_periods])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # 학습 데이터를 3차원으로 변환
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 스텝별 테스트 데이터 저장
    X_test, y_test = [], []
    for i in range(time_steps, ts_test_len):
        X_test.append(ts_test[i - time_steps:i])
        y_test.append(ts_test[i:i + for_periods])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print("데이터 크기 확인")
    print("train X Shape: ", X_train.shape)
    print("train Y Shape: ", y_train.shape)
    print("test X Shape: ", X_test.shape)
    print("test Y Shape: ", y_test.shape)

    # 실제 데이터 테이블 확인
    # Convert the 3D shape of X_train to a data frame so we can see:
    X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0], X_train.shape[1])))
    y_train_see = pd.DataFrame(y_train)
    X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0], X_test.shape[1])))

    # 데이터를 Tensor로 변경
    train_x_tensor = torch.FloatTensor(X_train).to(device)
    train_y_tensor = torch.FloatTensor(y_train).to(device)
    test_x_tensor = torch.FloatTensor(X_test).to(device)
    test_y_tensor = torch.FloatTensor(y_test).to(device)

    # Tensor 형태로 데이터셋 정의
    dataset = TensorDataset(train_x_tensor, train_y_tensor)
    dataloader = DataLoader(dataset,
                            batch_size=128,
                            shuffle=False)

    class LSTM(torch.nn.Module):
        # 기본변수, layer를 초기화해주는 생성자
        def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
            super(LSTM, self).__init__()
            self.hidden_dim = hidden_dim
            self.seq_len = seq_len
            self.output_dim = output_dim
            self.layers = layers

            self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                                      dropout = 0.5,
                                      batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

        # 학습 초기화를 위한 함수
        def reset_hidden_state(self):
            self.hidden = (torch.zeros(self.layers, self.seq_len, self.hidden_dim),
                           torch.zeros(self.layers, self.seq_len, self.hidden_dim))

        # 예측을 위한 함수
        def forward(self, x):
            x, _status = self.lstm(x)
            x = self.fc(x[:, -1])
            return x

    def train_model(model, train_df, num_epochs=None, verbose=10, patience=10):

        criterion = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        nb_epochs = num_epochs

        # epoch마다 loss 저장
        train_hist = np.zeros(nb_epochs)

        for epoch in range(nb_epochs):
            avg_cost = 0
            total_batch = len(train_df)

            for batch_idx, samples in enumerate(train_df):
                x_train, y_train = samples

                # seq별 hidden state reset
                model.reset_hidden_state()

                # H(x) 계산
                outputs = model(x_train)

                # cost 계산
                loss = criterion(outputs, y_train)

                # cost로 H(x) 개선
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_cost += loss / total_batch

            train_hist[epoch] = avg_cost

            if epoch % verbose == 0:
                print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))

            # patience번째 마다 early stopping 여부 확인
            if (epoch % patience == 0) & (epoch != 0):

                # loss가 커졌다면 early stop
                if train_hist[epoch - patience] < train_hist[epoch]:
                    print('\n Early Stopping')

                    break

        return model.eval(), train_hist

    # 모델 학습
    net = LSTM(1, 10, 7, 1, 1).to(device)
    model, train_hist = train_model(net, dataloader, num_epochs=500, verbose=1, patience=5)

    # 예측 테스트
    with torch.no_grad():
        pred = []
        for pr in range(len(test_x_tensor)):
            model.reset_hidden_state()

            predicted = model(torch.unsqueeze(test_x_tensor[pr], 0))
            predicted = torch.flatten(predicted).item()
            pred.append(predicted)

    print('R2 SCORE : ', round(R2(test_y_tensor.cpu(), pred), 6))
    print('RMSE SCORE : ', round(RMSE(test_y_tensor.cpu(), pred), 6))
    print('MAPE SCORE : ', round(MAPE(test_y_tensor.cpu(), pred), 6))

    # plt.figure(figsize=(8, 3))
    # plt.plot(np.arange(len(pred)), pred, color='red', label="pred")
    # plt.plot(np.arange(len(test_y_tensor)), test_y_tensor.cpu(), color='blue', label="true")
    # plt.title("Loss plot for 500 epoch")
    # plt.xlabel("Epoch")
    # plt.ylabel("loss")
    # plt.legend()
    # plt.show()

'''모델 검증하는 메소드'''
# MSE
def MSE(y, pred_y):
    return mean_squared_error(y, pred_y)


# RMSE
def RMSE(y, pred_y):
    return MSE(y, pred_y) ** 0.5


# R^2
def R2(y, pred_y):
    return r2_score(y, pred_y)


# MAPE
def MAPE(y, pred_y):
    return mean_absolute_percentage_error(y, pred_y)