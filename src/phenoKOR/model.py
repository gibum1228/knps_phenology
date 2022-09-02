import numpy as np
import pandas as pd
import torch
from prophet import Prophet
import pmdarima as pm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader  # 데이터로더
from torch.utils.data import TensorDataset  # 텐서데이터셋

import preprocessing

# 전역변수
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 파이토치 gpu로 돌리기
ROOT, MIDDLE = preprocessing.get_info()


def fit_prophet():
    df = preprocessing.get_final_data()
    df = df[['date', 'avg']]
    df.columns = ['ds', 'y']

    model = Prophet(daily_seasonality=True)
    model.add_seasonality(name='yearly', period=365, fourier_order=10, mode='additive')

    model.fit(df, algorithm='Newton')

    forecast = 730  # forecast만큼 이후를 예측
    df_forecast = model.make_future_dataframe(periods=forecast)  # 예측할 ds 만들기
    df_forecast = model.predict(df_forecast)  # 비어진 ds만큼 예측

    model.plot(df_forecast, xlabel="date", ylabel="evi", figsize=(30, 10))
    model.plot_components(df_forecast)
    plt.show()


# LSTM 네트워크 구조
class LSTM(torch.nn.Module):
    # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, hidden_dim, step, output_dim, layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.step = step
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = torch.nn.LSTM(1, hidden_dim, num_layers=layers,
                                  dropout=dropout,
                                  batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

    # 학습 초기화를 위한 함수
    def reset_hidden_state(self):
        self.hidden = (torch.zeros(self.layers, self.step, self.hidden_dim),
                       torch.zeros(self.layers, self.step, self.hidden_dim))

    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x


# 모델 학습하기
def train_LSTM(model, train_df, num_epochs=None, verbose=1, patience=5):
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


# 스텝별 학습 데이터 저장하기
def split_data(df, step, y_count):
    x, y = [], []

    for i in range(step, len(df)):
        x.append(df[i - step:i])
        y.append(df[i:i + y_count])

    return np.reshape(np.array(x), (len(x), step, 1)), np.array(y)


# LSTM 학습 자동화 코드
def fit_LSTM():
    save_df = pd.DataFrame(columns=["code", "class_num", "step", "batch_size", "hidden_dim",
                                    "dropout", "layer", "r^2", "rmse", "mape"])

    # step, y_count: step개의 데이터로 y_count개의 데이터를 학습
    option = {
        "step": range(5, 11), "y_count": 1, "batch_size": [128, 256], "hidden_dim": range(5, 11),
        "dropout": [0.3, 0.4, 0.5], "layer": [1, 2], "knps_name": preprocessing.get_knps_name_en(),
        "class_num": range(4),
        "epoch": 500
    }

    for knps_name in option["knps_name"]:
        for class_num in option["class_num"]:
            for step in option["step"]:
                for batch_size in option["batch_size"]:
                    for hidden_dim in option["hidden_dim"]:
                        for dropout in option["dropout"]:
                            for layer in option["layer"]:
                                db = {
                                    "knps": knps_name,
                                    "class_num": class_num,
                                    "start_year": "2003",
                                    "end_year": "2021"
                                }
                                df = preprocessing.get_final_data(db)

                                # 학습 데이터(03-19)와 테스트 데이터(20-21) 분리
                                df_train = (df[df['date'] < "2020-01-01"])['avg'].values
                                df_test = (df[df['date'] >= "2020-01-01"])['avg'].values

                                # 학습 데이터와 테스트 데이터 만들기
                                train_x, train_y = split_data(df_train, option["step"], option["y_count"])
                                test_x, test_y = split_data(df_test, option["step"], option["y_count"])

                                print("데이터 크기 확인")
                                print("train X Shape: ", train_x.shape)
                                print("train Y Shape: ", train_y.shape)
                                print("test X Shape: ", test_x.shape)
                                print("test Y Shape: ", test_y.shape)

                                # 실제 데이터 테이블 확인
                                X_train_see = pd.DataFrame(np.reshape(train_x, (train_x.shape[0], train_x.shape[1])))
                                y_train_see = pd.DataFrame(train_y)
                                X_test_see = pd.DataFrame(np.reshape(test_x, (test_x.shape[0], test_x.shape[1])))

                                # 데이터를 Tensor로 변경
                                train_x_tensor = torch.FloatTensor(train_x).to(device)
                                train_y_tensor = torch.FloatTensor(train_y).to(device)
                                test_x_tensor = torch.FloatTensor(test_x).to(device)
                                test_y_tensor = torch.FloatTensor(test_y).to(device)

                                # Tensor 형태로 데이터셋 정의
                                dataset = TensorDataset(train_x_tensor, train_y_tensor)
                                dataloader = DataLoader(dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False)

                                # 모델 학습
                                net = LSTM(hidden_dim, step, 1, 1, dropout).to(device)
                                model, train_hist = train_LSTM(net, dataloader, num_epochs=option["epoch"])

                                # 예측 테스트
                                with torch.no_grad():
                                    pred = []
                                    for pr in range(len(test_x_tensor)):
                                        model.reset_hidden_state()

                                        predicted = model(torch.unsqueeze(test_x_tensor[pr], 0))
                                        predicted = torch.flatten(predicted).item()
                                        pred.append(predicted)

                                r2 = round(R2(test_y_tensor.cpu(), pred), 6)
                                rmse = round(RMSE(test_y_tensor.cpu(), pred), 6)
                                mape = round(MAPE(test_y_tensor.cpu(), pred), 6)

                                save_df.iloc[len(save_df)] = [knps_name, class_num, step, batch_size, hidden_dim,
                                                              dropout, layer, r2, rmse, mape]

    save_df.to_csv(f"{ROOT}data{MIDDLE}lstm_final.csv", index=False)
    # plt.figure(figsize=(8, 3))
    # plt.plot(np.arange(len(pred)), pred, color='red', label="pred")
    # plt.plot(np.arange(len(test_y_tensor)), test_y_tensor.cpu(), color='blue', label="true")
    # plt.title("Loss plot for 500 epoch")
    # plt.xlabel("Epoch")
    # plt.ylabel("loss")
    # plt.legend()
    # plt.show()


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


# 전체 모델별 검증 지표 비교하기
def model_compare():
    # 각 모델별 검증지표 데이터 로드
    data_arima = pd.read_csv(f"{ROOT}data{MIDDLE}arima_final.csv", usecols=[3, 4, 5])
    data_lstm = pd.read_csv(f"{ROOT}data{MIDDLE}lstm_final.csv", usecols=[0,1,6,7,8])
    data_prophet = pd.read_csv(f"{ROOT}data{MIDDLE}prophet_final.csv", usecols=[3,4,5])


    data_prophet['mape'] = data_prophet['mape'] / 100  # MAPE 스케일 맞추기

    # 컬럼명 재설정
    data_arima.columns = ['R2_arima', 'RMSE_arima', 'MAPE_arima']
    data_prophet.columns = ['R2_prophet', 'RMSE_prophet', 'MAPE_prophet']
    data_lstm.columns = ['code', 'class_num', 'R2_lstm', 'RMSE_lstm', 'MAPE_lstm']

    # 검증 지표 합치기
    final = pd.concat([data_lstm, data_prophet, data_arima], axis=1)

    r2 = final[['R2_lstm', 'R2_prophet', 'R2_arima']]
    rmse = final[['RMSE_lstm', 'RMSE_prophet', 'RMSE_arima']]
    mape = final[['MAPE_lstm', 'MAPE_prophet', 'MAPE_arima']]

    # 최적의 지표 선택
    max_r2 = r2.max(axis=1)
    min_rmse = rmse.min(axis=1)
    min_mape = mape.min(axis=1)
    max_r2_idx = r2.idxmax(axis=1)
    min_rmse_idx = rmse.idxmin(axis=1)
    min_mape_idx = mape.idxmin(axis=1)

    max_final = pd.concat([final['code'], final['class_num'], max_r2, max_r2_idx, min_rmse, min_rmse_idx, min_mape, min_mape_idx], axis=1)
    max_final.columns =['code', 'class_num', 'max_r2','max_r2_idx','min_rmse','min_rmse_idx', 'min_mape','min_mape_idx']
    max_final.to_csv(f"{ROOT}data{MIDDLE}modelcompare_result.csv")


    # Prophet과 Othermodel 비교하기
    r2_other = final[['R2_lstm', 'R2_arima']]
    rmse_other = final[['RMSE_lstm', 'RMSE_arima']]
    mape_other = final[['MAPE_lstm', 'MAPE_arima']]

    # Other 모델에서 가장 작은 값 선택
    min_rmse_other = rmse_other.min(axis=1)
    min_mape_other = mape_other.min(axis=1)

    # Prophet 모델과 Other 모델 차이값 계산
    rmse_diff = min_rmse_other - final['RMSE_prophet']
    mape_diff = min_mape_other - final['MAPE_prophet']

    # 최종 비교
    prop_other_final = pd.concat(
        [final['code'], final['class_num'], final['RMSE_prophet'], min_rmse_other, rmse_diff, final['MAPE_prophet'],
         min_mape_other, mape_diff], axis=1)
    prop_other_final.columns = ['code', 'class_num', 'rmse_prophet', 'rmse_other', 'rmse_diff', 'mape_prophet',
                                'mape_other', 'mape_diff']

    prop_other_final.to_csv(f"{ROOT}data{MIDDLE}final_compare_result.csv")


# Arima 학습 및 예측 자동화
def arima():
    R2_list = []
    RMSE_list = []
    MAPE_list = []
    cls = []
    code = []

    for knps in preprocessing.get_knps_name_en():
        for i in range(4):
            data = preprocessing.get_final_data(knps, i)
            data['date'] = pd.to_datetime(data['date'])

            # 학습 데이터(03-19)와 테스트 데이터(20-21) 분리
            train_data = data[data['date'] <= '2020-01-01']
            test_data = data[data['date'] >= '2020-01-01']

            # ARIMA MODEL
            model = pm.ARIMA(order=(2, 1, 0),   # (p,d,q)
                             seasonal_order=(2, 1, 0, 46),  # 계절 파라미터 (P,D,Q)
                             scoring='mse'
                             )
            model_fit = model.fit(train_data['avg'])
            model_predict = model_fit.predict(len(test_data['avg']))

            R2_list.append(R2(test_data['avg'], model_predict))
            RMSE_list.append(RMSE(test_data['avg'], model_predict))
            MAPE_list.append(MAPE(test_data['avg'], model_predict))
            cls.append(i)
            code.append(knps)

    df = pd.DataFrame({'park': code, 'class': cls, 'R2': R2_list, 'RMSE': RMSE_list, 'MAPE': MAPE_list})

    df.to_csv(f"{ROOT}data{MIDDLE}arima_final.csv")
