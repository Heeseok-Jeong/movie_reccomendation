# 패키지 import
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# data = pd.read_csv('./ml-25m/ratings.csv')
data = pd.read_csv('./ml-latest-small/ratings.csv')
print(data.shape)

# 데이터를 Train과 Validation(=Test)데이터 6:4 비율로 나눔
np.random.seed(3)
msk1 = np.random.rand(len(data)) < 0.6
train = data[msk1].copy()
val = data[~msk1].copy()
test = data[~msk1].copy()
print(train.head())
print(train.shape)
print(val.head())
print(val.shape)
print(test.head())
print(test.shape)

# 다음은 Pandas의 컬럼을 범주형의 id로 인코드해주는 함수이다
# train_col이 있을 때(valid or test), 여기 없는 사용자 id나 영화이름을 가진 데이터는 사라짐
def proc_col(col, train_col=None):
    """ Encodes a pandas column with continous ids. """
    # Unique한 row를 찾는다 즉 사용자 혹은 영화이다
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    # 사용자/영화를 인덱스와 매핑해준다
    name2idx = {o:i for i,o in enumerate(uniq)}
    # 그리고 그것을 포맷팅해서 리턴한다
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

# 다음은 실제로 데이터를 인코딩으로 만들어주는 함수이다
# 위에서 정의해준 proc_col을 사용한다
def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids.
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        n2i,col,len_uniq = proc_col(df[col_name], train_col)
        #n2i는 트레인 기준으로 만듦
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

num_users = len(df_train.userId.unique())
num_items = len(df_train.movieId.unique())

# 학습된 모델 평가하고 정답과 비교해서 로스 얻음
def validation_loss(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(df_val.userId.values)
    items = torch.LongTensor(df_val.movieId.values)
    ratings = torch.FloatTensor(df_val.rating.values)
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    print("y_hat :", y_hat, ",", y_hat.shape)
    loss = F.mse_loss(y_hat, ratings)
    print("validation loss {:.3f}".format(loss.item()))

# 학습 함수
# 로스가 가장 낮은 에포크의 모델을 베스트 모델로 저장
def train_mf(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    print("Train,", "Learning Rate :", lr)
    best_loss = 100
    #저장 경로 파일 생성
    best_model_path = "./results/model.lr" + str(lr)[2:] + ".epochs" + str(epochs) + ".best.pth"
    with open(best_model_path, "w") as f:
            content = ""
            f.write(content)
    loss_path = "./results/model.lr" + str(lr)[2:] + ".epochs" + str(epochs) + ".loss.csv"
    with open(loss_path, "w") as f:
            content = "epochs,\t\tloss\n"
            f.write(content)
    print(best_model_path)
    print(loss_path)

    #학습 시작
    start = time.time()
    for i in range(epochs):
        users = torch.LongTensor(df_train.userId.values)
        items = torch.LongTensor(df_train.movieId.values)
        ratings = torch.FloatTensor(df_train.rating.values)
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #로스 저장
        with open(loss_path, "a") as f:
            content = str(i+1) + ",\t\t" + str(loss.item()) +"\n"
            f.write(content)
        #베스트모델 저장
        if best_loss > loss.item():
            best_loss = loss.item()
            print("Best model is saved, i :", i+1)
            torch.save(model, best_model_path)
        if i%10 == 9:
            print("epoch", i+1, ":", loss.item(), ", Best_loss :", best_loss)
    print()
    print("Valid")
    validation_loss(model, unsqueeze)
    print()
    print("Total time :", round(time.time()-start, 4), "seconds")

# 해당 유저에게 사이즈만큼의 영화를 추천해줌
# 테스트데이터 -> 해당유저데이터만 남김 -> 예측값 얻음
# -> 데이터에 예측값 붙임 -> 영화이름도 붙임 -> 정렬
# -> 개수만큼 출력
def test(model, userId=1, size=5):
    new_df_test = df_test.copy()
    userId -= 1

    #df_test(dataframe) 에서 userId거만 남기기
    is_userId = new_df_test['userId'] == userId
    new_df_test = new_df_test[is_userId]
#     print(new_df_test)

    model.eval()
    users = torch.LongTensor(new_df_test.userId.values)
    items = torch.LongTensor(new_df_test.movieId.values)
    y_hat = model(users, items)
#     print(type(y_hat))
#     print("y_hat :", y_hat, ",", y_hat.shape)

    #y_hat(tensor)를 predictions(pandas dataframe)으로 변환
    predictions = y_hat.detach().numpy()
    predictions = pd.DataFrame(predictions)
    predictions.columns = ["predictions"]
#     print()
#     print("predictions")
#     print(predictions)
#     print(predictions.head())

    #df_test(dataframe)에 predictions 붙이기 => new_df
    new_df_test.index = [i for i in range(len(new_df_test))]
#     print()
#     print("new_df_test")
#     print(new_df_test)
    new_df = pd.concat([new_df_test, predictions], axis=1)
#     print()
#     print("new_df")
#     print(new_df)

    #movie 이름 붙이기
    movies = pd.read_csv('./ml-latest-small/movies.csv')
    new_df = pd.merge(new_df, movies, on='movieId')
#     print()
#     print("movie name")
#     print(new_df)

    #predictions로 sort
    new_df = new_df.sort_values(by="predictions", ascending=False)
#     print()
#     print("sort")
#     print(new_df.head(10))

    #size만큼만 남기기
    new_df = new_df[:size]
    print()
    print("size")
    print(new_df)

    #결과 출력
    result = []
    new_df.index = [i for i in range(len(new_df))]
    print()
    print("Movie Recommendation for User", userId)
    for i in range(len(new_df)):
        result.append(new_df["title"][i])
        print(i+1, ":", result[i])

# 딥러닝 협업필터링 모델
# 임베딩 100짜리 2개의 레이어
class NNCollabFiltering(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, n_hidden=10):
        super(NNCollabFiltering, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.lin1 = nn.Linear(emb_size*2, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.drop1 = nn.Dropout(0.1)

    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        x = F.relu(torch.cat([U, V], dim=1))
        x = self.drop1(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# 모델 학습 시작 
# 얘가 valid 로스가 제일 낮고, 그래프가 오버피팅처럼 튀는 것이 없음 => 이 모델 쓰자
model_6 = NNCollabFiltering(num_users, num_items, emb_size=100)
train_mf(model_6, epochs=1000, lr=0.0005, wd=1e-6, unsqueeze=True)

# 모델 설정해서 예측 시작
PATH = "./results/model.lr0005.epochs1000.best.pth"
model = torch.load(PATH)
model.eval()
test(model, userId=1, size=10)
