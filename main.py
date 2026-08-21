import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# print(train.info())
# print(test.info())
# print(train.isna().sum())
# print(test.isna().sum())


# df 준비 : 파일 읽어오기
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")


# df 전처리 : 결측치, 중복, 인코딩, 파생칼럼
spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

for df in [train, test]:
    df[spend_cols] = df[spend_cols].fillna(0)
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["CryoSleep"] = df["CryoSleep"].fillna(df["TotalSpend"] == 0)
    df["CryoSleep"] = df["CryoSleep"].astype(int)
    df["Age"] = df["Age"].fillna(df.groupby("HomePlanet")["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())

train["Transported"] = train["Transported"].astype(int)

# # df에서 특징 고르기
features = ["CryoSleep", "Age"]
x = train[features]
y = train["Transported"]

# 학습
m = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
m.fit(x,y)

# 예측
result = m.predict(test[features])
test["Transported"] = result.astype(bool)

# 결과물 저장
test[["PassengerId", "Transported"]].to_csv("data/result.csv", index=False)