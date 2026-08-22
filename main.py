import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# df 준비 : 파일 읽어오기
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")


# df 전처리 : 결측치, 중복, 인코딩, 파생칼럼
train["Transported"] = train["Transported"].astype(int)

spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

for df in [train, test]:
    df[spend_cols] = df[spend_cols].fillna(0)
    df["TotalSpend"] = df[spend_cols].sum(axis=1)

    df["CryoSleep"] = df["CryoSleep"].fillna(df["TotalSpend"] == 0)
    df["CryoSleep"] = df["CryoSleep"].astype(int)

    df["Age"] = df["Age"].fillna(df.groupby("HomePlanet")["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["Group"].transform("count")
    df["HomePlanet"] = df["HomePlanet"].fillna(df.groupby("Group")["HomePlanet"].transform("first"))
    df["HomePlanet"] = df["HomePlanet"].fillna("Unknown")
    df["Destination"] = df["Destination"].fillna(df.groupby("Group")["Destination"].transform("first"))
    df["Destination"] = df["Destination"].fillna("Unknown")

    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = df["Deck"].fillna(
        df.groupby("Group")["Deck"].transform(lambda s: s.mode()[0] if s.notna().any() else None)
    )
    df["Side"] = df["Side"].fillna(
        df.groupby("Group")["Side"].transform(lambda s: s.mode()[0] if s.notna().any() else None)
    )
    df["Deck"] = df["Deck"].fillna(df["Deck"].mode()[0])
    df["Side"] = df["Side"].fillna(df["Side"].mode()[0])
# df에서 특징 고르기
num_features = ["CryoSleep", "Age", "GroupSize"] + spend_cols
cat_features = ["HomePlanet", "Destination", "Deck", "Side"]

x = pd.get_dummies(train[num_features + cat_features], columns=cat_features)
x_test = pd.get_dummies(test[num_features + cat_features], columns=cat_features)
x_test = x_test.reindex(columns=x.columns, fill_value=0)

y = train["Transported"]


# 학습
m = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
m.fit(x,y)


# 예측
result = m.predict(x_test)
test["Transported"] = result.astype(bool)


# 결과물 저장
test[["PassengerId", "Transported"]].to_csv("data/result.csv", index=False)


# print(train.info())
# print(test.info())
# print(train.isna().sum())
# print()
# print(test.isna().sum())
#
# x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
#
# for d in [3, 5, 8, 12, None]:
#     m2 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=d)
#     m2.fit(x_tr, y_tr)
#     print(f"max_depth={str(d):>4} → {m2.score(x_val, y_val):.4f}")