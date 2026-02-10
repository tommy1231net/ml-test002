from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. FastAPIのインスタンス化
app = FastAPI()

# 2. CORSの設定（重要！）
# 静的HTML（別のドメイン）から呼び出すために必須の設定です
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS設定を最強（何でも許可）に更新
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # すべてのドメインを許可
    allow_origin_regex=".*",         # nullオリジン対策：正規表現ですべて許可
    allow_credentials=False,         # "*" を使う場合は False にする必要があります
    allow_methods=["*"],             # GET, POST, OPTIONS 等すべて
    allow_headers=["*"],             # すべてのヘッダーを許可
)

# モデルとカラム情報の読み込み
model = joblib.load('penguin_model.joblib')
model_columns = joblib.load('model_columns.joblib')

# リクエストデータの型定義
class PenguinData(BaseModel):
    species: str
    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    sex: str

@app.get("/")
def read_root():
    return {"status": "Penguin Prediction API is running"}

@app.post("/predict")
def predict(data: PenguinData):
    # 1. 受信したデータをDataFrameに変換
    input_df = pd.DataFrame([data.dict()])
    
    # 2. 学習時と同じOne-Hot Encodingを適用
    input_df = pd.get_dummies(input_df)
    
    # 3. 学習時のカラム構成と一致させる（足りない列を0で埋める）
    final_df = pd.DataFrame(columns=model_columns)
    final_df = pd.concat([final_df, input_df]).fillna(0)
    final_df = final_df[model_columns]
    
    # 4. 予測実行
    prediction = model.predict(final_df)[0]
    
    return {"predicted_body_mass_g": float(prediction)}

if __name__ == "__main__":
    import uvicorn
    import os
    # Cloud Runが指定する環境変数PORTを取得。なければ8080
    port = int(os.environ.get("PORT", 8080))
    # 0.0.0.0 で待機しないと外部（Cloud Run）からアクセスできません
    uvicorn.run(app, host="0.0.0.0", port=port)