import os
import requests
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 1. 環境変数の確認
HF_TOKEN = os.environ.get("HF_TOKEN")
# より安定している 'small' モデルに変更してみます
STT_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/google/gemma-3-270m"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_stt(filename):
    with open(filename, "rb") as f:
        data = f.read()
    
    # モデルが準備できるまで最大3回リトライ
    for i in range(3):
        response = requests.post(STT_API_URL, headers=headers, data=data)
        
        if response.status_code == 200:
            return response.json().get("text", "")
        elif response.status_code == 503: # モデル読み込み中
            print(f"Model loading... wait 5s (Attempt {i+1})")
            time.sleep(5)
            continue
        elif response.status_code == 404:
            # 404の場合はモデル名を変更して再試行（予備：whisper-tiny）
            print("Model not found, trying fallback model...")
            fallback_url = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
            response = requests.post(fallback_url, headers=headers, data=data)
            if response.status_code == 200:
                return response.json().get("text", "")
        
        # それでもダメならエラー詳細を出す
        raise Exception(f"STT Error {response.status_code}: {response.text[:100]}")
    return ""

def query_sentiment(text):
    # 文言をさらにシンプルにしてAPIの負荷を下げます
    prompt = f"Text: {text}\nSentiment (Positive/Negative/Neutral):"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 5, "temperature": 0.1}
    }
    response = requests.post(SENTIMENT_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        # 生成されたテキストから感情だけを抽出
        res_text = result[0]['generated_text'].lower()
        if "positive" in res_text: return "Positive"
        if "negative" in res_text: return "Negative"
        return "Neutral"
    return "Neutral"

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        temp_filename = "temp_audio.wav"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        # 1. 文字起こし
        user_text = query_stt(temp_filename)
        
        # 2. 感情分析
        sentiment = "Unknown"
        if user_text:
            sentiment = query_sentiment(user_text)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return {"text": user_text if user_text else "(音声が聞き取れませんでした)", "sentiment": sentiment}

    except Exception as e:
        print(f"System Error: {e}")
        # クライアントに具体的なエラーを返しすぎないよう調整
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="static", html=True), name="static")
