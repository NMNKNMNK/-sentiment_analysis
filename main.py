import os
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from huggingface_hub import login

app = FastAPI()

# 環境設定
HF_TOKEN = os.environ.get("HF_TOKEN")
# 音声認識(Whisper)と感情分析(Gemma 3)のAPIエンドポイント
STT_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/google/gemma-3-270m"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 1. APIで文字起こしを実行する関数
def query_stt(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(STT_API_URL, headers=headers, data=data)
    if response.status_code != 200:
        raise Exception(f"STT API Error: {response.text}")
    return response.json().get("text", "")

# 2. APIで感情分析を実行する関数
def query_sentiment(text):
    prompt = f"Analyze the sentiment of the following Japanese text. Answer only with 'Positive', 'Negative', or 'Neutral'.\n\nText: {text}\nSentiment:"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 10, "temperature": 0.1, "return_full_text": False}
    }
    response = requests.post(SENTIMENT_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Sentiment API Error: {response.text}")
    
    result = response.json()
    return result[0]['generated_text'].strip().split('\n')[0]

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # 音声ファイルを一時保存
        temp_filename = "temp_audio.wav"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        # 1. APIで文字起こし
        user_text = query_stt(temp_filename)
        
        if not user_text or not user_text.strip():
            return {"text": "(音声が聞き取れませんでした)", "sentiment": "Unknown"}

        # 2. APIで感情分析
        sentiment = query_sentiment(user_text)
        
        os.remove(temp_filename)
        return {"text": user_text, "sentiment": sentiment}

    except Exception as e:
        print(f"Error detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="static", html=True), name="static")
