import os
import requests
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 環境設定
HF_TOKEN = os.environ.get("HF_TOKEN")
# 安定版のWhisper v3に変更
STT_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/google/gemma-3-270m"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_stt(filename):
    with open(filename, "rb") as f:
        data = f.read()
    
    # モデルが起動するまで最大3回リトライする
    for i in range(3):
        response = requests.post(STT_API_URL, headers=headers, data=data)
        if response.status_code == 200:
            return response.json().get("text", "")
        elif response.status_code == 503: # モデルがロード中の場合
            time.sleep(5)
            continue
        else:
            # HTMLが返ってきた場合などのためにエラー内容を分かりやすく
            raise Exception(f"STT API Error ({response.status_code}): {response.text[:100]}")
    return ""

def query_sentiment(text):
    prompt = f"Analyze the sentiment. Answer ONLY 'Positive', 'Negative', or 'Neutral'.\nText: {text}\nSentiment:"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 10, "temperature": 0.1, "return_full_text": False}
    }
    response = requests.post(SENTIMENT_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0]['generated_text'].strip().split('\n')[0]
    return "Neutral"

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        temp_filename = "temp_audio.wav"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        user_text = query_stt(temp_filename)
        
        if not user_text:
            return {"text": "(音声が認識できませんでした)", "sentiment": "Unknown"}

        sentiment = query_sentiment(user_text)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return {"text": user_text, "sentiment": sentiment}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="static", html=True), name="static")
