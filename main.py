import os
import requests
import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from huggingface_hub import login

app = FastAPI()

# 1. 環境設定
HF_TOKEN = os.environ.get("HF_TOKEN")
# Gemma 3-270m の推論APIエンドポイント
API_URL = "https://api-inference.huggingface.co/models/google/gemma-3-270m"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 2. Whisper（文字起こし）はローカルで実行（baseモデルならメモリ200MB程度で済みます）
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper loaded!")

# 3. 感情分析（Hugging Face API経由）
def analyze_sentiment_api(text: str):
    prompt = f"Analyze the sentiment of the following Japanese text. Answer only with 'Positive', 'Negative', or 'Neutral'.\n\nText: {text}\nSentiment:"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 10,
            "temperature": 0.1,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        # モデルがロード中の場合は503が返ることがあります
        raise Exception(f"API Error: {response.text}")
        
    result = response.json()
    # APIのレスポンス形式に合わせて抽出
    sentiment = result[0]['generated_text'].strip().split('\n')[0]
    return sentiment

# 4. 音声解析エンドポイント
@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        temp_filename = "temp_audio.wav"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        # 文字起こし
        audio_result = whisper_model.transcribe(temp_filename, language="ja")
        user_text = audio_result["text"]
        
        if not user_text.strip():
            return {"text": "(音声が聞き取れませんでした)", "sentiment": "Unknown"}

        # APIで感情分析
        sentiment = analyze_sentiment_api(user_text)
        
        os.remove(temp_filename)
        return {"text": user_text, "sentiment": sentiment}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 5. フロントエンド配信
app.mount("/", StaticFiles(directory="static", html=True), name="static")