from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests

app = FastAPI()

# Allow frontend connections from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API for Zephyr 7B model
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {
    "Authorization": "Bearer #hugging face token",
    "Content-Type": "application/json"
}

def format_prompt(user_msg):
    return f"You are a fashion expert. Only respond with helpful fashion advice.\n\nUser: {user_msg}\nNOVA:"

@app.post("/fashion")
async def chat(request: Request):
    data = await request.json()
    prompt = format_prompt(data["message"])
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        output = response.json()
        text = output[0]['generated_text']
        reply = text.split("NOVA:")[-1].strip()
        return {"reply": reply}
    else:
        return {"reply": " Model is currently unavailable. Please try again later."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
