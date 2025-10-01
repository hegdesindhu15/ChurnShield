# utils/gemini_utils.py
import requests
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA_vUu6GkPehCtjznsb-PfcIP5KbtZsclc")

def explain_plot(title, summary):
    prompt = (
        f"You are a churn analysis assistant. Help a retention manager understand this churn case:\n"
        f"Chart Title: {title}\nChart Summary: {summary}\n"
        f"Explain clearly, referring to the feature impacts. Be concise, professional, and data-informed."
    )

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    params = {"key": GEMINI_API_KEY}

    try:
        response = requests.post(url, headers=headers, json=payload, params=params)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[AI EXPLAIN ERROR] {e}")
        return "Unable to generate AI explanation."
