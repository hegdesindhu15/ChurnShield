import requests

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
headers = { "Content-Type": "application/json" }
params = { "key": "AIzaSyDk5Ah_HgJhsL2dbLH9p4ziRimgCCX0qRA" }

payload = {
  "contents": [{
    "parts": [{ "text": "Give 3 retention strategies for a vehicle insurance customer with high churn risk and frequent complaints." }]
  }]
}

response = requests.post(url, headers=headers, params=params, json=payload)
print(response.status_code)
print(response.text)
