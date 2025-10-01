import requests
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA_vUu6GkPehCtjznsb-PfcIP5KbtZsclc")  # or hardcode for now

def explain_plot(title, summary, plot_data=None):
    """
    Use Gemini API to generate an explanation for a chart.
    Optionally include actual chart data for better response.
    """
    try:
        values_text = ""
        if plot_data:
            labels = plot_data.get("labels", [])
            data = plot_data.get("data", [])
            if labels and data and len(labels) == len(data):
                values_text = "\nChart Data:\n" + "\n".join(
                    f"- {lbl}: {round(val * 100, 2)}%" if isinstance(val, (int, float)) and "churn" in lbl.lower()
                    else f"- {lbl}: {val}"
                    for lbl, val in zip(labels, data)
                )

        prompt = (
            f"You are a churn analysis assistant. Explain this chart to a product manager.\n"
            f"Chart Title: {title}\nChart Summary: {summary}\n"
            f"{values_text}\n"
            f"Be concise (3-4 lines max), helpful, and avoid repeating the title. Focus on trends and insight."
        )

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        params = {"key": GEMINI_API_KEY}

        response = requests.post(url, headers=headers, json=payload, params=params)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    except Exception as e:
        print(f"[AI EXPLAIN ERROR] {e}")
        return "Unable to generate explanation."
    

def explain_top_churn_drivers(top_features_text, raw_data):
    """
    Use Gemini API to generate an explanation for top churn drivers in vehicle insurance.
    """
    try:
        # Extract top features from the text
        top_features = []
        for line in top_features_text.split('\n'):
            if line.startswith('-'):
                feature_name = line.split(':')[0].replace('- ', '').strip()
                top_features.append(feature_name)
        
        # Include actual values for the top features
        relevant_data = "\n".join(
            [f"- {key}: {raw_data[key]}" for key in raw_data if key in top_features]
        )

        prompt = (
            f"You are a churn analysis assistant for a vehicle insurance platform. Below are the top features contributing to a customer's churn risk, in the format '- Feature: SHAP value'. "
            f"Also provided are the customer's actual values for these features. "
            f"Provide a concise explanation (3-4 lines max, formatted as bullet points with '-' prefix) summarizing why these features increase churn risk, focusing on vehicle insurance factors (e.g., missed payments, claims history). "
            f"Use plain language for non-technical users, avoid mentioning SHAP values, and include the actual feature values in your explanation.\n\n"
            f"Features:\n{top_features_text}\n\n"
            f"Customer's Actual Values:\n{relevant_data}"
        )

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        params = {"key": GEMINI_API_KEY}

        response = requests.post(url, headers=headers, json=payload, params=params)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    except Exception as e:
        print(f"[AI CHURN DRIVERS ERROR] {e}")
        return "Unable to generate churn drivers explanation."

def explain_retention_strategy(customer_facts_text, raw_data):
    """
    Use Gemini API to generate an explanation for retention strategies in vehicle insurance.
    """
    try:
        # Extract top features from customer facts (e.g., Missed Payments, Customer Complaints)
        top_features = ['Missed_Payments', 'Customer_Complaints', 'Complaint_Resolution_Time']
        relevant_data = "\n".join(
            [f"- {key}: {raw_data[key]}" for key in raw_data if key in top_features]
        )

        prompt = (
            f"You are a churn analysis assistant for a vehicle insurance platform. Below are customer details and the customer's actual values for key features contributing to churn. "
            f"Provide a concise set of retention strategies (3-4 lines max, formatted as points with '-' prefix) to prevent churn, focusing on vehicle insurance factors (e.g., addressing missed payments, tailoring for vehicle type, etc). "
            f"Use plain language for non-technical users and include the actual feature values in your strategies where relevant.\n\n"
            f"Customer Facts:\n{customer_facts_text}\n\n"
            f"Customer's Actual Values for Key Features:\n{relevant_data}"
        )

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        params = {"key": GEMINI_API_KEY}

        response = requests.post(url, headers=headers, json=payload, params=params)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    except Exception as e:
        print(f"[AI RETENTION STRATEGY ERROR] {e}")
        return "Unable to generate retention strategy explanation."