import requests

GPT_OSS_URL = "http://10.180.148.183:8001/v1/sacs/afdsdfsdv"
GPT_OSS_TOKEN = "*****-######"

def ask_gpt_oss(messages):

    headers = {
        "Authorization": f"Bearer {GPT_OSS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": messages
    }

    response = requests.post(
        GPT_OSS_URL,
        json=payload,
        headers=headers
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
