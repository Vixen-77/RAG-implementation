import requests

r = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "gemma3:1b",
        "prompt": "Dis bonjour en une phrase",
        "stream": False
    }
)

print(r.json()["response"])
