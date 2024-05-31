import requests


api_url = "http://localhost:" + str(8080) + "/generate"
headers = {"User-Agent": "Benchmark Client"}
payload = {
            "prompt": "Hello, world!",
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 100,
            "ignore_eos": True,
            "stream": False,
        }

api_url1 = "http://10.0.0.6:" + str(9082) + "/generate"
response = requests.post(api_url1, headers=headers, json=payload)
print(response.text)
