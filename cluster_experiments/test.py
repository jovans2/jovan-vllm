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

api_url1 = "http://localhost:" + str(8080) + "/epoch_update"
response = requests.post(api_url1, headers=headers, json=payload)

