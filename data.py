import requests
import time
import json
from sign import sign_request

url = "https://api.valr.com/v1/marketdata/BTCZAR/tradehistory?skip=0&limit=100"

headers = {
    "X-VALR-API-KEY": "",
    "X-VALR-SIGNATURE": "",
    "X-VALR-TIMESTAMP": ""
}

secret = ""
api = ""

try:
    with open("../Keys.json") as f:
        json_data = json.load(f)
        
        headers["X-VALR-API-KEY"] = json_data["apikey"]
        api = json_data["apikey"]
        secret = json_data["secret"]
        
        timestamp = int(time.time()*1000)
        
        sig  = sign_request(json_data["secret"], timestamp, "GET", "/v1/marketdata/BTCZAR/tradehistory?skip=0&limit=100")
        
        # print(sig)
        
        headers["X-VALR-SIGNATURE"] = sig
        headers["X-VALR-TIMESTAMP"] = str(timestamp)
except Exception as e:
    print(f"Failed to read json_data: {e}")
    # Exit if the keys file can't be read
    exit()
    

response = requests.get(url, headers=headers)



if response.status_code == 200:
    print(response.json())  # Print the JSON response
else:
    print(f"Failed to retrieve data: {response.status_code}")
    print(response.json())



flag = False
before = response.json()[-1]["id"]
limit = 100

while flag == False:
    url = f"https://api.valr.com/v1/marketdata/BTCZAR/tradehistory?limit={limit}&beforeId={before}"
    
    headers["X-VALR-API-KEY"] = json_data["apikey"]
    api = json_data["apikey"]
    secret = json_data["secret"]
    
    timestamp = int(time.time()*1000)
    
    sig  = sign_request(json_data["secret"], timestamp, "GET", f"/v1/marketdata/BTCZAR/tradehistory?limit={limit}&beforeId={before}")
    
    # print(sig)
    
    headers["X-VALR-SIGNATURE"] = sig
    headers["X-VALR-TIMESTAMP"] = str(timestamp)
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print(response.json())  # Print the JSON response
        before = response.json()[-1]["id"]
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        print(response.json())
        flad = True