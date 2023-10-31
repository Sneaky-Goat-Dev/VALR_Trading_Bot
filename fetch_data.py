import requests
import time
import json
from decouple import config
from sign import sign_request
import mysql.connector
from datetime import datetime
from dateutil import parser
import logging
import cProfile


# Initialize logging
logging.basicConfig(filename='api_calls.log', level=logging.ERROR)

# API rate limit variables
api_calls_per_minute = 0
api_calls_per_second = 0
minute_start_time = time.time()

# Function to reset rate limits
def reset_rate_limits():
    global api_calls_per_minute
    global api_calls_per_second
    global minute_start_time
    api_calls_per_minute = 0
    api_calls_per_second = 0
    minute_start_time = time.time()

# Function to handle API rate limits and perform the request
def rate_limited_get(url, headers):
    global api_calls_per_minute
    global api_calls_per_second

    # Calculate remaining time in current minute
    elapsed_time = time.time() - minute_start_time
    remaining_time = max(60 - elapsed_time, 0)  # Ensure the remaining time is never negative

    if api_calls_per_minute >= 1000 or api_calls_per_second >= 230:
        logging.warning("Rate limit exceeded. Waiting...")
        time.sleep(remaining_time)
        reset_rate_limits()

    response = requests.get(url, headers=headers)

    if response.status_code == 429:
        logging.error("Rate limit exceeded. Waiting...")
        time.sleep(remaining_time)
        reset_rate_limits()
        return rate_limited_get(url, headers)

    api_calls_per_minute += 1
    api_calls_per_second += 1

    return response

# Function to get the id of the trade with the most historic date from the database
def get_oldest_id():
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM valr_trading_bot.tradehistory ORDER BY tradedAt ASC LIMIT 1")
        oldest_id = cursor.fetchone()[0]
        return oldest_id
    except Exception as e:
        logging.error(f"An error occurred while fetching the oldest id: {e}")
        return None
    finally:
        cursor.close()

# Connect to the database
connection = mysql.connector.connect(
    user=config("DB_USER"),
    password=config("DB_PASSWORD"),
    host=config("DB_HOST"),
    port=config("DB_PORT"),
    database=config("DB_NAME")
)

# Function to insert trade data into the database
def insert_trade_data(trade_data_list):
    try:
        cursor = connection.cursor()
        insert_query = """
            INSERT IGNORE INTO valr_trading_bot.tradehistory (
                id, sequenceId, currencyPair, tradedAt, takerSide, price, quantity, quoteVolume
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        cursor.executemany(insert_query, trade_data_list)
        connection.commit()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        connection.rollback()
    finally:
        cursor.close()

# URL and headers setup
url = "https://api.valr.com/v1/marketdata/BTCZAR/tradehistory?skip=0&limit=100"
headers = {
    "X-VALR-API-KEY": "",
    "X-VALR-SIGNATURE": "",
    "X-VALR-TIMESTAMP": ""
}

# Read the Keys.json file
try:
    with open("../Keys.json") as f:
        json_data = json.load(f)
        headers["X-VALR-API-KEY"] = json_data["apikey"]
        timestamp = int(time.time()*1000)
        sig = sign_request(json_data["secret"], timestamp, "GET", "/v1/marketdata/BTCZAR/tradehistory?skip=0&limit=100")
        headers["X-VALR-SIGNATURE"] = sig
        headers["X-VALR-TIMESTAMP"] = str(timestamp)
except Exception as e:
    logging.error(f"Failed to read json_data: {e}")
    exit()

# Initialize 'before' with the oldest id from the database
oldest_id = get_oldest_id()
if oldest_id is not None:
    before = oldest_id
else:
    # Fall back to the oldest id from the initial API call
    response = rate_limited_get(url, headers)
    if response.status_code == 200:
        before = response.json()[-1]["id"]  # Assuming the API returns sorted data
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        print(response.json())
        exit()  # Exit the script if the initial call fails

records = []  # Variable to hold the records
batch_size = 5000  # Changeable variable to set the number of records to collect before inserting

flag = False
limit = 100

while flag == False:
    url = f"https://api.valr.com/v1/marketdata/BTCZAR/tradehistory?limit={limit}&beforeId={before}"
    
    headers["X-VALR-API-KEY"] = json_data["apikey"]
    api = json_data["apikey"]
    secret = json_data["secret"]
    
    timestamp = int(time.time()*1000)
    
    sig = sign_request(json_data["secret"], timestamp, "GET", f"/v1/marketdata/BTCZAR/tradehistory?limit={limit}&beforeId={before}")
    
    headers["X-VALR-SIGNATURE"] = sig
    headers["X-VALR-TIMESTAMP"] = str(timestamp)
    
    response = rate_limited_get(url, headers)  # Changed this line
    
    if response.status_code == 200:
        records.extend(response.json())
        print(records[-1])
        before = response.json()[-1]["id"]
        
        if len(records) >= batch_size:
            records_tuples = [(rec['id'], rec['sequenceId'], rec['currencyPair'], parser.parse(rec['tradedAt']).strftime('%Y-%m-%d %H:%M:%S'), rec['takerSide'], rec['price'], rec['quantity'], rec['quoteVolume']) for rec in records]
            insert_trade_data(records_tuples)
            records = []
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        print(response.json())
        flag = True
