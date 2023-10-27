import requests
import time
import json
from decouple import config
from sign import sign_request
import csv
import io
import mysql.connector
from datetime import datetime
from dateutil import parser

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
        print("before")

        # Create a cursor to execute SQL commands
        cursor = connection.cursor()

        # SQL command to insert a record into the database
        insert_query = """
            INSERT IGNORE INTO valr_trading_bot.tradehistory (
                id, sequenceId, currencyPair, tradedAt, takerSide, price, quantity, quoteVolume
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """

        # Execute the SQL command to insert all records
        cursor.executemany(insert_query, trade_data_list)

        # Commit the transaction
        connection.commit()

        print("after")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Rollback the transaction on error
        connection.rollback()
    finally:
        # Ensure the cursor is closed when done
        cursor.close()


# URL and headers setup
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
        
        sig = sign_request(json_data["secret"], timestamp, "GET", "/v1/marketdata/BTCZAR/tradehistory?skip=0&limit=100")
        
        headers["X-VALR-SIGNATURE"] = sig
        headers["X-VALR-TIMESTAMP"] = str(timestamp)
except Exception as e:
    print(f"Failed to read json_data: {e}")
    exit()

response = requests.get(url, headers=headers)

records = []  # Variable to hold the records
batch_size = 1000  # Changeable variable to set the number of records to collect before inserting

if response.status_code == 200:
    print(response.json())  # Print the JSON response
    records.extend(response.json())
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
    
    sig = sign_request(json_data["secret"], timestamp, "GET", f"/v1/marketdata/BTCZAR/tradehistory?limit={limit}&beforeId={before}")
    
    headers["X-VALR-SIGNATURE"] = sig
    headers["X-VALR-TIMESTAMP"] = str(timestamp)
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # print(response.json())  # Print the JSON response
        records.extend(response.json())
        print(records[-1])
        before = response.json()[-1]["id"]
        
        if len(records) >= batch_size:
            records_tuples = [(rec['id'], rec['sequenceId'], rec['currencyPair'], parser.parse(rec['tradedAt']).strftime('%Y-%m-%d %H:%M:%S'), rec['takerSide'], rec['price'], rec['quantity'], rec['quoteVolume']) for rec in records]
            insert_trade_data(records_tuples)  # Note the function name change here
            records = []  # Reset the records list
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        print(response.json())
        flag = True


# Close the database session
connection.close()