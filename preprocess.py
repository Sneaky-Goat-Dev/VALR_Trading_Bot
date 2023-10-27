import time
import json
from decouple import config
import mysql.connector
from datetime import datetime, timedelta
from dateutil import parser
import pandas as pd
import numpy as np
import logging


logging.basicConfig(filename='preprocessing.log', level=logging.INFO)


# Connect to the database
connection = mysql.connector.connect(
    user=config("DB_USER"),
    password=config("DB_PASSWORD"),
    host=config("DB_HOST"),
    port=config("DB_PORT"),
    database=config("DB_NAME")
)

cursor = connection.cursor()


# Initialize previous values for certain features
previous_values = {
    'simple_moving_average': None,
    'exponential_moving_average': None,
    'rsi': None,
    'macd': None,
    'market_trend': None,
    'signal_line': None
}


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window).mean()
    long_ema = data.ewm(span=long_window).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]


def calculate_features(data):
    df = pd.DataFrame(data, columns=[
        'id', 'price', 'quantity', 'currencyPair',
        'tradedAt', 'takerSide', 'sequenceId', 'quoteVolume'
    ])
    
    # Ensure the data is sorted by tradedAt
    df = df.sort_values(by='tradedAt')
    
    # Fill missing values
    df.ffill(inplace=True)  # Forward fill
    df.bfill(inplace=True)  # Backward fill, in case the first values are missing

    
    macd, signal_line = calculate_macd(df['price'])
    
    features = {
        'opening_price': df['price'].iloc[0],
        'closing_price': df['price'].iloc[-1],
        'high_price': df['price'].max(),
        'low_price': df['price'].min(),
        'price_change': df['price'].iloc[-1] - df['price'].iloc[0],
        'total_volume': df['quantity'].sum(),
        'volume_weighted_avg_price': (df['price'] * df['quantity']).sum() / df['quantity'].sum(),
        'number_of_trades': len(df),
        'buy_volume': df[df['takerSide'] == 'buy']['quantity'].sum(),
        'sell_volume': df[df['takerSide'] == 'sell']['quantity'].sum(),
        'simple_moving_average': df['price'].rolling(window=5).mean().iloc[-1],  # 5-period SMA
        'exponential_moving_average': df['price'].ewm(span=5).mean().iloc[-1],  # 5-period EMA
        # The below calculations are simplistic and may need to be refined
        'rsi': calculate_rsi(df['price']),
        'macd': macd,
        'market_trend': 1 if df['price'].iloc[-1] > df['price'].iloc[0] else 0,  # Bullish if closing price > opening price
        'signal_line': signal_line 
    }
    
    return features


try:
    
    connection.start_transaction()

    # Define the time range for the past 3 months
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)  # Adjust as necessary



    # Loop through each hour of the past 3 months
    current_time = start_time
    while current_time < end_time:
        interval_start = current_time
        interval_end = current_time + timedelta(hours=1)
        
        # Query the raw data for this interval
        query = """
            SELECT * FROM valr_trading_bot.tradehistory
            WHERE tradedAt >= %s AND tradedAt < %s
        """
        cursor.execute(query, (interval_start, interval_end))
        raw_data = cursor.fetchall()
        
        if raw_data:  # Ensure there is data for this interval
            # Calculate features
            features = calculate_features(raw_data)
            
            # Check for NaN values in certain features and use previous values if needed
            for key in previous_values.keys():
                if pd.isnull(features[key]):
                    logging.warning(f'NaN value found for {key} at interval {interval_start} to {interval_end}, using previous value.')
                    features[key] = previous_values[key]  # Use the previous value if current value is NaN
                else:
                    previous_values[key] = features[key]  # Update the previous value if current value is non-NaN
            
            # Insert the preprocessed data into the preprocessed table
            insert_query = """
                INSERT INTO valr_trading_bot.tradehistory_preprocessed_1hour (
                    id,
                    interval_start,
                    interval_end,
                    opening_price,
                    closing_price,
                    high_price,
                    low_price,
                    price_change,
                    total_volume,
                    volume_weighted_avg_price,
                    number_of_trades,
                    buy_volume,
                    sell_volume,
                    simple_moving_average,
                    exponential_moving_average,
                    rsi,
                    macd,
                    market_trend,
                    signal_line
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON DUPLICATE KEY UPDATE
                    opening_price = VALUES(opening_price),
                    closing_price = VALUES(closing_price),
                    high_price = VALUES(high_price),
                    low_price = VALUES(low_price),
                    price_change = VALUES(price_change),
                    total_volume = VALUES(total_volume),
                    volume_weighted_avg_price = VALUES(volume_weighted_avg_price),
                    number_of_trades = VALUES(number_of_trades),
                    buy_volume = VALUES(buy_volume),
                    sell_volume = VALUES(sell_volume),
                    simple_moving_average = VALUES(simple_moving_average),
                    exponential_moving_average = VALUES(exponential_moving_average),
                    rsi = VALUES(rsi),
                    macd = VALUES(macd),
                    market_trend = VALUES(market_trend),
                    signal_line = VALUES(signal_line)
            """
            
            # # Example of checking for NaN values in the features dictionary
            # for key, value in features.items():
            #     if pd.isnull(value):
            #         logging.error(f'NaN value found for {key} at interval {interval_start} to {interval_end}')
            #         # You might want to assign a default value or skip the insertion
            #         features[key] = None  # Assuming your database allows NULL values in these columns
                    
                    
            # Example of logging the features just before insertion
            logging.info(f'Inserting features: {features}')
            
            
            cursor.execute(insert_query, (
                str(interval_start),  # Generate a unique ID for each interval
                interval_start,
                interval_end,
                features['opening_price'],
                features['closing_price'],
                features['high_price'],
                features['low_price'],
                features['price_change'],
                features['total_volume'],
                features['volume_weighted_avg_price'],
                features['number_of_trades'],
                features['buy_volume'],
                features['sell_volume'],
                features['simple_moving_average'],
                features['exponential_moving_average'],
                features['rsi'],
                features['macd'],
                features['market_trend'],
                features['signal_line']
            ))
            
            # Commit the transaction
            connection.commit()
        
        # Move to the next interval
        current_time = interval_end

    connection.commit()
except Exception as e:
    logging.error(f"An error occurred: {e}")
    connection.rollback()
    print(f"An error occurred: {e}")
finally:
    if cursor: cursor.close()
    if connection: connection.close()