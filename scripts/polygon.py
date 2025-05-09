import json
import logging
import multiprocessing
import sys
import time

import pandas as pd
import urllib3
from polygon import RESTClient
from tqdm import tqdm

import scripts.config as config
from scripts.sql import HistoricalDataSQLHelper, SQLHelper

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)

timespans = ["minute", "hour", "day"]

# Timeframes and how many days back to collect
timeframe_data_collection = {
    (5, "minute"): 365 * 3,
    (15, "minute"): 365 * 3,
    (30, "minute"): 365 * 3,
    (1, "hour"): 365 * 3,
    (2, "hour"): 365 * 3,
    (4, "hour"): 365 * 3,
    (1, "day"): 365 * 3,
}


def countdown_with_sleep(total_seconds):
    logging.info(f"Sleeping for {total_seconds} seconds...")
    for remaining in range(total_seconds, 0, -1):
        # Print the countdown on the same line
        sys.stdout.write(f"Seconds remaining: {remaining} \r")
        sys.stdout.flush()
        time.sleep(1)

    # Clear the line and print completion message
    logging.info("Done sleeping.")


class Polygon:
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key=api_key)

    def fetch_historical_data(
        self, ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical data from Polygon.io.

        Parameters:
        - ticker: str, the ticker symbol (e.g., 'C:EURUSD')
        - multiplier: int, the size of the time window
        - timespan: str, the size of the time window ('minute', 'hour', 'day', etc.)
        - from_date: str, start date in 'YYYY-MM-DD' format
        - to_date: str, end date in 'YYYY-MM-DD' format

        Returns:
        - DataFrame containing the historical data
        """
        # logging.info(f"Fetching historical data for {ticker} ({multiplier}-{timespan}) from {from_date} to {to_date}...")
        aggs_list = []

        def get_aggs():
            for agg in self.client.list_aggs(
                ticker, multiplier, timespan, from_=from_date, to=to_date, limit=50000
            ):
                aggs_list.append(agg)

        succeeded = False
        while not succeeded:
            try:
                get_aggs()
                succeeded = True
            # Retry after pause if we hit the 5 calls/minute free tier rate limit
            except urllib3.exceptions.MaxRetryError:
                logging.error(f"Hit free tier rate limit - Retrying in 1 minute...")
                countdown_with_sleep(60)

        data = pd.DataFrame(aggs_list)

        # Sort the data by timestamp and make timestamp the first column
        data = data.sort_values("timestamp")
        data = data[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "transactions",
            ]
        ]

        # Capitalize column names
        data.columns = [col.capitalize() for col in data.columns]
        data = data.rename(columns={"Vwap": "VWAP"})

        # Round columns to appropriate decimals
        num_decimals = 3 if "JPY" in ticker else 5
        data["Open"] = data["Open"].round(num_decimals)
        data["High"] = data["High"].round(num_decimals)
        data["Low"] = data["Low"].round(num_decimals)
        data["Close"] = data["Close"].round(num_decimals)

        return data

    def filter_collected_data_dates(
        self, existing_data: pd.DataFrame, from_date: str, to_date: str
    ) -> dict:
        """
        Filter data collection range based on the initial from date.

        Parameters:
        - data: DataFrame, the data to filter
        - from_date: str, the initial from date in 'YYYY-MM-DD' format
        - to_date: str, the initial to date in 'YYYY-MM-DD' format

        Returns:
        - dict: A dictionary containing the filtered from_date and to_date to collect from
        - None: If there is no data left to collect within the range
        """
        filtered_dates = {"from_date": from_date, "to_date": to_date}

        # Date ranges are exclusive
        day_before_to_date = (pd.to_datetime(to_date) - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

        filter_from = existing_data["Timestamp"].str.contains(from_date).any()
        filter_to = existing_data["Timestamp"].str.contains(day_before_to_date).any()

        # Convert Timestamp to datetime
        existing_data["Timestamp"] = pd.to_datetime(existing_data["Timestamp"])

        # Check if initial date is in any timestamp, if so, get the most recent date past that point
        if filter_from:
            # Get most recent date past initial from date
            most_recent_date = existing_data[existing_data["Timestamp"] > from_date][
                "Timestamp"
            ].max()
            most_recent_date = most_recent_date.strftime("%Y-%m-%d")

            # Update filtered dates
            filtered_dates["from_date"] = most_recent_date

        # If from_date and to_date are the same, return None to indicate no data to collect
        if pd.to_datetime(filtered_dates["from_date"]) == pd.to_datetime(
            to_date
        ) - pd.Timedelta(days=1):
            return None

        # Check if to_date is in any timestamp, if so, get the furthest back date before that point
        if filter_to:
            # Get most recent date past initial from date
            least_recent_date = existing_data[existing_data["Timestamp"] < to_date][
                "Timestamp"
            ].min()
            least_recent_date = least_recent_date.strftime("%Y-%m-%d")

            # Update filtered dates
            filtered_dates["to_date"] = least_recent_date

        # logging.info(f"Filtered data collection dates: {filtered_dates['from_date']} to {filtered_dates['to_date']}.")

        return filtered_dates

    def collect_data(self, pair: str) -> None:
        # logging.info(f"Collecting data for {pair}...")
        ticker = f"C:{pair.replace('/', '')}"
        sql_helper = HistoricalDataSQLHelper(f'data/{ticker.split(":")[1]}.db')

        for timeframe, days_back in timeframe_data_collection.items():

            multiplier, timespan = timeframe
            from_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime(
                "%Y-%m-%d"
            )
            to_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            # to_date = "2025-03-24"

            db_table = f"{multiplier}_{timespan}"

            # Filter data collection range based on the initial from date
            existing_data = sql_helper.get_historical_data(
                db_table, from_date=from_date, to_date=to_date
            )

            if existing_data is not None:
                filtered_dates = self.filter_collected_data_dates(
                    existing_data, from_date, to_date
                )

                # If there is no data left to collect within the range, skip
                if filtered_dates is None:
                    # logging.info(f"Data collection for {ticker} ({multiplier}-{timespan}) already completed.")
                    continue

                # Update from_date and to_date
                from_date, to_date = (
                    filtered_dates["from_date"],
                    filtered_dates["to_date"],
                )

            # Fetch data from Polygon API
            collected_data = self.fetch_historical_data(
                ticker, multiplier, timespan, from_date, to_date
            )

            # Store data in database
            sql_helper.insert_historical_data(collected_data, db_table)

        # Close connection to this pair's database
        sql_helper.close_connection()


def collect_pair_data(pair: str) -> None:
    polygon = Polygon(api_key="fVHL9IoeV1qyxZ2GV_knfnQJVZaqvuqP")
    polygon.collect_data(pair)

    return pair


def collect_all_data() -> None:
    pairs_to_collect = [pair for pair in config.MAJOR_FOREX_PAIRS]
    for pair in pairs_to_collect:
        base, quote = pair.split("/")
        # If USD not in pair, also collect supplemental "QUOTE/USD" pair data for
        # calculating pip values in future backtesting
        if "USD" not in [base, quote] and f"{quote}/USD" not in pairs_to_collect:
            pairs_to_collect.append(f"{quote}/USD")

    processes = round(multiprocessing.cpu_count() / 4)
    with multiprocessing.Pool(processes=processes) as pool:
        with tqdm(total=len(pairs_to_collect), desc="Collecting data") as pbar:
            for result in pool.imap_unordered(collect_pair_data, pairs_to_collect):
                pbar.update()
                logging.info(f"Collected data for {result}.")

    logging.info("Data collection complete.")
