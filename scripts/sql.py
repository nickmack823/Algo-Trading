import json
import logging
import os
import sqlite3
import time

import pandas as pd

from scripts import config

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)


class SQLHelper:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(
            f"{db_path}", check_same_thread=False
        )  # Use check_same_thread=False for SQLitedb_filename)
        self.cursor = self.conn.cursor()
        self.db_path = db_path

    def close_connection(self) -> None:
        self.conn.close()

    def safe_execute(
        self, cursor: sqlite3.Cursor, query: str, params=None, retries=3, delay=1
    ):
        for attempt in range(retries):
            try:
                if params:
                    return cursor.execute(query, params)
                else:
                    return cursor.execute(query)
            except sqlite3.OperationalError as e:
                if "disk I/O" in str(e) and attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise

    def create_table(
        self, table_name: str, columns: dict, foreign_keys: list = None
    ) -> None:
        """
        Create a new table in the database.

        Parameters:
        - table_name (str): Name of the table to create.
        - columns (dict): Dictionary of column names and data types.
        - foreign_keys (list): Optional list of foreign key definitions (as strings).
        """
        # Build the column definitions
        column_definitions = ", ".join(
            f"{column} {data_type}" for column, data_type in columns.items()
        )

        # Append foreign key constraints if provided
        if foreign_keys:
            fk_definitions = ", ".join(foreign_keys)
            column_definitions = column_definitions + ", " + fk_definitions

        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})"
        self.safe_execute(self.cursor, query)

    def get_database_tables(self) -> list:
        """
        Get the list of tables in the database.

        Returns:
        - list: A list of table names.
        """
        # Get all table names from the database
        existing_tables = [
            table[0]
            for table in self.safe_execute(
                self.cursor, "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]

        return existing_tables


class HistoricalDataSQLHelper(SQLHelper):
    def __init__(self, db_path: str):
        super().__init__(db_path)

    def get_database_tables(self) -> list[str]:
        """Get the list of tables in the database, ordered by timeframes from smallest to largest.

        Returns:
            list[str]: A list of table names.
        """
        tables = super().get_database_tables()

        # Define desired order as strings like '5_minute', '15_minute', etc.
        desired_order = [
            f"{multiplier}_{timespan}"
            for multiplier, timespan in config.TIMESPAN_MULTIPLIER_PAIRS
        ]

        # Sort existing tables based on the desired order
        ordered_tables = sorted(
            tables,
            key=lambda t: (
                desired_order.index(t) if t in desired_order else float("inf")
            ),  # unknown tables go last
        )

        return ordered_tables

    def insert_historical_data(self, data: pd.DataFrame, table: str) -> None:
        """
        Stores DataFrame into SQLite, dynamically handling new columns,
        without indexing Timestamp, and overwriting rows with matching timestamps.
        """
        # logging.info(f"Storing data into '{table}'...")

        # Ensure Timestamp is in datetime format
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], unit="ms")

        # Check if table exists
        self.safe_execute(
            self.cursor,
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'",
        )
        table_exists = self.cursor.fetchone() is not None

        if not table_exists:
            # Table doesn't exist yet, simply store data
            data.to_sql(table, self.conn, index=False)
            # logging.info(f"Created new table '{table}' and inserted data.")
        else:
            # Dynamically add missing columns
            self.safe_execute(self.cursor, f"PRAGMA table_info('{table}')")
            existing_columns = {col[1] for col in self.cursor.fetchall()}
            new_columns = set(data.columns) - existing_columns

            if new_columns:
                for column in new_columns:
                    dtype = (
                        "REAL"
                        if pd.api.types.is_numeric_dtype(data[column])
                        else "TEXT"
                    )
                    alter_sql = f'ALTER TABLE "{table}" ADD COLUMN "{column}" {dtype};'
                    self.safe_execute(self.cursor, alter_sql)
                # logging.info(f"Added columns '{new_columns}' to '{table}'.")

            # Delete existing rows with timestamps matching incoming data
            timestamps = data["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
            # SQLite has a limit of 999 variables per query, handle in chunks
            chunk_size = 900
            for i in range(0, len(timestamps), chunk_size):
                chunk = timestamps[i : i + chunk_size]
                placeholders = ",".join(["?"] * len(chunk))
                delete_query = (
                    f'DELETE FROM "{table}" WHERE Timestamp IN ({placeholders});'
                )
                self.safe_execute(self.cursor, delete_query, chunk)

            # Append new rows
            data.to_sql(table, self.conn, index=False, if_exists="append")
            # logging.info(f"Updated table '{table}' with new data.")

        # Commit changes
        self.conn.commit()
        # logging.info(f"Successfully stored data into '{table}'.")

    def get_historical_data(
        self,
        table: str,
        from_date: str = None,
        to_date: str = None,
        columns: list = None,
    ) -> pd.DataFrame:
        """
        Get historical data from a SQLite database.

        Parameters:
        - table: str, the table name
        - from_date: str, start date in 'YYYY-MM-DD' format
        - to_date: str, end date in 'YYYY-MM-DD' format
        - columns: list, the columns to select

        Returns:
        - DataFrame containing the historical data
        """
        selection = "*"
        if columns:
            selection = f"{', '.join(columns)}"

        query = f"SELECT {selection} FROM '{table}'"
        # Filter by from or to date or both
        if from_date and to_date:
            query += f" WHERE Timestamp BETWEEN '{from_date}' AND '{to_date}'"

        # Read the data from the database
        try:
            data = pd.read_sql(query, con=self.conn)
        except Exception as e:
            if "no such table" not in e.args[0]:
                logging.error(f"Error reading data from {self.db_path}: {e}")
            return None

        # logging.info(f"Data loaded from {self.db_filename} (Table: {table})")

        return data

    def get_all_data_files() -> list[str]:
        return [f for f in os.listdir(config.DATA_FOLDER) if f.endswith(".db")]


class BacktestSQLHelper(SQLHelper):
    def __init__(self):
        db_path = f"{config.BACKTESTING_FOLDER}/{config.BACKTESTING_DB_NAME}"
        super().__init__(db_path)

        existing_tables = self.get_database_tables()
        if "ForexPairs" not in existing_tables:
            self.create_table(
                "ForexPairs",
                {
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "symbol": "TEXT",
                    "base": "TEXT",
                    "quote": "TEXT",
                },
            )
            self.insert_forex_pairs(config.MAJOR_FOREX_PAIRS)
        if "StrategyConfigurations" not in existing_tables:
            self.create_table(
                "StrategyConfigurations",
                {
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "name": "TEXT",
                    "description": "TEXT",
                    "parameters": "TEXT",
                },
            )
        if "BacktestRuns" not in existing_tables:
            self.create_table(
                "BacktestRuns",
                {
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "Pair_ID": "INTEGER",
                    "Strategy_ID": "INTEGER",
                    "Timeframe": "TEXT",
                    "Data_Start_Date": "TEXT",
                    "Data_End_Date": "TEXT",
                    "Total_Trades": "INTEGER",
                    "Winning_Trades": "INTEGER",
                    "Gross_Profit": "REAL",
                    "Gross_Loss": "REAL",
                    "Net_Profit": "REAL",
                    "Total_Return_Pct": "REAL",
                    "Win_Rate": "REAL",
                    "Profit_Factor": "REAL",
                    "Max_Drawdown": "REAL",
                    "Average_Trade_Duration_Minutes": "REAL",
                    "Initial_Balance": "REAL",
                    "Final_Balance": "REAL",
                    "Sharpe_Ratio": "REAL",
                    "Sortino_Ratio": "REAL",
                    "Calmar_Ratio": "REAL",
                    "Recovery_Factor": "REAL",
                    "Win_Loss_Ratio": "REAL",
                    "Trade_Expectancy_Pct": "REAL",
                    "Expectancy_Per_Day_Pct": "REAL",
                    "Trade_Return_Std": "REAL",
                    "Trades_Per_Day": "REAL",
                    "Max_Consecutive_Wins": "INTEGER",
                    "Max_Consecutive_Losses": "INTEGER",
                    "Max_Pct_Margin_Required": "REAL",
                },
                foreign_keys=[
                    "FOREIGN KEY (Pair_ID) REFERENCES ForexPairs(id)",
                    "FOREIGN KEY (Strategy_ID) REFERENCES StrategyConfigurations(id)",
                ],
            )
        if "Trades" not in existing_tables:
            self.create_table(
                "Trades",
                {
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "Backtest_ID": "INTEGER",
                    "Timestamp": "TIMESTAMP",
                    "Exit_Timestamp": "TIMESTAMP",
                    "Direction": "INTEGER",
                    "Entry_Price": "REAL",
                    "Units": "REAL",
                    "Exit_Price": "REAL",
                    "PnL": "REAL",
                    "Net_Pips": "REAL",
                    "Commission": "REAL",
                    "Return_Pct": "REAL",
                    "Duration_Minutes": "REAL",
                    "Starting_Balance": "REAL",
                    "End_Balance": "REAL",
                    "Leverage": "REAL",
                    "Margin_Required": "REAL",
                    "Margin_Required_Pct": "REAL",
                    "Pip_Value": "REAL",
                    "Plan_Source": "TEXT",
                    "Position_Events": "TEXT",
                },
                foreign_keys=["FOREIGN KEY (Backtest_ID) REFERENCES BacktestRuns(id)"],
            )

    def insert_forex_pairs(self, pairs: list[str]) -> None:
        """Insert symbol, base, and quote into ForexPairs

        Args:
            pairs (list[str]): List of forex pairs in format 'EUR/USD'
        """
        for pair in pairs:
            base, quote = pair.split("/")
            self.safe_execute(
                self.cursor,
                "INSERT INTO ForexPairs (symbol, base, quote) VALUES (?, ?, ?)",
                (pair, base, quote),
            )
        self.conn.commit()

    def get_forex_pair_id(self, pair: str) -> int:
        """
        Retrieve the ID of a forex pair from the ForexPairs table.

        Args:
            pair (str): Forex pair symbol in the format 'BASE/QUOTE' (e.g., 'EUR/USD').

        Returns:
            int: The ID of the forex pair from the database.
        """
        self.safe_execute(
            self.cursor, "SELECT id FROM ForexPairs WHERE symbol = ?", (pair,)
        )

        return self.cursor.fetchone()[0]

    def upsert_strategy_configuration(
        self, name: str, description: str, parameters: dict
    ) -> int:
        # Convert the parameter settings to a JSON string (or handle differently if needed)
        """
        Upsert a strategy configuration into the database.

        If the configuration already exists, this function returns its primary key.
        If it doesn't exist, this function inserts it into the database and returns the primary key of the new row.

        Args:
            name (str): Name of the strategy configuration
            description (str): Description of the strategy configuration
            parameters (dict): Dictionary of parameter names and values

        Returns:
            int: Primary key of the strategy configuration
        """
        params_json = json.dumps(parameters)

        # Check if the configuration already exists
        id = self.select_strategy_configuration(name, description, parameters)

        # If it doesn't exist, insert it
        if id is None:
            query = "INSERT INTO StrategyConfigurations (name, description, parameters) VALUES (?, ?, ?)"

            # Execute the INSERT statement
            self.safe_execute(self.cursor, query, (name, description, params_json))
            self.conn.commit()

            id = self.cursor.lastrowid

        return id

    def select_strategy_configuration(
        self, name: str, description: str, parameters: dict
    ) -> int:
        params_json = json.dumps(parameters)

        # Check if the configuration already exists
        query = "SELECT id FROM StrategyConfigurations WHERE name = ? AND description = ? AND parameters = ?"
        self.safe_execute(self.cursor, query, (name, description, params_json))
        existing_config = self.cursor.fetchone()

        # If it exists, return its primary key
        id = None
        if existing_config is not None:
            id = existing_config[0]

        return id

    def insert_backtest_run(
        self, pair_id: int, strategy_config_id: int, metrics: pd.DataFrame
    ) -> int:
        """
        Inserts a new backtest run record into the BacktestRuns table and returns the run_id.

        Args:
            pair_id (int): The ID of the forex pair from the ForexPairs table.
            strategy_config_id (int): The ID of the strategy configuration from the StrategyConfigurations table.
            metrics (pd.DataFrame): A one-row DataFrame containing aggregated run metrics.

        Returns:
            int: The run_id of the newly inserted backtest run.
        """
        # Get the list of column names from the metrics DataFrame.
        columns = metrics.columns.tolist()

        # Create the same number of placeholders as there are columns.
        placeholders = ", ".join(["?"] * len(columns))

        # Build the INSERT query.
        query = (
            f"INSERT INTO BacktestRuns (Pair_ID, Strategy_ID, {', '.join(columns)}) "
            f"VALUES (?, ?, {placeholders})"
        )

        # Extract the single row of metrics and convert each value to a native Python type.
        row_values = tuple(
            v.item() if hasattr(v, "item") else v for v in metrics.iloc[0]
        )

        # Execute the INSERT statement with the flattened tuple of values.
        self.safe_execute(
            self.cursor, query, (pair_id, strategy_config_id, *row_values)
        )
        self.conn.commit()

        # Return the last inserted run_id.
        return self.cursor.lastrowid

    def backtest_run_exists(
        self, pair_id: int, strategy_config_id: int, timeframe: str
    ) -> bool:
        """Checks if a backtest run exists in the database

        Args:
            pair_id (int): The ID of the forex pair from the ForexPairs table.
            strategy_config_id (int): The ID of the strategy configuration from the StrategyConfigurations table.
            timeframe (str): The timeframe of the backtest

        Returns:
            bool: A boolean indicating whether the backtest run exists in the database
        """
        self.safe_execute(
            self.cursor,
            "SELECT id FROM BacktestRuns WHERE Pair_ID = ? AND Strategy_ID = ? AND Timeframe = ?",
            (pair_id, strategy_config_id, timeframe),
        )

        return self.cursor.fetchone() is not None

    def insert_trades(self, trades: pd.DataFrame) -> None:
        """
        Inserts multiple trade records into the Trades table in the database.

        Args:
            trades (pd.DataFrame): DataFrame containing trade data with column names matching the Trades table.
        Returns:
            None
        """
        # Convert 'Timestamp' and 'Exit_Timestamp' columns to string (so SQLite can convert to TIMESTAMP)
        trades["Timestamp"] = pd.to_datetime(trades["Timestamp"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        trades["Exit_Timestamp"] = pd.to_datetime(trades["Exit_Timestamp"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Prepare the SQL query based on DataFrame columns
        columns = trades.columns.tolist()
        placeholders = ", ".join(["?"] * len(columns))
        query = f"INSERT INTO Trades ({', '.join(columns)}) VALUES ({placeholders})"

        # Convert DataFrame rows into a list of tuples
        data_tuples = [tuple(row) for row in trades.to_numpy()]

        # Execute batch insertion
        self.cursor.executemany(query, data_tuples)
        self.conn.commit()
