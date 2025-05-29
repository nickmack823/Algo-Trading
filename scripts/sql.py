import hashlib
import json
import logging
import os
import random
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd

from scripts import config

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)


class SQLHelper:
    def __init__(self, db_path: str, read_only: bool = False):
        if read_only:
            db_uri = f"file:{db_path}?mode=ro"
            self.conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
        else:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.safe_execute(self.conn.cursor(), "PRAGMA journal_mode=WAL;")
            self.safe_execute(self.conn.cursor(), "PRAGMA synchronous=NORMAL;")
        self.cursor = self.conn.cursor()
        self.db_path = db_path

    def close_connection(self) -> None:
        self.conn.close()

    def safe_execute(
        self, cursor: sqlite3.Cursor, query: str, params=None, retries=7, delay=1.5
    ) -> sqlite3.Cursor:
        for attempt in range(retries):
            try:
                if params:
                    return cursor.execute(query, params)
                return cursor.execute(query)
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                wait = random.uniform(delay, delay + 1.5)
                if "database is locked" in msg or "disk i/o" in msg:
                    logging.warning(
                        f"[SQLite] Locked (attempt {attempt+1}/{retries}) — retrying in {wait:.2f}s"
                    )
                    time.sleep(wait)
                else:
                    raise
        raise sqlite3.OperationalError(f"[FAILED] Query gave persistent lock: {query}")

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
    def __init__(self, db_path: str, read_only: bool = False):
        super().__init__(db_path, read_only)

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
            from_dt = datetime.strptime(from_date, "%Y-%m-%d")
            to_dt = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(
                days=1
            )  # Add 1 day to include end date
            query += f" WHERE Timestamp BETWEEN '{from_dt}' AND '{to_dt}'"

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


class IndicatorCacheSQLHelper(SQLHelper):
    def __init__(
        self, db_path: str = f"{config.BACKTESTING_FOLDER}/{config.BACKTESTING_DB_NAME}"
    ):
        super().__init__(db_path)

        self.cache_dir = f"{config.BACKTESTING_FOLDER}/indicator_caches"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.create_table(
            "IndicatorCache",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "cache_key": "TEXT UNIQUE",
                "forex_pair": "TEXT",
                "timeframe": "TEXT",
                "data_start_date": "TEXT",
                "data_end_date": "TEXT",
                "indicator_name": "TEXT",
                "parameter_hash": "TEXT",
                "filepath": "TEXT",
            },
        )

    def _generate_cache_key(
        self,
        indicator_name,
        parameters: dict,
        forex_pair: str,
        timeframe: str,
        data_start_date: str,
        data_end_date: str,
    ) -> tuple[str, str]:
        param_str = str(sorted(parameters.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        key = f"{indicator_name}:{param_hash}:{forex_pair}:{timeframe}:{data_start_date}:{data_end_date}"
        return key, param_hash

    def fetch(
        self,
        indicator_name: str,
        parameters: dict,
        forex_pair: str,
        timeframe: str,
        data_start_date: str,
        data_end_date: str,
    ) -> pd.DataFrame | None:
        key, _ = self._generate_cache_key(
            indicator_name,
            parameters,
            forex_pair,
            timeframe,
            data_start_date,
            data_end_date,
        )

        cursor = self.safe_execute(
            self.cursor,
            "SELECT filepath FROM IndicatorCache WHERE cache_key = ?",
            (key,),
        )

        result = cursor.fetchone()
        if result and os.path.exists(result[0]):
            return pd.read_feather(result[0])
        return None

    def store_in_feather(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        parameters: dict,
        forex_pair: str,
        timeframe: str,
        data_start_date: str,
        data_end_date: str,
    ) -> dict:
        key, param_hash = self._generate_cache_key(
            indicator_name,
            parameters,
            forex_pair,
            timeframe,
            data_start_date,
            data_end_date,
        )
        filename = f"{hashlib.md5(key.encode()).hexdigest()}.feather"
        filepath = os.path.join(self.cache_dir, filename)

        df.reset_index(drop=True).to_feather(filepath)

        return {
            "key": key,
            "forex_pair": forex_pair,
            "timeframe": timeframe,
            "data_start_date": data_start_date,
            "data_end_date": data_end_date,
            "indicator_name": indicator_name,
            "param_hash": param_hash,
            "filepath": filepath,
        }

    def insert_cache_items(self, cache_items: list[dict]):
        """
        Insert cache items into the IndicatorCache table.

        cache_items: A list of cache item dictionaries with keys: key, forex_pair, timeframe, data_start_date, data_end_date, indicator_name, param_hash, filepath
        """
        for item in cache_items:
            _ = self.safe_execute(
                self.cursor,
                """INSERT OR IGNORE INTO IndicatorCache (
                    cache_key, forex_pair, timeframe, data_start_date, data_end_date, indicator_name, parameter_hash, filepath
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["key"],
                    item["forex_pair"],
                    item["timeframe"],
                    item["data_start_date"],
                    item["data_end_date"],
                    item["indicator_name"],
                    item["param_hash"],
                    item["filepath"],
                ),
            )

        self.conn.commit()


class BacktestSQLHelper(SQLHelper):
    def __init__(self, read_only: bool = False):
        db_path = f"{config.BACKTESTING_FOLDER}/{config.BACKTESTING_DB_NAME}"
        super().__init__(db_path, read_only)

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
                    "Trading_Start_Date": "TEXT",
                    "Total_Trades": "INTEGER",
                    "Winning_Trades": "INTEGER",
                    "Gross_Profit": "REAL",
                    "Gross_Loss": "REAL",
                    "Net_Profit": "REAL",
                    "Total_Return_Pct": "REAL",
                    "Win_Rate": "REAL",
                    "Profit_Factor": "REAL",
                    "Max_Drawdown": "REAL",
                    "Max_Drawdown_Pct": "REAL",
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
                    "Max_Margin_Required_Pct": "REAL",
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
        if "CompositeScores" not in existing_tables:
            self.create_table(
                "CompositeScores",
                {
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "Backtest_ID": "INTEGER",
                    "Study_ID": "INTEGER DEFAULT NULL",
                    "Study_Name": "TEXT",
                    "Trial_ID": "INTEGER",
                    "Score": "REAL",
                    "Exploration_Space": "TEXT",
                    "Timestamp": "TIMESTAMP",
                },
                foreign_keys=[
                    "FOREIGN KEY (Backtest_ID) REFERENCES BacktestRuns(id)",
                    "FOREIGN KEY (Study_ID) REFERENCES StudyMetadata(id) ON DELETE SET NULL ON UPDATE CASCADE",
                ],
            )
        if "StudyMetadata" not in existing_tables:
            self.create_table(
                "StudyMetadata",
                {
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "Study_Name": "TEXT",
                    "Pair": "TEXT",
                    "Timeframe": "TEXT",
                    "Exploration_Space": "TEXT",
                    "Best_Score": "REAL",
                    "Best_Trial": "INTEGER",
                    "Time_to_Best": "REAL",
                    "Total_Time_Sec": "REAL",
                    "N_Trials": "INTEGER",
                    "N_Completed": "INTEGER",
                    "N_Pruned": "INTEGER",
                    "Avg_Score": "REAL",
                    "Std_Score": "REAL",
                    "Stop_Reason": "TEXT",
                },
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
        self,
        pair_id: int,
        strategy_config_id: int,
        timeframe: str,
        metrics: pd.DataFrame,
    ) -> int:
        """
        Inserts a new backtest run record into the BacktestRuns table and returns the run_id.

        Args:
            pair_id (int): The ID of the forex pair from the ForexPairs table.
            strategy_config_id (int): The ID of the strategy configuration from the StrategyConfigurations table.
            timeframe (str): The timeframe of the backtest run.
            metrics (pd.DataFrame): A one-row DataFrame containing aggregated run metrics.

        Returns:
            int: The run_id of the newly inserted backtest run.
        """
        assert (
            pair_id is not None
            and strategy_config_id is not None
            and timeframe is not None
            and metrics is not None
        )
        # Get the list of column names from the metrics DataFrame.
        columns = metrics.columns.tolist()

        # Create the same number of placeholders as there are columns.
        placeholders = ", ".join(["?"] * len(columns))

        # Build the INSERT query.
        query = (
            f"INSERT INTO BacktestRuns (Pair_ID, Strategy_ID, Timeframe, {', '.join(columns)}) "
            f"VALUES (?, ?, ?, {placeholders})"
        )

        # Extract the single row of metrics and convert each value to a native Python type.
        row_values = tuple(
            v.item() if hasattr(v, "item") else v for v in metrics.iloc[0]
        )

        # Execute the INSERT statement with the flattened tuple of values.
        self.safe_execute(
            self.cursor, query, (pair_id, strategy_config_id, timeframe, *row_values)
        )
        self.conn.commit()

        # Return the last inserted run_id.
        return self.cursor.lastrowid

    def select_backtest_run_by_config(
        self,
        pair_id: int,
        strategy_config_id: int,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """Checks if a backtest run exists in the database

        Args:
            pair_id (int): The ID of the forex pair from the ForexPairs table.
            strategy_config_id (int): The ID of the strategy configuration from the StrategyConfigurations table.
            timeframe (str): The timeframe of the backtest
            start_date (str): The start date of the backtest (YYYY-MM-DD)
            end_date (str): The end date of the backtest (YYYY-MM-DD)

        Returns:
            int: The run_id of the backtest run
        """
        self.safe_execute(
            self.cursor,
            "SELECT id FROM BacktestRuns WHERE Pair_ID = ? AND Strategy_ID = ? AND Timeframe = ? AND Data_Start_Date = ? AND Data_End_Date = ?",
            (pair_id, strategy_config_id, timeframe, start_date, end_date),
        )

        result = self.cursor.fetchone()

        if result is None:
            return None

        run_id = result[0]

        return run_id

    def select_backtest_run_metrics_by_id(self, run_id: int) -> pd.DataFrame:
        self.safe_execute(
            self.cursor,
            "SELECT * FROM BacktestRuns WHERE id = ?",
            (run_id,),
        )

        result = self.cursor.fetchone()

        if result is None:
            return None

        # Get column names from cursor
        columns = [desc[0] for desc in self.cursor.description]

        # Build DataFrame with proper column alignment
        return pd.DataFrame([result], columns=columns)

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

    def insert_study_metadata(self, study_metadata: pd.DataFrame) -> int:
        """
        Inserts multiple study metadata records into the StudyMetadata table in the database.

        Args:
            study_metadata (pd.DataFrame): DataFrame containing study metadata data with column names matching the StudyMetadata table.
        Returns:
            int: The last inserted row id
        """
        # Check if study metadata already exists
        self.safe_execute(
            self.cursor,
            "SELECT id FROM StudyMetadata WHERE Study_Name = ?",
            (study_metadata["Study_Name"][0],),
        )

        result = self.cursor.fetchone()

        if result is not None:
            logging.info(f"Study metadata already exists: {result[0]}")
            return result[0]

        # Prepare the SQL query based on DataFrame columns
        columns = study_metadata.columns.tolist()
        placeholders = ", ".join(["?"] * len(columns))
        query = (
            f"INSERT INTO StudyMetadata ({', '.join(columns)}) VALUES ({placeholders})"
        )

        # Convert DataFrame rows into a list of tuples
        data_tuples = [tuple(row) for row in study_metadata.to_numpy()]

        # Execute batch insertion
        self.cursor.executemany(query, data_tuples)
        self.conn.commit()

        study_name = study_metadata["Study_Name"].iloc[0]

        logging.info(f"Inserted study metadata: {study_name}")

    def select_study_id(self, study_name: str) -> int:
        self.safe_execute(
            self.cursor,
            "SELECT id FROM StudyMetadata WHERE Study_Name = ?",
            (study_name,),
        )

        result = self.cursor.fetchone()

        if result is None:
            return None

        study_id = result[0]

        return study_id

    def select_study_metadata(self, study_name: str) -> dict | None:
        self.safe_execute(
            self.cursor,
            "SELECT * FROM StudyMetadata WHERE Study_Name = ?",
            (study_name,),
        )

        result = self.cursor.fetchone()

        if result is None:
            return None

        # Get column names from cursor
        columns = [desc[0] for desc in self.cursor.description]

        # Build dictionary with proper column alignment
        return dict(zip(columns, result))

    def insert_composite_score(
        self,
        backtest_id: int,
        study_name: str,
        study_id: int,
        trial_id: int,
        score: float,
        exploration_space: str,
        timestamp: str,
    ):
        self.safe_execute(
            self.cursor,
            "INSERT INTO CompositeScores (Backtest_ID, Study_Name, Study_ID, Trial_ID, Score, Exploration_Space, Timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                backtest_id,
                study_name,
                study_id,
                trial_id,
                score,
                exploration_space,
                timestamp,
            ),
        )
        self.conn.commit()

    def backfill_composite_scores(self, study_id: int) -> None:
        """Backfills the Study_ID column in the CompositeScores table by joining with the StudyMetadata table."""
        study_name = self.safe_execute(
            self.cursor,
            "SELECT Study_Name FROM StudyMetadata WHERE id = ?",
            (study_id,),
        ).fetchone()[0]
        self.safe_execute(
            self.cursor,
            f"""
            UPDATE CompositeScores
            SET Study_ID = {study_id}
            WHERE Study_ID IS NULL AND Study_Name = '{study_name}'
            """,
        )

        rows_updated = self.cursor.rowcount

        self.conn.commit()

        logging.info(f"[{study_name}] Backfilled {rows_updated} composite scores.")

        return rows_updated

    def select_composite_score(self, backtest_id: int) -> float | None:
        """Selects a composite score from the CompositeScores table

        Args:
            backtest_id (int): The ID of the backtest run

        Returns:
            float: The composite score of the backtest run
        """
        query = "SELECT Score FROM CompositeScores WHERE Backtest_ID = ?"
        self.safe_execute(self.cursor, query, (backtest_id,))

        result = self.cursor.fetchone()

        if result is None:
            return None

        score = result[0]

        return score

    def export_composite_results_with_metrics(
        self, output_path: str = "composite_results.csv"
    ) -> None:
        """
        Exports all composite scores along with corresponding backtest run metrics to a CSV file.

        Args:
            output_path (str): Path to save the CSV file.
        """
        # Get all composite scores
        self.safe_execute(self.cursor, "SELECT * FROM CompositeScores")
        composite_rows = self.cursor.fetchall()
        if not composite_rows:
            logging.warning("No composite scores found to export.")
            return

        composite_cols = [desc[0] for desc in self.cursor.description]
        composite_df = pd.DataFrame(composite_rows, columns=composite_cols)

        # Get all corresponding backtest runs
        backtest_ids = tuple(composite_df["Backtest_ID"].unique())
        id_list = (
            f"({','.join(str(i) for i in backtest_ids)})"
            if len(backtest_ids) > 1
            else f"({backtest_ids[0]})"
        )
        self.safe_execute(
            self.cursor, f"SELECT * FROM BacktestRuns WHERE id IN {id_list}"
        )
        backtest_rows = self.cursor.fetchall()
        backtest_cols = [desc[0] for desc in self.cursor.description]
        backtests_df = pd.DataFrame(backtest_rows, columns=backtest_cols)

        # Rename for merge
        backtests_df.rename(columns={"id": "Backtest_ID"}, inplace=True)

        # Merge and export
        merged_df = composite_df.merge(backtests_df, on="Backtest_ID", how="left")

        # Sort by Score
        merged_df.sort_values(by="Score", ascending=False, inplace=True)

        merged_df.to_csv(output_path, index=False)
        logging.info(
            f"Exported composite results to {output_path} ({len(merged_df)} rows)."
        )

    def select_top_percent_strategies(
        self, exploration_space: str = "default", top_percent: float = 10.0
    ) -> list[dict]:
        """
        Selects the top N% performing strategies from a given exploration space
        based on composite score.

        Args:
            exploration_space (str): Exploration space label to filter Phase 1 runs.
            top_percent (float): Top X percent to keep (e.g., 10.0 = top 10%).

        Returns:
            list[dict]: Strategy metadata including indicator config and score.
        """
        assert 0 < top_percent <= 100, "top_percent must be between 0 and 100"

        query = """
            SELECT 
                cs.Score,
                cs.Trial_ID,
                cs.Study_Name,
                br.Net_Profit,
                br.Timeframe,
                fp.symbol AS Pair,
                sc.id AS StrategyConfig_ID,
                sc.parameters AS Parameters_JSON
            FROM CompositeScores cs
            JOIN BacktestRuns br ON br.id = cs.Backtest_ID
            JOIN StrategyConfigurations sc ON br.Strategy_ID = sc.id
            JOIN ForexPairs fp ON br.Pair_ID = fp.id
            WHERE cs.Exploration_Space = ?
        """

        self.safe_execute(self.cursor, query, (exploration_space,))
        rows = self.cursor.fetchall()

        if not rows:
            return []

        # Parse and structure the results
        results = [
            {
                "Score": row[0],
                "Trial_ID": row[1],
                "Study_Name": row[2],
                "Net_Profit": row[3],
                "Timeframe": row[4],
                "Pair": row[5],
                "StrategyConfig_ID": row[6],
                "Parameters": json.loads(row[7]),
            }
            for row in rows
        ]

        # Sort by composite score and slice top N%
        results.sort(key=lambda x: x["Score"], reverse=True)
        keep_count = max(1, int(len(results) * top_percent / 100.0))
        return results[:keep_count]
