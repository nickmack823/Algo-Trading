import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.widgets import MultiCursor
from matplotlib.dates import num2date

def plot_backtest(data: pd.DataFrame, trades: list, title: str = "Backtest Visualization"):
    """
    Visualizes backtest results with two synchronized subplots (price and ATR),
    a crosshair cursor that spans both plots, and annotations showing the current
    x (date/time) and y (price or ATR) values near the cursor.
    
    The price subplot will have its y-axis set based solely on the Close data,
    so it won't automatically drop to 0.
    
    Parameters:
      data (pd.DataFrame): Historical market data with a DatetimeIndex (or a 'Timestamp' column that will be converted).
                           Must include at least 'Close' and 'ATR' columns.
      trades (list): List of trade objects with attributes such as:
                     - entry_timestamp, entry_price, trade_type,
                     - exit_timestamp (if closed), exit_price (if closed).
      title (str): Title for the chart.
    """
    # Ensure the data uses a DatetimeIndex.
    if not isinstance(data.index, pd.DatetimeIndex):
        if "Timestamp" in data.columns:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            data.set_index('Timestamp', inplace=True)
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Timestamp' column.")
    
    # Create figure with two subplots and use constrained_layout to prevent overlap.
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, 
        constrained_layout=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    
    # --- Top Subplot: Price and Indicators ---
    ax1.plot(data.index, data['Close'], label='Close Price', color='black', linewidth=1.5)
    if 'EMA_9' in data.columns:
        ax1.plot(data.index, data['EMA_9'], label='EMA 9', color='blue', linewidth=1)
    if 'EMA_21' in data.columns:
        ax1.plot(data.index, data['EMA_21'], label='EMA 21', color='orange', linewidth=1)
    
    # Plot trade markers on ax1.
    for trade in trades:
        entry_time = pd.to_datetime(trade.entry_timestamp)
        entry_price = trade.entry_price
        if trade.trade_type.upper() == 'BUY':
            entry_style = {'marker': '^', 'color': 'green', 's': 100, 'label': 'Buy Entry'}
            exit_style  = {'marker': 'v', 'color': 'red', 's': 100, 'label': 'Buy Exit'}
        elif trade.trade_type.upper() == 'SELL':
            entry_style = {'marker': 'v', 'color': 'red', 's': 100, 'label': 'Sell Entry'}
            exit_style  = {'marker': '^', 'color': 'green', 's': 100, 'label': 'Sell Exit'}
        else:
            continue

        ax1.scatter(entry_time, entry_price, **entry_style)
        if getattr(trade, 'exit_timestamp', None) and getattr(trade, 'exit_price', None):
            exit_time = pd.to_datetime(trade.exit_timestamp)
            exit_price = trade.exit_price
            # Add a small horizontal offset if entry and exit occur on the same candle.
            if entry_time == exit_time:
                exit_time += pd.Timedelta(minutes=1)
            ax1.scatter(exit_time, exit_price, **exit_style)
            ax1.plot([entry_time, exit_time], [entry_price, exit_price],
                     color='gray', linestyle='--', linewidth=1)
    
    # Remove duplicate legend entries.
    handles, labels = ax1.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    ax1.legend(unique.values(), unique.keys())
    ax1.set_ylabel("Price")
    ax1.set_title(title)
    
    # Set y-axis limits for the price chart based solely on the price data.
    price_min = data['Close'].min()
    price_max = data['Close'].max()
    margin = (price_max - price_min) * 0.05  # 5% margin
    ax1.set_ylim(price_min - margin, price_max + margin)
    
    # --- Bottom Subplot: ATR ---
    if 'ATR' in data.columns:
        ax2.plot(data.index, data['ATR'], label='ATR', color='purple', linewidth=1)
        ax2.set_ylabel("ATR")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "ATR data not available", horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
    ax2.set_xlabel("Date")
    
    # Set x-axis formatter to display dates.
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    # --- Synchronized Cursor Across Subplots ---
    multi = MultiCursor(fig.canvas, (ax1, ax2), color='gray', lw=1, horizOn=True, vertOn=True)
    
    # --- Annotations for Displaying X and Y Values ---
    annot1 = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                          bbox=dict(boxstyle="round", fc="w"),
                          arrowprops=dict(arrowstyle="->"))
    annot1.set_visible(False)
    annot2 = ax2.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                          bbox=dict(boxstyle="round", fc="w"),
                          arrowprops=dict(arrowstyle="->"))
    annot2.set_visible(False)

    def on_move(event):
        """
        Called when the user moves the mouse over the plot. Updates the
        annotations (tooltips) on the plot to display the current date and
        price/ATR value under the mouse cursor. The function implements a
        throttle to prevent excessive updates, only allowing an update if
        at least 0.05 seconds (50ms) have passed since the last update.
        """
        # Throttle: update only if at least 0.05 seconds (50ms) have passed.
        if event.inaxes == ax1:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            annot1.xy = (x, y)
            date = num2date(x)
            annot1.set_text(f"{date.strftime('%Y-%m-%d %H:%M:%S')}\nPrice: {y:.2f}")
            annot1.set_visible(True)
            annot2.set_visible(False)
            fig.canvas.draw_idle()
        elif event.inaxes == ax2:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            annot2.xy = (x, y)
            date = num2date(x)
            annot2.set_text(f"{date.strftime('%Y-%m-%d %H:%M:%S')}\nATR: {y:.2f}")
            annot2.set_visible(True)
            annot1.set_visible(False)
            fig.canvas.draw_idle()
        else:
            annot1.set_visible(False)
            annot2.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.show()
