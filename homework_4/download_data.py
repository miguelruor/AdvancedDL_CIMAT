import yfinance as yf
import pandas as pd


def main():
    symbols = [
        "EURUSD=X",
        "JPYUSD=X",
        "GBPUSD=X",
        "MXNUSD=X",
        "CHFUSD=X",
        "CL=F",
        "GC=F",
    ]

    start_date = pd.to_datetime("2023-01-01", utc=True)
    end_date = pd.to_datetime("2024-01-01", utc=True)
    interval = "1h"

    # Dictionary { symbol: prices }
    data = {}

    # Fetch data for each symbol. Download hourly-level data for 1 year
    for sym in symbols:
        data[sym] = yf.download(sym, start=start_date, end=end_date, interval=interval)
        print(
            f"Descarga de datos para {sym}: {data[sym].shape[0]} datos - desde {data[sym].index[0]} a {data[sym].index[-1]}"
        )

        # change to UTC
        data[sym].index = data[sym].index.tz_convert("UTC")

    # we consider close prices
    data = {sym: data[sym]["Close"] for sym in symbols}

    # consider the case when timeseries do not have
    start_timestamp = max([data[sym].index[0] for sym in symbols])
    end_timestamp = min([data[sym].index[-1] for sym in symbols])

    # hours from start_timestamp to end_timestamp
    time_index = pd.date_range(start=start_timestamp, end=end_timestamp, freq="1h")

    print(
        "Intervalo de tiempo de los datos: desde", start_timestamp, "a", end_timestamp
    )

    # detecting missing times
    all_times_set = set(time_index)
    for sym in symbols:
        missing_times = all_times_set.difference(data[sym].index)
        print(f"Tiempos faltantes para símbolo {sym}: {len(missing_times)}")
        print("\n")

    for sym in symbols:
        data[sym] = data[sym].reindex(time_index, method="ffill")
        print(f"Longitud de serie de tiempo {sym}: {data[sym].size}")

    dataset_df = pd.DataFrame(data)
    dataset_df.to_csv("currencies-ts.csv")


if __name__ == "__main__":
    main()
