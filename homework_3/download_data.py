import yfinance as yf
import pandas as pd


def main():
    cryptos = [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "XRP-USD",
        "ADA-USD",
        "SOL-USD",
        "DOGE-USD",
    ]

    start_date = "2023-09-18"
    end_date = "2024-09-19"
    interval = "1h"

    # Dictionary { crypto: prices }
    crypto_data = {}

    # Fetch data for each crypto. Download hourly-level data
    for crypto in cryptos:
        crypto_data[crypto] = yf.download(
            crypto, start=start_date, end=end_date, interval=interval
        )
        print(
            f"Descarga de datos para {crypto}: {crypto_data[crypto].shape[0]} datos - desde {crypto_data[crypto].index[0]} a {crypto_data[crypto].index[-1]}"
        )

    # we consider close prices
    crypto_data = {crypto: crypto_data[crypto]["Close"] for crypto in cryptos}

    # filter extra hours. we use the exact interval of time we used for Seq2Seq models

    start_timestamp = pd.to_datetime("2023-09-18 03:00:00+00:00")
    end_timestamp = pd.to_datetime("2024-09-18 02:00:00+00:00")

    time_index = pd.date_range(
        start=start_timestamp, end=end_timestamp, freq="1h"
    )  # hours from start_timestamp to end_timestamp

    print(
        "Intervalo de tiempo de los datos: desde", start_timestamp, "a", end_timestamp
    )

    # detecting missing times
    all_times_set = set(time_index)
    for crypto in cryptos:
        missing_times = all_times_set.difference(crypto_data[crypto].index)
        print(f"Tiempos faltantes para crypto {crypto}: {len(missing_times)}")
        print(missing_times)
        print("\n")

    for crypto in cryptos:
        crypto_data[crypto] = crypto_data[crypto].reindex(time_index, method="ffill")
        print(f"Longitud de serie de tiempo {crypto}: {crypto_data[crypto].size}")

    dataset_df = pd.DataFrame(crypto_data)
    dataset_df.to_csv("cryptocoins-timeseries.csv")


if __name__ == "__main__":
    main()
