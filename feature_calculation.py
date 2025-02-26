import numpy as np 
import pandas as pd
import ta
import os
from scipy.stats import linregress

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the CSV data.
    """
    df = pd.read_csv(file_path, header=[0, 1])
    df.rename(columns={'Unnamed: 0_level_0': 'Date'}, inplace=True)
    df = df.drop(columns=[('Date', 'Unnamed: 0_level_1.1')])
    df = df.iloc[1:, :].reset_index(drop=True)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    date_column_candidates = [col for col in df.columns if 'Date' in col]
    if date_column_candidates:
        df.rename(columns={date_column_candidates[0]: 'Date'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    
    return df


def split_data_by_ticker(df):
    """
    Split the main DataFrame into separate DataFrames for each ticker.
    """
    tickers = list(set([col.split('_')[0] for col in df.columns if '_' in col and col != 'Date']))
    ticker_dataframes = {}
    
    for ticker in tickers:
        columns = ['Date'] + [col for col in df.columns if col.startswith(ticker + '_')]
        ticker_df = df[columns].copy()
        ticker_df.columns = ['Date'] + [col.replace(ticker + '_', '') for col in ticker_df.columns if col != 'Date']
        
        for col in ticker_df.columns:
            if col != 'Date':
                ticker_df[col] = pd.to_numeric(ticker_df[col], errors='coerce')
        
        ticker_dataframes[ticker] = ticker_df

    return tickers, ticker_dataframes

    
def calculate_slope(data):
    """
    Calculate slope, p-value, and R^2 value given a data series.
    """
    if len(data) < 2 or np.isnan(data).any():
        return np.nan, np.nan, np.nan
    
    x_axis = np.arange(len(data))
    regression_model = linregress(x_axis, data)
    slope, r_value, p_value = round(regression_model.slope, 3), round(abs(regression_model.rvalue), 3), round(regression_model.pvalue, 4)
    
    return slope, r_value, p_value


def apply_rolling_metrics(series, window, metric_func):
    """
    Apply a rolling window calculation for slope, r_value, and p_value.
    """
    slopes = np.full(len(series), np.nan)
    r_values = np.full(len(series), np.nan)
    p_values = np.full(len(series), np.nan)

    for i in range(window - 1, len(series)):
        window_data = series[i - window + 1:i + 1]
        if not np.isnan(window_data).any():
            slope, r_value, p_value = metric_func(window_data)
            slopes[i] = slope
            r_values[i] = r_value
            p_values[i] = p_value

    return slopes, r_values, p_values


def calculate_eom(high, low, volume, window = 14, norm_factor = 1e6):
    """
    Calculate the EOM given a data series.
    """
    delta_p = ((high + low) / 2).diff()
    price_range = high - low
    volume_adjustment = volume / norm_factor
    eom = delta_p / (price_range * volume_adjustment)
    rolling_eom = eom.rolling(window = window).mean()
    
    return rolling_eom

def calculate_acc_dist(high, low, close, volume):
    """
    Calculate the Accumulation/Distribution Index (Acc/Dist) given a data series.
    """
    mfv = (((close - low) - (high - close)) / (high - low)) * volume
    
    acc_dist = mfv.cumsum()
    
    return acc_dist

def calculate_features(ticker_dataframes, tickers, history_to_use = 5):
    # Dictionary to store feature dataframes
    technical_indicators_dictionary = {}

    # List of features and metrics
    features = [
        "rsi_5", "rsi_10", "rsi_15",
        "stoch_5", "stoch_10", "stoch_15",
        "acc_dist",
        "eom_5", "eom_10", "eom_20",
        "cci_5", "cci_10", "cci_20",
        "daily_return", "volume_returns"
    ]
    trend_metrics = ["slope", "r_value", "p_value"]

    # Initialize DataFrames for each feature and trend metric
    for feature in features:
        if feature != 'acc_dist':
            technical_indicators_dictionary[feature] = pd.DataFrame(index = ticker_dataframes[list(ticker_dataframes.keys())[0]]['Date'])
        if feature != 'daily_return':
            for metric in trend_metrics:
                technical_indicators_dictionary[f"{feature}_{metric}"] = pd.DataFrame(index = ticker_dataframes[list(ticker_dataframes.keys())[0]]['Date'])

    # Loop through each ticker to calculate features
    for ticker in tickers:
        ticker_data = ticker_dataframes[ticker]
        dates = ticker_data['Date']

        # Relative Strength Index (RSI)
        for history in [5, 10, 15]:
            rsi = ta.momentum.RSIIndicator(ticker_data['Adj Close'], window = history, fillna = False).rsi()
            rsi[ticker_data['Adj Close'].isna()] = np.nan
            slopes, r_values, p_values = apply_rolling_metrics(
                rsi.values, history_to_use, calculate_slope
            )

            technical_indicators_dictionary[f"rsi_{history}"][ticker] = rsi.values
            technical_indicators_dictionary[f"rsi_{history}_slope"][ticker] = slopes
            technical_indicators_dictionary[f"rsi_{history}_r_value"][ticker] = r_values
            technical_indicators_dictionary[f"rsi_{history}_p_value"][ticker] = p_values

        # Stochastic Oscillator
        for history in [5, 10, 15]:
            stoch = ta.momentum.StochasticOscillator(
                ticker_data['High'], ticker_data['Low'], ticker_data['Adj Close'], 
                window = history, smooth_window = int(history / 3), fillna = False
            ).stoch()
            slopes, r_values, p_values = apply_rolling_metrics(
                stoch.values, history_to_use, calculate_slope
            )

            technical_indicators_dictionary[f"stoch_{history}"][ticker] = stoch.values
            technical_indicators_dictionary[f"stoch_{history}_slope"][ticker] = slopes
            technical_indicators_dictionary[f"stoch_{history}_r_value"][ticker] = r_values
            technical_indicators_dictionary[f"stoch_{history}_p_value"][ticker] = p_values

        # Accumulation/Distribution Index (Acc/Dist)
        acc_dist = calculate_acc_dist(ticker_data['High'], ticker_data['Low'], ticker_data['Adj Close'], ticker_data['Volume'])
        slopes, r_values, p_values = apply_rolling_metrics(
            acc_dist.values, history_to_use, calculate_slope
        )

        technical_indicators_dictionary['acc_dist_slope'][ticker] = slopes
        technical_indicators_dictionary['acc_dist_r_value'][ticker] = r_values
        technical_indicators_dictionary['acc_dist_p_value'][ticker] = p_values

        # Ease of Movement (EoM)
        for history in [5, 10, 20]:
            eom = calculate_eom(ticker_data['High'], ticker_data['Low'], ticker_data['Volume'], window = history)
            slopes, r_values, p_values = apply_rolling_metrics(
                eom.values, history_to_use, calculate_slope
            )

            technical_indicators_dictionary[f"eom_{history}"][ticker] = eom.values
            technical_indicators_dictionary[f"eom_{history}_slope"][ticker] = slopes
            technical_indicators_dictionary[f"eom_{history}_r_value"][ticker] = r_values
            technical_indicators_dictionary[f"eom_{history}_p_value"][ticker] = p_values

        # Commodity Channel Index (CCI)
        for history in [5, 10, 20]:
            cci = ta.trend.cci(
                ticker_data['High'], ticker_data['Low'], ticker_data['Adj Close'], window = history, constant = 0.015, fillna = False
            )
            slopes, r_values, p_values = apply_rolling_metrics(
                cci.values, history_to_use, calculate_slope
            )

            technical_indicators_dictionary[f"cci_{history}"][ticker] = cci.values
            technical_indicators_dictionary[f"cci_{history}_slope"][ticker] = slopes
            technical_indicators_dictionary[f"cci_{history}_r_value"][ticker] = r_values
            technical_indicators_dictionary[f"cci_{history}_p_value"][ticker] = p_values

        # Daily Returns
        daily_return = ticker_data['Adj Close'].pct_change()
        technical_indicators_dictionary['daily_return'][ticker] = daily_return.values

        # Volume Returns
        volume_returns = ticker_data['Volume'].pct_change()
        slopes, r_values, p_values = apply_rolling_metrics(
            volume_returns.values, history_to_use, calculate_slope
        )

        technical_indicators_dictionary['volume_returns'][ticker] = volume_returns.values
        technical_indicators_dictionary['volume_returns_slope'][ticker] = slopes
        technical_indicators_dictionary['volume_returns_r_value'][ticker] = r_values
        technical_indicators_dictionary['volume_returns_p_value'][ticker] = p_values

    return technical_indicators_dictionary


def save_feature_dataframes(feature_dataframes, output_path='feature_data/'):
    """
    Save each ticker's features as a separate CSV file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for feature, df in feature_dataframes.items():
        for ticker in df.columns:
            file_name = f"{ticker}_{feature}.csv"
            file_path = os.path.join(output_path, file_name)
            df[[ticker]].to_csv(file_path, index=True)

def main():
    file_path = './Data/Russell2000_total.csv'
    output_path = './Data/feature_data/'
    
    df = load_and_preprocess_data(file_path)
    tickers, ticker_dataframes = split_data_by_ticker(df)
    feature_dataframes = calculate_features(ticker_dataframes, tickers)
    save_feature_dataframes(feature_dataframes, output_path)
    
    print("Feature data has been successfully saved to:", output_path)

if __name__ == "__main__":
    main()