from detection_engine import Surpriver
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.dates as mdates


class Result_analysis:
    def __init__(self, start_date, num_days, period_days):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.num_days = num_days
        self.period_days = period_days
        self.results = []
        self.ticker_dataframes = {}
        self.feature_dataframes = {}
        self.error_dataframes = self.load_error()
        self.Surpriver = []
        self.mad_group_0 = []
        self.mad_group_1 = []
        self.mad_date = []
        self.all_features_list = None
        
        if not self.ticker_dataframes:
            self.process_data()
        #************************************
        if not self.feature_dataframes:
            self.Load_features()
            self.standardize_features()


    def load_error(self):
        data = pd.read_csv('Data/Russell2000_error_30.csv')
        data.set_index('Date', inplace = True)
        tickers = data.columns
        feature_dataframes = {}
        for ticker in tickers:
            feature_dataframes[ticker] = data[ticker].dropna()
        return feature_dataframes
    
    
    def process_data(self):
        # read
        print("Downloading data for all stocks...")
        file_path = 'Data/Russell2000_total.csv'  # Replace with your file path
        df = pd.read_csv(file_path)

        # Rename 'Unnamed: 0_level_0' to 'Date' and set the first value of the column to 'Date'
        df.rename(columns = {'Unnamed: 0_level_0': 'Date'}, inplace = True)
        df.loc[0, 'Date'] = 'Date'

        # Identify unique tickers
        tickers = {col.split('.')[0] for col in df.columns if '.' in col}
        #************************************
        tickers.remove('Unnamed: 0_level_0')

        for ticker in tickers:
            # Extract columns for the ticker, including 'Date'
            columns = ['Date'] + [col for col in df.columns if col.split('.')[0] == ticker]
            ticker_df = df[columns].copy()

            # Use the first row as column names and drop unnecessary rows
            ticker_df.columns = ticker_df.iloc[0]
            ticker_df = ticker_df[2:].reset_index(drop = True)

            # Convert 'Date' to datetime and other columns to numeric
            ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], format = '%Y-%m-%d', errors = 'coerce')
            ticker_df.iloc[:, 1:] = ticker_df.iloc[:, 1:].apply(pd.to_numeric, errors = 'coerce')

            # Store the processed DataFrame
            self.ticker_dataframes[ticker] = ticker_df
        return

    
    #************************************
    # add a loading function to load all features into a dictionary, with key: feature_name, items: dataframe
    def Load_features(self):
        print("Loading all features...")
        self.all_features_list = pd.read_csv('Data/feature_data/all_features_df.csv')['feature'].values.tolist()
        for feature in self.all_features_list:
            self.feature_dataframes[feature] = pd.read_csv(f"Data/feature_data/{feature}_features.csv")
            self.feature_dataframes[feature]['Date'] = pd.to_datetime(self.feature_dataframes[feature]['Date'])
            self.feature_dataframes[feature].set_index('Date', inplace = True)


    #************************************
    # Method to standardize a single row
    def standardize_row(self, row):
        # Extract valid (non-NaN) values
        valid_values = row.dropna()
        if valid_values.empty:
            return row  # Return unchanged if all values are NaN
        # Standardize valid values
        scaler = StandardScaler()
        standardized_values = scaler.fit_transform(valid_values.values.reshape(-1, 1)).flatten()
        # Replace valid values in the original row with standardized values
        row.loc[valid_values.index] = standardized_values
        return row
    

    #************************************
    # Method to standardize all features
    def standardize_features(self):
        for key, df in self.feature_dataframes.items():
            try:
                # Replace infinite values before standardization
                df.replace([float('inf'), float('-inf')], np.nan, inplace = True)
                # Apply row-wise standardization
                self.feature_dataframes[key] = df.apply(self.standardize_row, axis = 1)
            except Exception as e:
                print(f"Error processing DataFrame '{key}': {e}")
                # Handle errors gracefully: Fill problematic rows with neutral values (e.g., 50)
                self.feature_dataframes[key] = df.apply(lambda row: row.fillna(50) if row.isnull().all() else self.standardize_row(row), axis = 1)
        print()


    def Load_Error_Terms(self):
        all_error_list = pd.read_csv('Data/Russell2000_error.csv')
        all_error_list

    
    def create_model(self, start_time, end_time):
        #************************************
        self.model = Surpriver(START = start_time, END = end_time, ticker_dataframes = self.ticker_dataframes, feature_dataframes = self.feature_dataframes, error_dataframes = self.error_dataframes)
        return self.model
    

    # calculate anonaly scores
    def process_results(self, raw_results):
        name, score, date1 = [], [], []
        for item in raw_results:
            name.append(item[1])
            score.append(item[0])
            date1.append(item[2]['Datetime'].iloc[-1].date().strftime('%Y-%m-%d'))
        result_df = pd.DataFrame({'symbol': name, 'score': score, 'Date': date1})
        result_df['anomaly'] = result_df['score'] < 0
        return result_df


    # run detection for multiple time slots
    def run_detection(self):
        date_list = [(self.start_date + timedelta(days = i)).strftime('%Y-%m-%d') for i in range(self.num_days)]
        for start_i in date_list:
            end_date = datetime.strptime(start_i, '%Y-%m-%d') + timedelta(days = self.period_days)
            end_i = end_date.strftime('%Y-%m-%d')
            print(start_i, end_i)
            self.create_model(start_i, end_i)
            raw_results = self.model.find_anomalies()
            dispersion_metrics = self.model.calculate_MAD()

            # Extract MAD values for each group
            self.mad_group_0.append(dispersion_metrics.loc[0, 'MAD'])
            self.mad_group_1.append(dispersion_metrics.loc[1, 'MAD'])
            self.mad_date.append(end_i)

            processed_data = self.process_results(raw_results)
            self.results.append(processed_data)
        combined_data = pd.concat(self.results, ignore_index = True)
        self.df = combined_data
        return combined_data
    
    
    # shap plot
    def Run_daily_Result_performance(self, time = '2024-08-01'):
        end_time = pd.to_datetime(time) +pd.Timedelta(days = self.period_days)
        #************************************
        # pass ticker_dataframes and feature_dataframes to the model
        end_time = end_time.strftime('%Y-%m-%d')
        model = Surpriver(START = time, END = end_time, ticker_dataframes = self.ticker_dataframes, feature_dataframes = self.feature_dataframes, error_dataframes = self.error_dataframes)
        self.Surpriver = model
        model.find_anomalies()
        return model.calculate_shape()
    
    
    def Single_stock_plot(self, ticker_name = 'AAPL'):
        # Set start and end dates
        start_date = self.start_date - pd.Timedelta(days = 3)
        end_date = self.start_date + pd.Timedelta(days = self.num_days + self.period_days + 1)

        # Get stock price data
        stock_prices = self.ticker_dataframes[ticker_name]
        stock_prices.rename(columns = {'Datetime': 'Date'}, inplace = True)

        # Calculate returns
        returns_price = stock_prices.copy()
        returns_price['Return'] = stock_prices['Close'].pct_change()
        returns_price = returns_price[(returns_price['Date'] >= start_date) & (returns_price['Date'] <= end_date)].dropna()

        # Filter and merge with anomaly data
        df = self.df
        df_symbol = df[df.symbol == ticker_name]
        df_symbol['Date'] = pd.to_datetime(df_symbol['Date'])
        returns_price['Date'] = pd.to_datetime(returns_price['Date'])
        plot_data = pd.merge(df_symbol, returns_price, on = 'Date', how = 'left').set_index('Date')

        plt.figure(figsize = (12, 6))
        plt.plot(plot_data.index, plot_data['Return'], label = 'Returns', color = 'blue', linewidth = 1.5)

        # Highlight anomalies
        anomaly_dates = plot_data.index[plot_data['anomaly'] == 1]
        anomaly_values = plot_data['Return'][plot_data['anomaly'] == 1]
        plt.scatter(anomaly_dates, anomaly_values, color = 'red', label = 'Anomalies', marker = 'o', zorder = 5)

        plt.title(f'Time Series of {ticker_name} Returns with Anomalies', fontsize = 16)
        plt.xlabel('Date', fontsize = 14)
        plt.ylabel('Returns', fontsize = 14)
        plt.legend(fontsize = 12, loc = 'upper left', frameon = True, shadow = True)
        plt.grid(visible = True, linestyle = '--', linewidth = 0.7, alpha = 0.7)
        plt.xticks(rotation = 45, fontsize = 10)
        plt.tight_layout()
        plt.show()


    def MAD_plot(self):
        # Ensure mad_date is properly formatted as datetime
        self.mad_date = pd.to_datetime(self.mad_date) 

        # Select 10 evenly spaced dates for x-axis ticks
        num_dates = len(self.mad_date)
        selected_indices = np.linspace(0, num_dates - 1, 10, dtype = int) 
        selected_dates = self.mad_date[selected_indices]

        academic_colors = ['#4E79A7', '#F28E2B']

        plt.figure(figsize = (12, 8))
        plt.stackplot(self.mad_date, self.mad_group_0, self.mad_group_1, labels = ['Normal Group MAD', 'Anomaly Group MAD'], colors = academic_colors, alpha = 0.85)

        plt.xticks(selected_dates, rotation = 45, fontsize = 12)
        plt.title('Mean Absolute Deviation (MAD) Over Time', fontsize = 18, fontweight = 'bold')
        plt.xlabel('Date', fontsize = 14, labelpad = 10)
        plt.ylabel('Mean Absolute Deviation (MAD)', fontsize = 14, labelpad = 10)
        plt.legend(fontsize = 12, loc = 'upper left', frameon = False)
        plt.grid(visible = True, linestyle = '--', linewidth = 0.5, alpha = 0.7, color = 'gray')
        plt.tight_layout()
        plt.show()


    # anomaly scores(anomaly score with absolute price change value)
    def anomaly_plot(self):
        df_anomaly = self.model.calculate_future_stats()
        return df_anomaly


if __name__ == "__main__":
    test_start_time = '2023-05-27'
    example1 = Result_analysis(test_start_time, 300, 30)
    final_results = example1.run_detection()
    example1.MAD_plot()
    example1.Run_daily_Result_performance(test_start_time)
    example1.Single_stock_plot('AAOI')
    anomaly_df = example1.anomaly_plot()