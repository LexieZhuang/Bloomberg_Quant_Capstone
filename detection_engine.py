import json
import collections
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from data_loader import DataEngine
import warnings
import shap
warnings.filterwarnings("ignore")

plt.rc('grid', linestyle="dotted", color='#a0a0a0')
plt.rcParams['axes.edgecolor'] = "#04383F"

class Args:
	def __init__(self):
		self.top_n = 2
		self.min_volume = 5000
		self.history_to_use = 5
		self.is_load_from_dictionary = 0
		self.data_dictionary_path = "dictionaries/data_dictionary.npy"
		self.data_feature_path = "Data/Feature_Values"
		self.is_save_dictionary = 1
		self.data_granularity_minutes = 60
		self.is_test = 1
		self.future_bars = 5 # Future for testing = 1 
		self.volatility_filter = 0.05
		self.output_format = "CLI"
		self.stock_list = "stocks.txt"
		self.data_source = "yahoo_finance"
# Get arguments (either from argparse or manually set in Jupyter)
args = Args()

# args = argParser.parse_args()
top_n = args.top_n
min_volume = args.min_volume
history_to_use = args.history_to_use
is_load_from_dictionary = args.is_load_from_dictionary
data_dictionary_path = args.data_dictionary_path
data_feature_path = args.data_feature_path
is_save_dictionary = args.is_save_dictionary
data_granularity_minutes = args.data_granularity_minutes
is_test = args.is_test
future_bars = args.future_bars
volatility_filter = args.volatility_filter
output_format = args.output_format.upper()
stock_list = args.stock_list
data_source = args.data_source


class Surpriver:
	def __init__(self,START,END,ticker_dataframes,feature_dataframes,error_dataframes):
		print("Surpriver has been initialized...")
		self.TOP_PREDICTIONS_TO_PRINT = top_n
		self.HISTORY_TO_USE = history_to_use
		self.MINIMUM_VOLUME = min_volume
		self.IS_LOAD_FROM_DICTIONARY = is_load_from_dictionary
		self.DATA_DICTIONARY_PATH = data_dictionary_path
		self.IS_SAVE_DICTIONARY = is_save_dictionary
		self.DATA_GRANULARITY_MINUTES = data_granularity_minutes
		self.IS_TEST = is_test
		self.FUTURE_BARS_FOR_TESTING = future_bars
		self.VOLATILITY_FILTER = volatility_filter
		self.OUTPUT_FORMAT = output_format
		self.STOCK_LIST = stock_list
		self.DATA_SOURCE = data_source
		self.start_time = START
		self.end_time = END
		self.ticker_dataframes = ticker_dataframes
		self.feature_dataframes = feature_dataframes
		self.error_dataframes = error_dataframes
		self.key_list = []
		self.dispersion_metrics = None
		self.threshold = None
		self.df_anomaly_normal = None
		

		# Create data engine
		self.dataEngine = DataEngine(self.HISTORY_TO_USE, self.DATA_GRANULARITY_MINUTES, 
							self.IS_SAVE_DICTIONARY, self.IS_LOAD_FROM_DICTIONARY,self.DATA_DICTIONARY_PATH,
							self.MINIMUM_VOLUME,
							self.IS_TEST, self.FUTURE_BARS_FOR_TESTING,
							self.VOLATILITY_FILTER,
							self.STOCK_LIST, self.DATA_SOURCE,self.start_time,self.end_time, self.ticker_dataframes,self.feature_dataframes,self.error_dataframes)
		
	def is_nan(self, object):
		return object != object

	def calculate_percentage_change(self, old, new):
		return ((new - old) * 100) / old

	def calculate_return(self, old, new):
		return new / old

	def parse_large_values(self, value):
		if value < 1000:
			value = str(value)
		elif value >= 1000 and value < 1000000:
			value = round(value / 1000, 2)
			value = str(value) + "K"
		else:
			value = round(value / 1000000, 1)
			value = str(value) + "M"

		return value

	def calculate_volume_changes(self, historical_price):
		volume = list(historical_price["Volume"])
		dates = list(historical_price["Datetime"])
		dates = [str(date) for date in dates]

		# Get volume by date
		volume_by_date_dictionary = collections.defaultdict(list)
		for j in range(0, len(volume)):
			date = dates[j].split(" ")[0]
			volume_by_date_dictionary[date].append(volume[j])

		for key in volume_by_date_dictionary:
			volume_by_date_dictionary[key] = np.sum(volume_by_date_dictionary[key]) # taking average as we have multiple bars per day. 

		# Get all dates
		all_dates = list(reversed(sorted(volume_by_date_dictionary.keys())))
		latest_date = all_dates[0]
		latest_data_point =  list(reversed(sorted(dates)))[0]

		# Get volume information
		today_volume = volume_by_date_dictionary[latest_date]
		average_vol_last_five_days = np.mean([volume_by_date_dictionary[date] for date in all_dates[1:6]])
		average_vol_last_twenty_days = np.mean([volume_by_date_dictionary[date] for date in all_dates[1:20]])
		return latest_data_point, self.parse_large_values(today_volume), self.parse_large_values(average_vol_last_five_days), self.parse_large_values(average_vol_last_twenty_days)


	def calculate_recent_volatility(self, historical_price):
		close_price = list(historical_price["Close"])
		volatility_five_bars = np.std(close_price[-5:])
		volatility_twenty_bars = np.std(close_price[-20:])
		volatility_all = np.std(close_price)
		return volatility_five_bars, volatility_twenty_bars, volatility_all


	def calculate_future_performance(self, future_data):
		CLOSE_PRICE_INDEX = 3
		price_at_alert = future_data[0][CLOSE_PRICE_INDEX]
		prices_in_future = [item[CLOSE_PRICE_INDEX] for item in future_data[1:]]
		prices_in_future = [item for item in prices_in_future if item != 0]
		total_sum_percentage_change = abs(sum([self.calculate_percentage_change(price_at_alert, next_price) for next_price in prices_in_future]))
		future_volatility = np.std(prices_in_future)
		return total_sum_percentage_change, future_volatility
	
    
	def find_anomalies(self):

		# Gather data for all stocks
		if self.IS_LOAD_FROM_DICTIONARY == 0:
			features, historical_price_info, future_prices, future_errors, symbol_names, keys_list = self.dataEngine.collect_data_for_all_tickers()
			if not self.key_list:
				self.key_list = keys_list
		else:
			# Load data from dictionary
			features, historical_price_info, future_prices, future_errors, symbol_names, keys_list = self.dataEngine.load_data_from_dictionary()
			if not self.key_list:
				self.key_list = keys_list		

		detector = IsolationForest(n_estimators = 100, random_state = 0)
		detector.fit(features)
		self.detector = detector
		self.features = features
		predictions = detector.decision_function(features)
		features_df = pd.DataFrame(features)
		self.features_df = features_df

		# Print top predictions with some statistics
		predictions_with_output_data = [[predictions[i], symbol_names[i], historical_price_info[i], future_prices[i], future_errors[i]] for i in range(0, len(predictions))]
		predictions_with_output_data = list(sorted(predictions_with_output_data))
		results = []

		for item in predictions_with_output_data[:self.TOP_PREDICTIONS_TO_PRINT]:
			# Get some stats to print
			prediction, symbol, historical_price, future_price, future_error = item
			# Check if future data is present or not
			if self.IS_TEST == 1 and len(future_price) == 0:
				print("No future data is present. Please make sure that you ran the prior command with is_test enabled or disable that command now. Exiting now...")
				exit()

			latest_date, today_volume, average_vol_last_five_days, average_vol_last_twenty_days = self.calculate_volume_changes(historical_price)
			volatility_vol_last_five_days, volatility_vol_last_twenty_days, _ = self.calculate_recent_volatility(historical_price)
			if average_vol_last_five_days == None or volatility_vol_last_five_days == None:
				continue

			if self.IS_TEST == 0:
				
				if self.OUTPUT_FORMAT == "CLI":
					print("Last Bar Time: %s\nSymbol: %s\nAnomaly Score: %.3f\nToday Volume: %s\nAverage Volume 5d: %s\nAverage Volume 20d: %s\nVolatility 5bars: %.3f\nVolatility 20bars: %.3f\n----------------------" % 
																	(latest_date, symbol, prediction,
																	today_volume, average_vol_last_five_days, average_vol_last_twenty_days,
																	volatility_vol_last_five_days, volatility_vol_last_twenty_days))
				results.append({
					'latest_date' : latest_date,
					'Symbol' : symbol,
					'Anomaly Score' : prediction,
					'Today Volume' : today_volume,
					'Average Volume 5d' : average_vol_last_five_days,
					'Average Volume 20d' : average_vol_last_twenty_days,
					'Volatility 5bars' : volatility_vol_last_five_days,
					'Volatility 20bars' : volatility_vol_last_twenty_days
				})

			else:
				# Testing so show what happened in the future
				future_abs_sum_percentage_change, _ = self.calculate_future_performance(future_price)
				future_error_percentage_change = future_error
				if self.OUTPUT_FORMAT == "CLI":
					print("Last Bar Time: %s\nSymbol: %s\nAnomaly Score: %.3f\nToday Volume: %s\nAverage Volume 5d: %s\nAverage Volume 20d: %s\nVolatility 5bars: %.3f\nVolatility 20bars: %.3f\nFuture Absolute Sum Price Changes: %.2f\n----------------------" % 
																	(latest_date, symbol, prediction,
																	today_volume, average_vol_last_five_days, average_vol_last_twenty_days,
																	volatility_vol_last_five_days, volatility_vol_last_twenty_days,
																	future_abs_sum_percentage_change))
				results.append({
					'latest_date' : latest_date,
					'Symbol' : symbol,
					'Anomaly Score' : prediction,
					'Today Volume' : today_volume,
					'Average Volume 5d' : average_vol_last_five_days,
					'Average Volume 20d' : average_vol_last_twenty_days,
					'Volatility 5bars' : volatility_vol_last_five_days,
					'Volatility 20bars' : volatility_vol_last_twenty_days,
					'Future Absolute Sum Price Changes' : future_abs_sum_percentage_change,
					'Future Error Mean': future_error_percentage_change
				})

		if self.OUTPUT_FORMAT == "JSON":
			self.store_results(results)
		self.prediction = predictions_with_output_data
		return predictions_with_output_data


	def calculate_shape(self):
		"""
		Function to calculate shap value.
		"""
		column_names = self.key_list
		shap_values = shap.TreeExplainer(self.detector).shap_values(np.array(self.features_df))
		return shap.summary_plot(shap_values, features=np.array(self.features_df), feature_names = column_names)


	def store_results(self, results):
		"""
		Function for storing results in a file
		"""
		today = dt.datetime.today().strftime('%Y-%m-%d')
		
		prefix = "results"

		if self.IS_TEST != 0:
			prefix = "results_future"

		file_name = '%s_%s.json' % (prefix, str(today))

		#Print results to Result File
		with open(file_name, 'w+') as result_file:
			json.dump(results, result_file)

		print("Results stored successfully in", file_name)


	def calculate_MAD(self):
		predictions_with_output_data = self.prediction
		anomalous_score = []

		for item in predictions_with_output_data:
			prediction, symbol, historical_price, future_price, future_error= item
			anomalous_score.append(prediction)
		self.threshold = np.percentile(anomalous_score, 5) - 0.01
		anomalous_vs_normal = np.array([1 if anomalous_score[x] < self.threshold else 0 for x in range(0, len(anomalous_score))])
		# quantify the dispersion using Variance and IQR
		# Create a DataFrame for calculations
		data = pd.DataFrame({
			'anomalous_score': anomalous_score,
			'group': anomalous_vs_normal
		})
		# Calculate Variance and IQR for each group
		self.dispersion_metrics = data.groupby('group').agg(Count = ('anomalous_score', 'size'), Std = ('anomalous_score', 'std'), MAD = ('anomalous_score', lambda x: np.mean(np.abs(x - np.mean(x)))))
		self.dispersion_metrics.columns = ['size', 'Standard Deviation', 'MAD']
		print(self.dispersion_metrics['size'])
		return self.dispersion_metrics
	

	def calculate_future_stats(self):
		"""
		Calculate different stats for future data to show whether the anomalous stocks found were actually better than non-anomalous ones
		"""
		predictions_with_output_data = self.prediction
		future_change = []
		anomalous_score = []
		historical_volatilities = []
		future_volatilities = []
		future_errors = []
		symbols = []
		for item in predictions_with_output_data:
			prediction, symbol, historical_price, future_price, future_error= item
			future_sum_percentage_change, future_volatility = self.calculate_future_performance(future_price)
			_, _, historical_volatility = self.calculate_recent_volatility(historical_price)
			error_term = future_error.mean()

			# Skip for when there is a reverse split, the yfinance package does not handle that well so percentages get weirdly large
			if abs(future_sum_percentage_change) > 250 or self.is_nan(future_sum_percentage_change) == True or self.is_nan(prediction) == True:
				continue

			symbols.append(symbol)
			future_change.append(future_sum_percentage_change)
			anomalous_score.append(prediction)
			future_volatilities.append(future_volatility)
			historical_volatilities.append(historical_volatility)
			future_errors.append(error_term)
		
		print("\n*************** Future Performance ***************")

		# Plot
		FONT_SIZE = 14
		colors = ['#c91414' if anomalous_score[x] < 0 else '#035AA6' for x in range(0, len(anomalous_score))]
		anomalous_vs_normal = np.array([1 if anomalous_score[x] < self.threshold else 0 for x in range(0, len(anomalous_score))])

		self.df_anomaly_normal = pd.DataFrame({'Symbol': symbols, 'Anomalous_Score': anomalous_score, 'Future_Error': future_errors, 'Anomaly':anomalous_vs_normal})
		
		# Calculate the variance and IQR annotations for plotting
		group_0_var = self.dispersion_metrics.loc[0, 'Standard Deviation']
		group_0_iqr = self.dispersion_metrics.loc[0, 'MAD']
		group_1_var = self.dispersion_metrics.loc[1, 'Standard Deviation']
		group_1_iqr = self.dispersion_metrics.loc[1, 'MAD']

		plt.figure(figsize = (10, 6))
		plt.scatter(np.array(anomalous_score)[anomalous_vs_normal == 1], np.array(future_errors)[anomalous_vs_normal == 1], marker = 'v', color = '#c91414')
		plt.scatter(np.array(anomalous_score)[anomalous_vs_normal == 0], np.array(future_errors)[anomalous_vs_normal == 0], marker = 'P', color = '#035AA6')

		# Add annotations for Standard Deviation and IQR
		x_min, x_max = plt.xlim()
		y_min, y_max = plt.ylim()
		plt.text(x_min + 0.01, y_max - 0.05, f"Anomalous\nStd: {group_1_var:.2f}\nMAD: {group_1_iqr:.2f}", color='#c91414', fontsize=FONT_SIZE)
		plt.text(x_max - 0.03, y_max - 0.05, f"Normal\nStd: {group_0_var:.2f}\nMAD: {group_0_iqr:.2f}", color='#035AA6', fontsize=FONT_SIZE)


		plt.axvline(x = self.threshold, linestyle = '--', color = '#848484')
		plt.xlabel("Anomaly Score", fontsize = FONT_SIZE)
		plt.ylabel("Future Errors", fontsize = FONT_SIZE)
		plt.xticks(fontsize = FONT_SIZE)
		plt.yticks(fontsize = FONT_SIZE)
		plt.legend(["Anomalous", "Normal"], loc = 'lower left', fontsize = FONT_SIZE)
		plt.title("Anomaly Score vs Future Errors", fontsize = FONT_SIZE)
		plt.tight_layout()
		plt.grid()
		plt.show()
		return self.df_anomaly_normal