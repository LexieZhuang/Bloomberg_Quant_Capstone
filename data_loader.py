# Basic libraries
import os
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from feature_generator import TAEngine
import warnings
warnings.filterwarnings("ignore")

class DataEngine:
	def __init__(self, history_to_use, data_granularity_minutes, is_save_dict, 
			  is_load_dict, dict_path,min_volume_filter, is_test, future_bars_for_testing, 
			  volatility_filter, stocks_list, data_source, START,END,ticker_dataframes,
			  feature_dataframes,error_dataframes):
		print("Data engine has been initialized...")
		self.DATA_GRANULARITY_MINUTES = data_granularity_minutes
		self.IS_SAVE_DICT = is_save_dict
		self.IS_LOAD_DICT = is_load_dict
		self.DICT_PATH = dict_path
		self.VOLUME_FILTER = min_volume_filter
		self.FUTURE_FOR_TESTING = future_bars_for_testing
		self.IS_TEST = is_test
		self.VOLATILITY_THRESHOLD = volatility_filter
		self.DATA_SOURCE = data_source
		self.start_time = START
		self.end_time = END
		self.ticker_dataframes = ticker_dataframes
		#************************************
		self.feature_dataframes = ticker_dataframes
		self.error_dataframes = error_dataframes
		self.current = None

		# Stocks list
		self.directory_path = str(os.path.dirname(os.path.abspath(__file__)))
		self.stocks_file_path = self.directory_path + f"/stocks/{stocks_list}"
		self.stocks_list = []

		# Load stock names in a list
		self.load_stocks_from_file()

		#************************************
		# Load Technical Indicator engine
		self.taEngine = TAEngine(history_to_use = history_to_use, feature_dataframes = feature_dataframes)

		# Dictionary to store data. This will only store and save data if the argument is_save_dictionary is 1.
		self.features_dictionary_for_all_symbols = {}
		
		# Data length
		self.stock_data_length = []
		

	def load_stocks_from_file(self):
		"""
		Load stock names from the file
		"""
		print("Loading all stocks from file...")
		stocks_list = open(self.stocks_file_path, "r").readlines()
		stocks_list = [str(item).strip("\n") for item in stocks_list]

		# Load symbols
		stocks_list = list(sorted(set(stocks_list)))
		print("Total number of stocks: %d" % len(stocks_list))
		self.stocks_list = stocks_list


	def get_most_frequent_key(self, input_list):
		counter = collections.Counter(input_list)
		counter_keys = list(counter.keys())
		frequent_key = counter_keys[0]
		return frequent_key
	
    
	def get_data(self, symbol):
		"""
		Get stock data.
		"""
		
		try:
			historical_prices = []
			future_prices_list = []
			future_error_list = []
			if symbol in self.ticker_dataframes.keys():
				stock_prices = self.ticker_dataframes[symbol]

				# stock_prices = stock_prices.reset_index()
				stock_prices.columns = ['Datetime', 'High', 'Low', 'Close', 'Volume']
				# filter the dataframe to stay within the required date
				stock_prices = stock_prices[(stock_prices['Datetime'] >= self.start_time)&(stock_prices['Datetime'] <= self.end_time)]
				stock_prices = stock_prices[['Datetime', 'High', 'Low', 'Close', 'Volume']]

				data_length = len(stock_prices.values.tolist())
				self.stock_data_length.append(data_length)

			else:
				print(symbol)

			if stock_prices is None:
				raise ValueError(f"Stock data not found for symbol: {symbol}")
				
			# After getting some data, ignore partial data based on number of data samples
			if len(self.stock_data_length) > 5:
				most_frequent_key = self.get_most_frequent_key(self.stock_data_length)
				if data_length != most_frequent_key:
					print('if data_length != most_frequent_key:')
					return [], [],[], True
			
			if self.IS_TEST == 1:
				#************************************
				# Find the timestamp to identify the corresponding features
				self.current = stock_prices['Datetime'].iloc[-self.FUTURE_FOR_TESTING]
				self.current = pd.to_datetime(self.current)

				stock_prices_list = stock_prices.values.tolist()
				stock_prices_list = stock_prices_list[1:]
				future_prices_list = stock_prices_list[-(self.FUTURE_FOR_TESTING + 1):]
				future_error_list = self.error_dataframes[symbol]
				end_t = pd.to_datetime(self.end_time) + pd.Timedelta(days=self.FUTURE_FOR_TESTING)
				end_t = end_t.strftime('%Y-%m-%d')
				future_error_list = future_error_list[(future_error_list.index >= self.end_time)&(future_error_list.index <= end_t)]
				historical_prices = stock_prices_list[:-self.FUTURE_FOR_TESTING]
				historical_prices = pd.DataFrame(historical_prices)
				historical_prices.columns = ['Datetime', 'High', 'Low', 'Close', 'Volume']
				if len(future_error_list)==0 or len(historical_prices)==0:
					return [], [],[],True

			else:
				# No testing
				stock_prices_list = stock_prices.values.tolist()
				stock_prices_list = stock_prices_list[1:]
				historical_prices = pd.DataFrame(stock_prices_list)
				historical_prices.columns = ['Datetime', 'High', 'Low', 'Close', 'Volume']
				future_prices_list = []
				future_error_list = []

			if len(stock_prices.values.tolist()) == 0:
				print('if len(stock_prices.values.tolist()) == 0:')
				return [],[],[],True
		except:
			print('excpet')
			return [], [],[], True

		return historical_prices, future_prices_list, future_error_list, False


	def calculate_volatility(self, stock_price_data):
		CLOSE_PRICE_INDEX = 3
		stock_price_data_list = stock_price_data.values.tolist()
		close_prices = [float(item[CLOSE_PRICE_INDEX]) for item in stock_price_data_list]
		close_prices = [item for item in close_prices if item != 0]
		volatility = np.std(close_prices)
		return volatility
    

	def collect_data_for_all_tickers(self):
		"""
		Iterates over all symbols and collects their data
		"""

		print("Processing data for all stocks...")
		features = []
		symbol_names = []
		historical_price_info = []
		future_price_info = []
		future_error_info = []
		
		for i in tqdm(range(len(self.stocks_list))):
			symbol = self.stocks_list[i]
			try:
				stock_price_data, future_prices, future_error, not_found = self.get_data(symbol)
				if not not_found:
					volatility = self.calculate_volatility(stock_price_data)
					if volatility < self.VOLATILITY_THRESHOLD:
						continue
				#************************************	
	
				features_dictionary = self.taEngine.get_technical_indicators(self.current, symbol)
				feature_list, keys_list = self.taEngine.get_features(features_dictionary)
				# Add to dictionary
				self.features_dictionary_for_all_symbols[symbol] = {"features": features_dictionary, "current_prices": stock_price_data, "future_prices": future_prices}
				
				# Save dictionary after every 100 symbols
				if len(self.features_dictionary_for_all_symbols) % 100 == 0 and self.IS_SAVE_DICT == 1:
					np.save(self.DICT_PATH, self.features_dictionary_for_all_symbols)

				if np.isnan(feature_list).any() == True:
					continue
				# Check for volume
				average_volume_last_30_tickers = np.mean(list(stock_price_data["Volume"])[-30:])
				if average_volume_last_30_tickers < self.VOLUME_FILTER:
					continue

				# Add to lists
				features.append(feature_list)
				symbol_names.append(symbol)
				historical_price_info.append(stock_price_data)
				future_price_info.append(future_prices)
				future_error_info.append(future_error)
			except Exception as e:
				print("Exception", e)
				continue
		# Sometimes, there are some errors in feature generation or price extraction, let us remove that stuff
		features, historical_price_info, future_price_info, future_error_info, symbol_names = self.remove_bad_data(features, historical_price_info, future_price_info, future_error_info,symbol_names)
		return features, historical_price_info, future_price_info, future_error_info, symbol_names, keys_list


	def load_data_from_dictionary(self):
		"""
		Iterates over all and collects their data
		"""
		print("Loading data from dictionary")
		dictionary_data = np.load(self.DICT_PATH,allow_pickle = True).item()

		features = []
		symbol_names = []
		historical_price_info = []
		future_price_info = []
		for symbol in dictionary_data:
			feature_list, keys_list = self.taEngine.get_features(dictionary_data[symbol]["features"])
			current_prices = dictionary_data[symbol]["current_prices"]
			future_prices = dictionary_data[symbol]["future_prices"]
			
			# Check if there is any null value
			if np.isnan(feature_list).any() == True:
				continue

			features.append(feature_list)
			symbol_names.append(symbol)
			historical_price_info.append(current_prices)
			future_price_info.append(future_prices)
		# Sometimes, there are some errors in feature generation or price extraction, let us remove that stuff
		features, historical_price_info, future_price_info, future_error_info, symbol_names = self.remove_bad_data(features, historical_price_info, future_price_info, future_error_info,symbol_names)
		return features, historical_price_info, future_price_info, symbol_names, keys_list


	def remove_bad_data(self, features, historical_price_info, future_price_info, future_error_info,symbol_names):
		"""
		Remove bad data i.e data that had some errors while scraping or feature generation
		"""

		length_dictionary = collections.Counter([len(feature) for feature in features])

		length_dictionary = list(length_dictionary.keys())

		if len(length_dictionary) == 0:
			return features, historical_price_info, future_price_info,future_error_info, symbol_names

		most_common_length = length_dictionary[0]
		filtered_features, filtered_historical_price, filtered_future_prices, filtered_symbols,filtered_future_error = [], [], [], [],[]
		for i in range(0, len(features)):
			if len(features[i]) == most_common_length:
				filtered_features.append(features[i])
				filtered_symbols.append(symbol_names[i])
				filtered_historical_price.append(historical_price_info[i])
				filtered_future_prices.append(future_price_info[i])
				filtered_future_error.append(future_error_info[i])

		return filtered_features, filtered_historical_price, filtered_future_prices,filtered_future_error, filtered_symbols