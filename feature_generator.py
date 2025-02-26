# Basic libraries
import numpy as np
from os import walk
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class TAEngine:
	#************************************
	def __init__(self, history_to_use, feature_dataframes):
		print("Technical Indicator Engine has been initialized")
		self.HISTORY_TO_USE = history_to_use
		self.feature_dataframes = feature_dataframes

	#************************************
	def get_technical_indicators(self, current,symbol):
		"""
		Given a pandas data frame with columns -> 'Open', 'High', 'Low', 'Close', 'Volume', extract different technical indicators and returns 
		"""
		technical_indicators_dictionary = {}
		for feature in self.feature_dataframes.keys():
			technical_indicators_dictionary[feature] = self.feature_dataframes[feature].loc[current, symbol]
		return technical_indicators_dictionary

	def get_features(self, features_dictionary):
		"""
		Extract features from the data dictionary. The data dictionary contains values for multiple TAs such as cci, rsi, stocks etc. But here, we will only use the price returns, volume returns, and eom values.
		"""

		all_keys = list(sorted(features_dictionary.keys()))

		filtered_keys = [
			'rsi_5_slope', 'rsi_10_p_value',
			'stoch_10_slope', 'stoch_10_p_value', 'stoch_15_r_value', 'eom_5', 'eom_5_slope', 'eom_5_r_value', 'eom_10_r_value',
			'eom_10', 'eom_10_slope', 'eom_20', 'eom_20_slope', 'cci_5', 'cci_5_slope',
			'cci_10_r_value', 'cci_20_slope', 'cci_20_r_value', 'daily_return', 'volume_returns',
			'volume_returns_slope', 'volume_returns_r_value'
		]
		feature_list = []
		# Loop through all keys and match with filtered_keys
		for key in all_keys:
			if key in filtered_keys:
				feature_list.append(features_dictionary[key]) 
			else:
				_ = None
		return feature_list, filtered_keys
