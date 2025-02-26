import pandas as pd
import yfinance as yf
import statsmodels.api as sm

file_path = 'Data/Russell2000_total.csv'
data = pd.read_csv(file_path, header=[0, 1], index_col=0)
russell2000_etf = yf.download("IWM", start="2014-01-02", end="2024-11-20")
russell2000_etf['Return'] = russell2000_etf['Adj Close'].pct_change()
iwm = russell2000_etf['Return'].dropna()
adj_close_data = data.xs('Adj Close', axis=1, level=1)
returns = adj_close_data.pct_change()
returns = returns.iloc[2:, :]
error_terms = []
window_size = 240

for ticker in returns.columns:
    aligned_stock = returns[ticker].dropna()
    aligned_iwm = iwm.loc[aligned_stock.index]

    aligned_data = pd.concat([aligned_stock, aligned_iwm], axis=1, join="inner").dropna()
    aligned_data.columns = ['Stock', 'IWM']

    if len(aligned_data) >= window_size:
        for start in range(0, len(aligned_data), window_size): 
            if start + window_size > len(aligned_data):
                break
    
            window_data = aligned_data.iloc[start:start + window_size]
            end_date = window_data.index[-1] 

            X = sm.add_constant(window_data['IWM'])
            y = window_data['Stock']
            model = sm.OLS(y, X).fit()

            residuals = y - model.predict(X)

            for date, error in residuals.items():
                error_terms.append({
                    'time': date,
                    'ticker': ticker,
                    'error': error
                })

error_terms_df = pd.DataFrame(error_terms)
error_terms_df.rename(columns = {'time':'Date','ticker':'Ticker','error':'Error'},inplace=True)
pivoted_data = error_terms_df.pivot(index='Date', columns='Ticker', values='Error')
pivoted_data.to_csv("Data/Russell2000_error.csv", index=False)
