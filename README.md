# Bloomberg Capstone -- Stock Return Anomaly Detection using Machine Learning  

## **Project Overview**
This project is a **collaboration between Bloomberg and NYU** to develop a machine learning-based model for detecting **stock price anomalies in the U.S. equity market.** By leveraging **time-series forecasting techniques and cross-sectional anomaly detection models**, we aim to identify irregularities in stock returns that could indicate market inefficiencies, trading opportunities, or underlying risks.

Our approach integrates two complementary machine learning models:
1. **Hybrid LSTM-GARCH Model** – A time-series-based model that captures stock return anomalies by predicting expected returns and estimating volatility.
2. **Cross-Sectional Isolation Forest Model** – An unsupervised learning model that identifies anomalies among stocks by analyzing liquidity, momentum, and volatility features.

This research is intended to **enhance financial risk assessment** and **improve market efficiency analysis**, supporting traders, investors, and portfolio managers in making data-driven decisions.

### **1. Cross-Sectional Isolation Forest Model(Multiple Stocks)**
**Isolation Forest model** examines **multiple stocks at the same time** to detect outliers across the stock market.

#### **Steps:**
1. **Feature Engineering**:
   - 22 features extracted from stock price data, including:
     - **Momentum indicators** (RSI, Stochastic Oscillator)
     - **Liquidity metrics** (Ease of Movement, Accumulation/Distribution Index)
     - **Volatility metrics** (Daily return variance)
2. **Isolation Forest Algorithm**:
   - Builds multiple **random decision trees** to isolate anomalies.
   - Stocks that require **fewer splits** to isolate are considered anomalies.
3. **Evaluation**:
   - Anomalies are assessed using **future price deviations** and **CAPM residual errors**.
#### **Visulization Example**:
<img width="636" alt="image" src="https://github.com/user-attachments/assets/9dc0a97e-825a-4abf-a214-2701c2e68fbe" />


## **Methodologies**
### **2. Hybrid LSTM-GARCH Model(Single Asset)**
Unlike the Isolation Forest model, which focuses on detecting stocks with anomaly based on the stock market overall performance. LSTM-GARCH model focus on detecting anoamly time spots in its' historical data for single stock.
The **LSTM-GARCH model** combines two powerful methods:
- **LSTM (Long Short-Term Memory)**: Predicts the expected return of individual stocks based on historical price movements.
- **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**: Estimates stock return volatility, accounting for time-varying risk.

This approach captures **unusual stock price movements** while filtering out normal market fluctuations.

#### **Visulization Example**:
<img width="636" alt="image" src="https://github.com/user-attachments/assets/69586561-b2f5-4aea-8811-5848495a7ea5" />

## **Folder Structure**
Stock_Anomaly_Detection
│── README.md                   # Project documentation
│── Capstone_final_paper.pdf     # Research paper detailing methodology and findings
│── data/                        # Dataset (Russell 2000 stock data)
│   │── Russell2000_total.csv    # Stock price data for Russell 2000 index
│   │── Russell2000_error.csv    # CAPM residual error terms
│   │── feature_data/            # Processed feature files for each stock
│── models/                      # Trained machine learning models
│── feature_calculation.py       # Extracts financial features from stock price data
│── feature_generator.py         # Computes technical indicators for anomaly detection
│── detection_engine.py          # Implements the Isolation Forest anomaly detection model
│── CAPM.py                      # Calculates residual errors using the Capital Asset Pricing Model (CAPM)
│── data_loader.py               # Loads and preprocesses stock data
│── result_analysis.py           # Evaluates anomaly detection results and generates reports
│── outputs/                     # Model predictions and evaluation reports
│── requirements.txt             # List of dependencies needed to run the project

## **Installation and Dependencies**

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/Stock_Anomaly_Detection.git
cd Stock_Anomaly_Detection
```

### **2. Create a Virtual Environment**

```bash
python -m venv env
source env/bin/activate   # Mac/Linux
env\Scripts\activate      # Windows
```

### **3. Install Required Libraries**

```bash
pip install -r requirements.txt
```

The project requires:

- **Data processing**: `numpy`, `pandas`, `scipy`, `statsmodels`
- **Machine Learning**: `scikit-learn`, `tensorflow`
- **Visualization**: `matplotlib`, `seaborn`

## **Usage**

### **1. Feature Extraction**

Run the script to preprocess raw stock data and compute technical indicators:

```bash
python feature_calculation.py
```

### **2. Model Training and Anomaly Detection**

Train the LSTM-GARCH and Isolation Forest models on Russell 2000 data:

```bash
python feature_generator.py
```

### **3. Analyzing Results**

Detect anomalies and evaluate performance:

```bash
python result_analysis.py
```

Results will be saved in the `outputs/` directory.


## **Acknowledgments**

This project was conducted in collaboration with **Bloomberg and NYU MFE**. Special thanks to **Bloomberg Quantitative Research Group** for providing guidance and data access via the **Bloomberg Terminal** under academic licensing.


