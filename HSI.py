import yfinance as yf
import time
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import json
import requests  # 將缺少的 requests 库导入

# 獲取香港股票數據
def get_hk_stock_data():
    stock_symbol = "^HSI"

    end_date = datetime.today()
    start_date = datetime(1970, 1, 1)

    hk_stock_data = yf.download(stock_symbol, start=start_date, end=end_date, interval="1d")
    hk_stock_data.dropna(inplace=True)
    return hk_stock_data

# 創建漲跌預測標籤欄位
def create_prediction_column(stock_data):
    stock_data["Next_Day_Close"] = stock_data["Close"].shift(-1)
    stock_data["Trend_Label"] = np.where(stock_data["Next_Day_Close"] > stock_data["Close"], "上漲", "下跌")
    return stock_data

# 添加技術指標
def add_technical_indicators(stock_data):
    stock_data["Daily_Return"] = stock_data["Adj Close"].pct_change()

    short_window = 10
    long_window = 50
    stock_data["Short_MA"] = stock_data["Adj Close"].rolling(window=short_window, min_periods=1, center=False).mean()
    stock_data["Long_MA"] = stock_data["Adj Close"].rolling(window=long_window, min_periods=1, center=False).mean()

    n = 14
    delta = stock_data["Adj Close"].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = loss.rolling(window=n, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    stock_data["RSI"] = rsi

    short_ema = stock_data["Adj Close"].ewm(span=12, adjust=False).mean()
    long_ema = stock_data["Adj Close"].ewm(span=26, adjust=False).mean()
    stock_data["MACD"] = short_ema - long_ema
    stock_data = create_prediction_column(stock_data)

    return stock_data

# 寫入數據至 Google Sheets
def write_to_google_sheet():
    hk_stock_data = get_hk_stock_data()
    if hk_stock_data.empty:
        print("無效的數據")
        return

    # 添加技術指標
    hk_stock_data = add_technical_indicators(hk_stock_data)

    # 排序DataFrame，讓今天的數據成為第一行
    hk_stock_data.sort_index(ascending=False, inplace=True)

    # 將NaN填充為0
    hk_stock_data.fillna(0, inplace=True)

    data_to_write = []
    for date, row in hk_stock_data.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        open_price = round(row["Open"], 2)
        high = round(row["High"], 2)
        low = round(row["Low"], 2)
        close = round(row["Close"], 2)
        adj_close = round(row["Adj Close"], 2)
        volume = round(row["Volume"], 2)
        daily_return = round(row["Daily_Return"], 4)
        short_ma = round(row["Short_MA"], 2)
        long_ma = round(row["Long_MA"], 2)
        rsi = round(row["RSI"], 2)
        macd = round(row["MACD"], 2)
        data_to_write.append([
            date_str, str(open_price), str(high), str(low), str(close), str(adj_close), str(volume), str(daily_return),
            str(short_ma), str(long_ma), str(rsi), str(macd)
        ])

    # 下載 JSON 憑證檔案
    json_url = "https://drive.google.com/uc?id=1QVaUpQn_PmKXjQIgkJD2L1XwMsb-6NRF"
    response = requests.get(json_url)
    json_content = response.content

    # 將 JSON 內容轉換為字典
    json_data = json.loads(json_content)

    # 授權使用 Google Sheets API
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json_data, scope)
    client = gspread.authorize(creds)

    sheet_name = "股市資料"
    sheet = client.open("股市資料").worksheet("大盤")

    # 寫入標題
    header = [
        "日期", "開盤價", "最高價", "最低價", "收盤價", "調整收盤價", "成交量", "漲跌幅度",
        "短期移動平均", "長期移動平均", "相對強弱指數(RSI)", "指數平滑異同移動平均線(MACD)"
    ]

    # 寫入標題
    sheet.clear()
    sheet.append_row(header)

    # 寫入數據
    sheet.append_rows(data_to_write)

if __name__ == "__main__":
    write_to_google_sheet()














































