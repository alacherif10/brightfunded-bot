import yfinance as yf

data = yf.download("EURUSD=X", period="5d", interval="1h", progress=False)

if data is None or data.empty:
    print("❌ No data returned. Check internet connection.")
else:
    print("✅ Data fetched successfully")
    print(f"Total candles: {len(data)}")
    print("\nLast 5 candles:")
    print(data.tail())