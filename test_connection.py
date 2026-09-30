import MetaTrader5 as mt5

LOGIN = 313416886
PASSWORD = "your_brightfunded_mt5_password"
SERVER = "BrightFunded-Server"

# ← Use the NEW path from Step 2:
TERMINAL_PATH = r"C:\Program Files\BrightFunded MT5\terminal64.exe"

if not mt5.initialize(path=TERMINAL_PATH, login=LOGIN, password=PASSWORD, server=SERVER):
    print("❌ initialize() failed, error code =", mt5.last_error())
    quit()

info = mt5.account_info()
if info:
    print("✅ SUCCESS")
    print(f"Broker: {info.company}")
    print(f"Balance: {info.balance}")
    print(f"Server: {info.server}")
    print(f"Login: {info.login}")

mt5.shutdown()