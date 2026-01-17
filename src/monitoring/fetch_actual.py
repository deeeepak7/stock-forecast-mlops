import yfinance as yf
from datetime import date,timedelta
from src.monitoring.actual_updater import update_actual


def fetch_latest_close(
        ticker:str
)->tuple[float,date]:
    ticker_obj=yf.Ticker(ticker)
    hist=ticker_obj.history(period="7d")
    if hist.empty:
        raise RuntimeError("no price data return")
    last_row=hist.iloc[-1]
    close_price=float(last_row["Close"])
    close_date=last_row.name.date()
    return close_price,close_date

def main():
    TICKER="AAPL"
    price,close_date=fetch_latest_close(TICKER)
    update_actual(
        actual_price=price,
        actual_date=close_date
    )
    print(f"[OK] Updated actuals for {TICKER} | "
        f"date={close_date} | price={price}")
    
if __name__=="__main__":
    main()    
