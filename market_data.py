import os
import pandas as pd
from datetime import datetime, timedelta, timezone
class market_data():
  def __init__(
    self, 
    path_to_market_data="../data/market_data.xlsx", 
    path_to_sp500="../sp500.xlsx", 
    shift=-3
    ):
    # path_to_market_data = "../data/market_data.xlsx"
    parquet_file_name = os.path.dirname(os.path.abspath(path_to_market_data)) + '/market_data.parquet'
    if not os.path.isfile(parquet_file_name):
      if not os.path.isfile(path_to_market_data):
        # path_to_sp500 = "../sp500.xlsx"
        df = pd.read_excel(path_to_sp500, engine='openpyxl')
        all_comps = []
        for idx in df.Symbol:
          all_comps += [idx]
        all_comps = " ".join(all_comps)
        import yfinance as yf
        df =  yf.download(all_comps, start="2016-08-01")["Adj Close"]
        df.to_excel(path_to_market_data, sheet_name="Sheet1", engine='openpyxl')
      else:
        df = pd.read_excel(path_to_market_data, engine='openpyxl')
      df = df[pd.notnull(df["A"])]
      df.Date = df.Date.dt.date
      df.set_index('Date', inplace=True)
    else:
      df = pd.read_parquet(parquet_file_name)

    import pandas_datareader.data as web

    start = df.index[0]
    end = df.index[-1]

    SP500 = web.DataReader(['sp500'], 'fred', start, end)
    # SP500 = web.DataReader(['^GSPC'], 'yahoo', start, end).Close
    merged = pd.merge(df, SP500, left_index=True, right_index=True)
    df = df.iloc[:,:-1].div(merged['sp500'], axis=0)

    a = df.shift(shift)
    self.returns = a / df

    vals = self.returns.values.tolist()
    inds = self.returns.index.tolist()
    vals_full = []
    inds_full = []
    for i1, i2, r1 in zip(inds[:-1], inds[1:], vals[:-1]):
      diff = (i2 - i1).days
      dates = [i1 + timedelta(days=d) for d in range(diff)]
      r1 = [r1] * diff
      # r1 = [[d] + r for d, r in zip(dates, r1)]
      vals_full += r1
      inds_full += dates
    # lol_full_prev += r2
    # df_copy = self.returns.iloc[0:0,:].copy()
    # df_copy[inds_full] = vals_full
    # c = list(self.returns)
    # d = [len(r) for r in vals_full]
    self.returns = pd.DataFrame(vals_full, columns=list(self.returns), index=inds_full)
    pass

  # def get_adj_close(self, ticket, dt):
  #   dt -= timedelta(hours=16)
  #   return list(self.df[ticket][self.df['Date'] <= dt])[-1]

  # def get_return(self, ticket, dt):
  #   dt1 = dt - timedelta(hours=16)
  #   dt2 = dt1 + timedelta(days=3)
  #   df = self.df.filter(items=["Date", ticket])
  #   q1 = df[df.Date <= dt1][ticket].iloc[-1]
  #   # a = self.df[(df.publ_date > '2014-07-23 07:30:00') & (df.publ_date < '2014-07-23 09:00:00')]
  #   # q1 = list(self.df[ticket][self.df['Date'] <= dt1])[-1]
  #   # df1 = self.df[[ticket, 'Date']][self.df['Date'] > dt1]
  #   df1 = df[(df.Date > dt1) & (df.Date <= dt2)][ticket]
  #   # df2 = df1[df1.Date <= dt2]
  #   # df2 = list(df1[ticket][df1['Date'] <= dt2])
  #   if df1.shape[0] > 0:
  #     q2 = df1.iloc[-1]
  #   else:
  #     df1 = df[df.Date > dt1][ticket]
  #     if df1.shape[0] > 0:
  #       q2 = df1.iloc[0]
  #     else:
  #       return 1.
  #   return q2 / q1

  # def get_return(self, ticket, date):
  #   return self.get_adj_close(ticket, date + timedelta(days=2)) / self.get_adj_close(ticket, date - timedelta(days=2))

