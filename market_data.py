import os
import pandas as pd
from datetime import datetime, timedelta, timezone

# import yfinance as yf
# df_yfinance =  yf.download('AFL', start="2021-10-10")['Close']
# print(df_yfinance)

def load_sp500():
  df_sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
  df_sp500.Symbol = df_sp500.Symbol.str.replace('.', '-', regex=False)
  df_sp500['Issuer'] = df_sp500.Security.str.replace('\s*\((Series|Class).*\)', '', regex=True)
  df_sp500['Issuer_html'] = df_sp500.Issuer.str.replace('&', '&amp;')
  return df_sp500

def update_market_data(path2market_data):
  # df_old = pd.read_excel(path, engine='openpyxl')
  import shutil
  # shutil.copy(path2market_data, path2market_data + '.backup')
  # df = pd.read_parquet(path2market_data)
  # start_date = str(
  #     (max(df.index) + timedelta(days=1)).date()
  #     )
  # start_date = "2016-08-01"
  from dateutil.relativedelta import relativedelta
  start_date = str(
    (datetime.today() - relativedelta(days=40)).date()
    )
  start_date = '2021-06-01'
  new_df = download_all_since(start_date)
  new_df = append_SP500(new_df)
  # df_yfinance.to_parquet(path2market_data)
  # merged = df.append(new_df)
  # merged.to_excel('md.xlsx', sheet_name="Sheet1", engine='openpyxl')
  new_df.to_parquet(path2market_data)
  check_null_values(new_df)
  pass

def check_null_values(df):
  wrong_cols = df.columns[df.isna().any()].tolist()
  if len(wrong_cols) == 0:
    print('no null values found')
    return True, wrong_cols
  else:
    print(len(wrong_cols), 'columns with null values found!!!')
    return False, wrong_cols

def fix_errors(path2market_data):
  df = pd.read_parquet(path2market_data)
  res, wrong_cols = check_null_values(df)
  if res:
    return
  all_comps = " ".join(wrong_cols)
  import yfinance as yf
  start_date = str((min(df.index)).date())
  df_yfinance =  yf.download(all_comps, start=start_date)["Adj Close"]
  df = df.combine_first(df_yfinance.loc[df.index])
  df.to_parquet(path2market_data)
  check_null_values(df)
  pass
  # df = df[pd.notnull(df["A"])]

def append_SP500(df):
  import pandas_datareader.data as web
  start = df.index[0]
  end = df.index[-1]
  SP500 = web.DataReader(['sp500'], 'fred', start, end)
  # SP500 = web.DataReader(['^GSPC'], 'yahoo', start, end).Close
  return pd.merge(df, SP500, left_index=True, right_index=True)

def download_all_since(start_date):
    all_comps = []
    df_sp500 = load_sp500()
    for idx in df_sp500.Symbol:
      all_comps += [idx]
    all_comps = " ".join(all_comps)
    import yfinance as yf
    df_yfinance =  yf.download(all_comps, start=start_date)["Adj Close"]
    return df_yfinance

def load_market_data(parquet_file_name, path_to_market_data):
  if not os.path.isfile(parquet_file_name):
    print('market_data.parquet not found. Creating...')
    if not os.path.isfile(path_to_market_data):
      print('market_data.xlsx not found. Creating...')
      # df = pd.read_excel(path_to_sp500, engine='openpyxl')
      # all_comps = []
      # for idx in df.Symbol:
      #   all_comps += [idx]
      # all_comps = " ".join(all_comps)
      # import yfinance as yf
      # df =  yf.download(all_comps, start="2016-08-01")["Adj Close"]
      df = download_all_since("2016-08-01")
      df.to_excel(path_to_market_data, sheet_name="Sheet1", engine='openpyxl')
    else:
      df = pd.read_excel(path_to_market_data, engine='openpyxl')
    df = df[pd.notnull(df["A"])]
    df.Date = df.Date.dt.date
    df.set_index('Date', inplace=True)
    # import pandas_datareader.data as web
    # start = df.index[0]
    # end = df.index[-1]
    # SP500 = web.DataReader(['sp500'], 'fred', start, end)
    # df = pd.merge(df, SP500, left_index=True, right_index=True)
    df = append_SP500(df)
    df.to_parquet(parquet_file_name, compression='gzip')
  else:
    df = pd.read_parquet(parquet_file_name)
  return df

def update_sp500(path_sp500):
  import pandas_datareader.data as web
  # path_sp500 = '../data/sp_500.xlsx'
  df_old = pd.read_excel(path_sp500, engine='openpyxl')
  df_old.set_index('DATE', inplace=True)
  start = df_old.index[-1] + timedelta(days=1)
  df = web.DataReader(['sp500'], 'fred', start)
  df = df[df['sp500'].notnull()]
  df = df_old.append(df)
  # df.to_parquet('../data/sp500.parquet')
  df.to_excel(path_sp500, sheet_name="Sheet1", engine='openpyxl')

class market_data():
  def __init__(
    self, 
    path_to_market_data="../data/market_data.xlsx", 
    #path_to_sp500="../sp500.xlsx", 
    shift=1
    ):
    # path_to_market_data = "../data/market_data.xlsx"
    parquet_file_name = os.path.dirname(os.path.abspath(path_to_market_data)) + '/market_data.parquet'
    df = load_market_data(parquet_file_name, path_to_market_data)

    # df.to_excel('df_with_sp500.xlsx', sheet_name="Sheet1", engine='openpyxl')
    df = df.iloc[:,:-1].div(df['sp500'], axis=0)

    a = df.shift(shift)
    self.returns = df / a # current date divided by the prior date
    mn = self.returns.mean(axis=1)
    self.returns = self.returns.subtract(mn, axis='rows')

    vals = self.returns.values.tolist()
    inds = self.returns.index.tolist()
    vals_full = []
    inds_full = []
    for i1, i2, r1 in zip(inds[1:], inds[:-1], vals[1:]):
      diff = (i1 - i2).days
      dates = [i1 + timedelta(days=d - diff + 1) for d in range(diff)]
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

if __name__ == '__main__':
  # market_data()
  # update_market_data('../data/market_data.parquet')
  # fix_errors('../data/market_data.parquet')
  update_sp500('../data/sp_500.xlsx')
