# import requests
# from goose3 import Goose
# goose = Goose()
# r = requests.get('https://www.theenterpriseleader.com')
# article_urls = goose.extract(raw_html=r.content).links
# r = requests.get(article_urls[0])
# content = goose.extract(raw_html=r.content)
# publ_date = content.publish_datetime_utc
# s = r.content.decode("utf-8")
# with open('output.txt', 'w') as f:
#   f.write(s)
# import re
# time_pattern = re.compile(
#   b" ("
#   b"name=\"cXenseParse:recs:publishtime"
#   b"|property=\"rnews:datePublished"
#   b"|property=\"article:published_time"
#   b"|name=\"datePublished"
#   b")\" content=\"(.+?)\""
# )
# time_p = re.search(time_pattern, r.content)
# if time_p:
#   publ_date_origin = time_p.group(2)
from collections import OrderedDict
from os import access
import requests
import asyncio
import aiohttp

from goose3 import Goose
goose = Goose()

from market_data import title_checker
tc = title_checker()

import pandas as pd
daily_dataframes = {}
# df = {
#         "publ_time"  : [],
#         "publ_time_origin" : [],
#         "title" : [],
#         "content" : [],
#         "comp_short" : [],
#         "comp_full" : [],
#         "url" : [],
#    }
# df = pd.DataFrame.from_dict(df)
# import pyarrow as pa
# import pyarrow.parquet as pq
# schema = pa.schema({
#         "publ_time"  : pa.timestamp(),
#         "publ_time_origin" : pa.string(),
#         "title" : pa.string(),
#         "content" : pa.string(),
#         "comp_short" : pa.string(),
#         "comp_full" : pa.string(),
#         "url" : pa.string(),
#    })

import sqlite3
con = sqlite3.connect('news.db')
cur = con.cursor()

# def add_news(list_of_news):
#   df = pd.DataFrame(list_of_news, columns=[
#         "publ_time",
#         "publ_time_origin",
#         "title",
#         "content",
#         "comp_short",
#         "comp_full",
#         "url",
#   ])
#   for publ_date, x in df.groupby(df.publ_time.dt.floor('d')):
#     if publ_date in daily_dataframes.keys():
#       daily_dataframes[publ_date] = daily_dataframes[publ_date].append(x)
#     else:
#       daily_dataframes[publ_date] = x
#     daily_dataframes[publ_date].to_parquet(publ_date)
#     pass

# from collections import defaultdict
# last_access = defaultdict(dict)
# import shelve
# with shelve.open('cache') as db:
#   for url in db:
#     last_access[url] = db[url]
# sites = ['https://www.theenterpriseleader.com', 'https://www.americanbankingnews.com', 
# 'https://www.com-unik.info', 'https://www.themarketsdaily.com', 'https://www.tickerreport.com',
# 'https://www.wkrb13.com']

async def get(url, session):
  try:
    async with session.get(url=url) as response:
      r = await response.read()
      r = await r.text()
      return r
      # print("Successfully got url {} with resp of length {}.".format(url, len(resp)))
  except Exception as e:
      print("Unable to get url {} due to {}.".format(url, e.__class__))

# async def main(urls):
#   async with aiohttp.ClientSession() as session:
#     ret = await asyncio.gather(*[get(url, session) for url in urls])
#   # print("Finalized all. Return is a list of len {} outputs.".format(len(ret)))

last_access = {k:v for k, v in cur.execute('SELECT * FROM last_access')}
# for root_url, last_url in last_access.items():
#   print('processing', root_url)
#   r = requests.get(root_url)

# loop = asyncio.get_event_loop()
# loop.run_until_complete(main(last_access.keys()))

async def process_links(url, session):
  async with session.get(url) as r:
    content = await r.text()
    # r = requests.get(url)
    # try:
    #   content = goose.extract(raw_html=content)
    # except lxml.etree.ParserError:
    #   continue
    content = goose.extract(raw_html=content)
    title = content.title
    comp_short, comp_name = tc.check_title(title)
    publ_date = content.publish_datetime_utc
    # print(publ_date)
    if comp_short is not None:
      publ_date_origin = content.publish_date
      text = content.cleaned_text
      return (publ_date, publ_date_origin, title, text, comp_short, comp_name, url)

async def main():
  # ret = await asyncio.gather(*[get(url, session) for url in urls])
  async with aiohttp.ClientSession() as session:
    while True:
      for root_url, last_url in last_access.items():
        print('processing', root_url)
        r = requests.get(root_url)
        article_urls = goose.extract(raw_html=r.content).links
        article_urls = list(OrderedDict.fromkeys(article_urls))
        if last_url != None:
          try:
            article_urls = article_urls[:article_urls.index(last_url)]
          except ValueError:
            print('articles missed on', root_url) 

        # if url != last_url and last_url != None:
        #   print('articles missed on', root_url) 
        #   pass
        if len(article_urls) > 0:
          df = []
          tasks = tuple(asyncio.create_task(process_links(url, session)) for url in article_urls)
          # for url in article_urls:
          #   process_links(url)
          L = await asyncio.gather(*tasks)
          df = [l for l in L if l is not None]
          last_url = article_urls[0]
          if len(df) > 0:
            cur.executemany("insert into news values (?, ?, ?, ?, ?, ?, ?)", df)
          cur.execute('update last_access set url = ? where root_url = ?', (last_url, root_url))
          con.commit()
          last_access[root_url] = last_url
          # add_news(df)

asyncio.run(main())
  # df = pa.Table.from_pandas(df)
  # pqwriter = None
  # for i, df in enumerate(pd.read_csv('sample.csv', chunksize=chunksize)):
  #     table = pa.Table.from_pandas(df)
  #     # for the first chunk of records
  #     if i == 0:
  #         # create a parquet write object giving it an output file
  #         pqwriter = pq.ParquetWriter('sample.parquet', table.schema)            
  #     pqwriter.write_table(table)

  # close the parquet writer
  # if pqwriter:
  #     pqwriter.close()  
  # with shelve.open('cache') as db:
  #   db[root_url] = last_access[root_url]

pass