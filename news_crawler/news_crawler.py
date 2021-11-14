from collections import OrderedDict
import asyncio
import aiohttp
import lxml

from goose3 import Goose
goose = Goose()

from market_data import title_checker
tc = title_checker()

daily_dataframes = {}
import sqlite3
con = sqlite3.connect('news.db')
cur = con.cursor()

last_access = {k:v for k, v in cur.execute('SELECT * FROM last_access')}

async def process_article_url(url, session):
  async with session.get(url) as r:
    content = await r.text()
    try:
      content = goose.extract(raw_html=content)
    except lxml.etree.ParserError:
      return
    # content = goose.extract(raw_html=content)
    title = content.title
    comp_short, comp_name = tc.check_title(title)
    publ_date = content.publish_datetime_utc
    if comp_short is not None:
      publ_date_origin = content.publish_date
      text = content.cleaned_text
      return (publ_date, publ_date_origin, title, text, comp_short, comp_name, url)

async def main():
  async with aiohttp.ClientSession() as session:
    while True:
      tasks = tuple(asyncio.create_task(process_root_url(session, root, last)) for root, last in last_access.items())
      L = await asyncio.gather(*tasks)
      # for root_url, last_url in last_access.items():
      #   await process_root_url(session, root_url, last_url)

async def process_root_url(session, root_url, last_url):
    print('processing', root_url)
    async with session.get(root_url) as r:
      content = await r.text()
      article_urls = goose.extract(raw_html=content).links
      article_urls = list(OrderedDict.fromkeys(article_urls))
      if last_url != None:
        try:
          article_urls = article_urls[:article_urls.index(last_url)]
        except ValueError:
          print('articles missed on', root_url) 

      if len(article_urls) > 0:
        df = []
        tasks = tuple(asyncio.create_task(process_article_url(url, session)) for url in article_urls)
        L = await asyncio.gather(*tasks)
        df = [l for l in L if l is not None]
        last_url = article_urls[0]
        if len(df) > 0:
          cur.executemany("insert into news values (?, ?, ?, ?, ?, ?, ?)", df)
        cur.execute('update last_access set url = ? where root_url = ?', (last_url, root_url))
        con.commit()
        last_access[root_url] = last_url

asyncio.run(main())
pass