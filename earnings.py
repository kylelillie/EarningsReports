import os
import re
import sqlite3
import pymongo
import datetime
import feedparser
import pandas as pd

if not os.path.exists(os.getcwd()+'/data'):
	os.mkdir(os.getcwd()+'/data')

'''//////////////////////////////
-- articles stored here
//////////////////////////////'''
class sql:

	def save(df):

		conn = sqlite3.connect('data/articles.db')
		df.to_sql('articles', conn, if_exists='append')
		
		conn.execute('VACUUM')
		
		conn.close()

	def query(query='SELECT * FROM articles'):

		conn = sqlite3.connect('data/articles.db')
		df = pd.read_sql_query(query, conn)
		
		conn.close()
		
		return (df)
		
'''//////////////////////////////
-- article meta tags stored here
//////////////////////////////'''
class mongo:

	def save():

		client = pymongo.MongoClient()
		db = client.article_tags
		
		post_data = {
			'entities':'',
			'category':'',
			'other':''
			}

		result = post.insert_one(post_data)

	def query():
		
		find = posts.find_one({
			'author':'Rando'
			})

		print (find)

today = datetime.datetime.now().strftime('%Y-%m-%d')

'''//////////////////////////////
-- list of feeds to scrape
//////////////////////////////'''
rss = ['https://www.globenewswire.com/RssFeed/subjectcode/13-Earnings%20Releases%20And%20Operating%20Results/feedTitle/GlobeNewswire%20-%20Earnings%20Releases%20And%20Operating%20Results'
	,'https://www.yahoo.com/news/rss/'
	,'http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/front_page/rss.xml'
	,'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en'
	,'http://feeds.reuters.com/reuters/businessNews'
	,'http://feeds.reuters.com/reuters/companyNews'
	,'http://feeds.reuters.com/Reuters/worldNews'
	,'https://rss.cbc.ca/lineup/canada-calgary.xml'
	,'https://rss.cbc.ca/lineup/canada-edmonton.xml'
	,'https://rss.cbc.ca/lineup/business.xml'
	,'https://rss.cbc.ca/lineup/politics.xml'
	,'https://www.theglobeandmail.com/?service=rss'
	,'http://feeds.feedburner.com/FP_TopStories'
	,'https://o.canada.com/feed/'
	,'https://www.dailyheraldtribune.com/feed'
	,'https://www.bankofcanada.ca/valet/fx_rss/FXUSDCAD'
	,'https://www.bankofcanada.ca/content_type/bos/feed/'
	,'https://www.jasper-alberta.com/RSSFeed.aspx?ModID=76&CID=All-0'
	,'https://www.highriveronline.com/rss/news'
	,'https://www.highriveronline.com/rss/ag-news'
	,'https://www.highlevel.ca/RSSFeed.aspx?ModID=76&CID=All-0'
	,'https://www.cochrane.ca/RSSFeed.aspx?ModID=76&CID=All-0'
	,'http://banff.ca/support/pages.xml'
	,'https://www.wetaskiwin.ca/RSSFeed.aspx?ModID=76&CID=All-0'
	,'https://mdbighorn.ca/RSSFeed.aspx?ModID=76&CID=All-0'
	,'https://www.brooks.ca/RSSFeed.aspx?ModID=76&CID=All-0'
	,'https://discoverairdrie.com/rss/news'
	,'https://lacombeonline.com/rss/news'
	,'https://www.mdtaber.ab.ca/RSSFeed.aspx?ModID=76&CID=All-0'
	,'https://www.nanton.ca/RSSFeed.aspx?ModID=76&CID=All-0'
	]

feeds = [] # list of feed objects
posts = []

for url in rss:
    feed = feedparser.parse(url)
    for post in feed.entries:
        posts.append((today,post.title, post.link, post.summary))

df = pd.DataFrame(posts, columns=['date','title', 'link', 'summary']) # pass data to init
df.summary = df.summary.replace(r'<[^>]*>','',regex=True)
df = df.drop_duplicates(['summary'])

sql.save(df)
print (sql.query())

'''
bonus m-lab downloads query!

SELECT
count(*), connection_spec.client.network.asn,
avg(8 * (web100_log_entry.snap.HCThruOctetsAcked /
    (web100_log_entry.snap.SndLimTimeRwin +
    web100_log_entry.snap.SndLimTimeCwnd +
    web100_log_entry.snap.SndLimTimeSnd))) AS download_Mbps,
    connection_spec.client_geolocation.city,
    connection_spec.client_geolocation.region,
    avg(connection_spec.client_geolocation.latitude) AS latitude,
    avg(connection_spec.client_geolocation.longitude) AS longitude
FROM `measurement-lab.ndt.web100`
WHERE connection_spec.client_geolocation.country_name='United States' AND  log_time > "2019-01-01" AND connection_spec.client_geolocation.city IS NOT NULL
AND ((web100_log_entry.snap.SndLimTimeRwin +
    web100_log_entry.snap.SndLimTimeCwnd +
    web100_log_entry.snap.SndLimTimeSnd) >0)
AND web100_log_entry.snap.HCThruOctetsAcked >= 0
group by      connection_spec.client.network.asn,connection_spec.client_geolocation.city,
    connection_spec.client_geolocation.region order by connection_spec.client_geolocation.region,connection_spec.client_geolocation.city
'''