import requests
import feedparser
import pandas as pd

rss = ['https://www.globenewswire.com/RssFeed/subjectcode/13-Earnings%20Releases%20And%20Operating%20Results/feedTitle/GlobeNewswire%20-%20Earnings%20Releases%20And%20Operating%20Results'
	,'https://www.yahoo.com/news/rss/'
	,'http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/front_page/rss.xml'
	]

feeds = [] # list of feed objects
posts = []

for url in rss:
    feed = feedparser.parse(url)
    for post in feed.entries:
        posts.append((post.title, post.link, post.summary))

df = pd.DataFrame(posts, columns=['title', 'link', 'summary']) # pass data to init

print (df.head())
print (df.summary[0])