import os
import re
import sqlite3
import pymongo
import datetime
import feedparser
import pandas as pd

import spacy
from tqdm import tqdm
import time
import pickle

import tensorflow_hub as hub
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

if not os.path.exists(os.getcwd()+'/data'):
	os.mkdir(os.getcwd()+'/data')


class sql:
	'''//////////////////////////////
	-- articles stored here --
	//////////////////////////////'''
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

class mongo:
	'''//////////////////////////////
	-- article meta tags stored here --
	//////////////////////////////'''
	def save(post_data={}):

		client = pymongo.MongoClient()
		db = client.article_tags
		
		post_data = {
			'entities':'',
			'category':'',
			'other':''
			}

		result = post.insert_one(post_data)

	def query(find={}):
		
		find = posts.find_one({
			'author':'Rando'
			})

		print (find)

class classify:

	def clean(df):
		# remove punctuation marks
		punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

		df['clean'] = df['summary'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

		# convert text to lowercase
		df.clean = df.clean.str.lower()

		# remove numbers
		#df['clean'] = df.clean.str.replace("[0-9]", " ")

		# remove whitespaces
		df.clean = df.clean.replace('&nbsp;',' ')
		df.clean = df.clean.apply(lambda x:' '.join(x.split()))
		
		def lemmatize(text):
		
			#import spaCy's language model
			#python -m spacy download en
			nlp = spacy.load('en', disable=['parser', 'ner'])
				
			output = []
		
			for i in text:
				s = [token.lemma_ for token in nlp(i)]
				output.append(' '.join(s))

			print (output[:5])
			return (output)
			
		df['clean'] = lemmatize(df['clean'])
		print (df['clean'].head())
		return (df)

	def elmo_vectors(x):
	
		elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
		embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
		
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.tables_initializer())
			# return average of ELMo features
			return sess.run(tf.reduce_mean(embeddings,1))
	
	def article(text=''):
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
		'https://allennlp.org/models'
		
		#following this:
		#https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/
		
		train = classify.clean(sql.query('SELECT * FROM articles LIMIT 400'))
		test = classify.clean(sql.query('SELECT * FROM articles ORDER BY title DESC LIMIT 300'))
				
		print (train.shape)
		print (test.shape)
		
		print (train.sample(15))
		
		train['summary'].value_counts(normalize = True)
		
		list_train = [train[i:i+100] for i in range(0,train.shape[0],100)]
		list_test = [test[i:i+100] for i in range(0,test.shape[0],100)]
		
		
		
		# Extract ELMo embeddings
		elmo_train = [classify.elmo_vectors(x['clean']) for x in list_train]
		elmo_test = [classify.elmo_vectors(x['clean']) for x in list_test]

		elmo_train_new = np.concatenate(elmo_train, axis = 0)
		elmo_test_new = np.concatenate(elmo_test, axis = 0)
		
		# save elmo_train_new
		pickle_out = open("elmo_train_03032019.pickle","wb")
		pickle.dump(elmo_train_new, pickle_out)
		pickle_out.close()

		# save elmo_test_new
		pickle_out = open("elmo_test_03032019.pickle","wb")
		pickle.dump(elmo_test_new, pickle_out)
		pickle_out.close()
		
		# load elmo_train_new
		pickle_in = open("elmo_train_03032019.pickle", "rb")
		elmo_train_new = pickle.load(pickle_in)

		# load elmo_train_new
		pickle_in = open("elmo_test_03032019.pickle", "rb")
		elmo_test_new = pickle.load(pickle_in)
		
		'''
		spaCy or something has a gpu import error
		'''
		pass
		
def fetch_articles():

	today = datetime.datetime.now().strftime('%Y-%m-%d')

	'''//////////////////////////////
	-- list of feeds to scrape --
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
	#print (sql.query())

#fetch_articles()
classify.article()

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