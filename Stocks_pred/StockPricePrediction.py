'''
Created on 03.02.2014

@author: Michael Kamp
'''

import collections as cl
import re
import numpy as np
from sklearn import svm
from datetime import datetime
import pickle
import os
import sklearn.decomposition as deco
import matplotlib.pyplot as plt
import math

RDSAM = "rdsam_2004_2012.fdc"
SP100 = "sp100_2004_2012.fdc"
TDIDF = "tfidf"

class StockPricePrediction:
	def __init__(self, stockFile, priceDiffThreshold = 0.05, warmUp = 50):
		self.stock_file = stockFile
		self.price_diff_t = priceDiffThreshold
		self.warm_up = 50
		self.preprocess()

	def preprocess(self):
		f = open(self.stock_file, 'r')
		lines = f.readlines()
		f.close()

		print "Reading prices...",
		self.prices = cl.defaultdict(list)
		self.dates = []
		self.stocks = lines[0].replace("\n","").split("\t")[1:]
		for line in lines[1:]:
			vals = line.replace("\n","").split("\t")
			date = datetime.strptime(vals[0], "%d.%m.%Y")
			if date <= datetime.strptime("01.01.2011","%d.%m.%Y"):
				continue
			self.dates.append(date)
			for i in xrange(1,len(vals)):
				if vals[i] != "NaN" and vals[i] != '0.0':
					self.prices[self.stocks[i-1]].append(float(vals[i]))
				else:
					if len(self.prices[self.stocks[i-1]]) > 0:
						self.prices[self.stocks[i-1]].append(self.prices[self.stocks[i-1]][-1])
					else:
						self.prices[self.stocks[i-1]].append(0.0)
		print "done." 
		'''p = ''
		for i in xrange(len(self.dates)):
			for stock in self.stocks:
				p += str(stock) + '\t' + str(self.dates[i]) + '\t' + str(self.prices[stock][i]) + '\n'
		f = open('prices2.csv','w')
		f.write(p)'''
		print "Preprocessing labels..."               
		self.labels = cl.defaultdict(list)
		self.classBalance = {1:0.0,-1:0.0,0:0.0}
		self.classBalancePerSock = {}
		for stock in self.stocks:
			self.classBalancePerSock[stock] = {1:0.0,-1:0.0,0:0.0}
			self.labels[stock].append(0) #first one is 0, because no price is known before
			for i in xrange(1,len(self.prices[stock])):
				label = 0
				if self.prices[stock][i-1] == 0.0:
					if self.prices[stock][i] > 0.0:
						label = 1 
						self.classBalance[1] += 1.0
						self.classBalancePerSock[stock][1] += 1.0
					elif self.prices[stock][i] < 0.0:
						label = -1
						self.classBalance[-1] += 1.0
						self.classBalancePerSock[stock][-1] += 1.0
					else:
						self.classBalance[0] += 1.0
						self.classBalancePerSock[stock][0] += 1.0
				else:
					if (self.prices[stock][i] - self.prices[stock][i-1]) / self.prices[stock][i-1] >= self.price_diff_t:
						label = 1 
						self.classBalance[1] += 1.0
						self.classBalancePerSock[stock][1] += 1.0                   
					elif (self.prices[stock][i] - self.prices[stock][i-1]) / self.prices[stock][i-1] <= -1*self.price_diff_t:
						label = -1
						self.classBalance[-1] += 1.0
						self.classBalancePerSock[stock][-1] += 1.0
					else:
						self.classBalance[0] += 1.0
						self.classBalancePerSock[stock][0] += 1.0
				self.labels[stock].append(label)
		print self.classBalance
		print "done."
		'''print "Reading features...",
		values = {}
		try:
			with open(TDIDF) as f:
				for l in f:
					l = re.sub(r'[{}:,\n\']','',l)
					line = re.split('\ +',l)
					word = line[0]
					if not values.has_key(word):
						values[word] = {}
					i = 1
					while (i < len(line)-1):
						try:
							date = datetime.strptime(line[i], "%Y-%m-%d")
							tdidf = line[i+1]
							values[word][date] = float(tdidf)
						except:
							k = 1
							#print line[i] + " is not a proper date."
						i = i + 2
				f.close()
		except IOError:
			print "\n File " + TDIDF + " cannot be found!"
		print "done."
		print "Aligning features...",
		self.features = []
		print self.dates
		for date in self.dates:
			self.features.append([])
			for word in values:
				if date not in values[word]:
					print "\n Date "+date+" missing in features."
					self.features[-1].append(self.features[-1][-1]) #take last features
				self.features[-1].append(values[word][date])
		#print self.features'''
		'''data = ''
		for stock in self.stocks:
			data += str(stock) + str(self.labels[stock]) + str(len(self.labels[stock])) + '\n'
		out = open('labels.csv','w')
		out.write(data)
		data = ''
		for i in xrange(len(self.dates)-1):
			data += str(self.dates[i]) + str(self.features[i]) + str(len(self.features[i])) + '\n'
		out = open('features.csv','w')
		out.write(data)
		print "done."'''
    
	def pca(self, n_components):
		print "Reducing the dimensionality to " + str(n_components)
		x = np.array(self.features)
		pca = deco.PCA(n_components)
		self.reduced_features = pca.fit(x).transform(x)
		print "Done:"
		'''np.set_printoptions(threshold='nan', suppress = True)
		out = open('reduced_features.csv','w')
		out.write(str(self.reduced_features[:,:]))'''
		#self.plot()
		self.features = self.reduced_features

	def plot(self):
		X = self.reduced_features[:,0]
		Y = self.reduced_features[:,1]
		print "Saving plots :"
		for stock in self.stocks:
			print "Saving " + str(stock)
			filename = ("Labelled features of " + str(stock) + ".jpg").replace(':','_')
			plt.scatter(X,Y, c = self.labels[stock])
			plt.colorbar()
			plt.title(str(stock))
			#plt.show()
			plt.savefig(filename, format='jpg')
			plt.clf()

		
	def calcPredictions(self):
		data = ''
		j = 1
		self.predictions = cl.defaultdict(list)
		self.confidences = cl.defaultdict(list)
		for stock in self.stocks:
			print j
			X = []
			y = []
			for i in xrange(len(self.dates)-1):
				#print self.dates[i]
				X.append(self.features[i])
				y.append(self.labels[stock][i])
				if i < self.warm_up:
					self.predictions[stock].append(0)
					self.confidences[stock].append([0.0,0.0,0.0])
				else:
					#print y
					pred = 0
					scores = [0.0,0.0,0.0]
					try:
						oSvm = svm.SVC(C = 200000000000000, probability=False, class_weight = 'auto')
						oSvm.fit(X,y)
						pred = oSvm.predict(self.features[i+1])
						scores = oSvm.decision_function(self.features[i+1])
					except:
						print ".",
					#print pred
					#print scores
					self.predictions[stock].append(pred)
					self.confidences[stock].append(max(scores))
					#print self.predictions
					#print self.confidences
			j += 1
		predFile = 'predFile.pickle_' + str(self.price_diff_t)# + 'pca'
		confFile = 'confFile.pickle_' + str(self.price_diff_t)# + 'pca'
		pickle.dump(self.predictions, open(predFile,'wb'))
		pickle.dump(self.confidences, open(confFile,'wb'))

		'''if not os.path.exists(predFile):
			pickle.dump(self.predictions, open(predFile,'wb'))
			pickle.dump(self.confidences, open(confFile,'wb'))
		else:
			self.predictions = pickle.load(predFile,'rb')
			self.confidences = pickle.load(confFile,'rb')
		for stock in self.stocks:
			data += 'Prediction of ' + str(stock) + '\t' + str(self.predictions[stock]) + '\n' + 'Confidences of ' + str(stock) + '\t' + str(self.confidences[stock]) + '\n'
		data += '--------------------------------------------------------------------\n'
		for stock in self.stocks:
			data += 'Prediction of ' + str(stock) + '\t' + str(self.predictions[stock][-1]) + '\n' + 'Confidences of ' + str(stock) + '\t' + str(self.confidences[stock][-1]) + '\n'
		out = open('prediction.csv','w')
		out.write(data)'''
		
def accuracyWithoutZeros():
	data = 'priceDiff,nonzero,match,accuracy\n'
	for priceDiff in np.arange(0.00,0.06,0.01):
		predFile = 'predFile.pickle_' + str(priceDiff) + 'pca'
		predictions = pickle.load(open(predFile,'rb'))
		predictor = StockPricePrediction(SP100,priceDiffThreshold = priceDiff)
		print 'Calculating accuracy only for increasing and decreasing stocks for priceDifference : ' + str(priceDiff)
		match = 0.0
		nonZero = 0.0
		for stock in predictor.stocks:
			if (len(predictor.labels[stock])-1) != len(predictions[stock]):
				print 'Number of prediciton labels is wrong !!!'
				print str(stock),str(len(predictor.labels[stock])), str(len(predictions[stock]))
				break
			else:
				for i in xrange(len(predictor.dates)-1):
					if predictor.labels[stock][i] != 0:
						nonZero += 1
						if predictor.labels[stock][i] == predictions[stock][i]:
							match += 1
		data += str(priceDiff) +','+ str(nonZero) +','+ str(match) +','+ str(match / nonZero) + '\n'
		print 'Done'
	out = open('accuracy_nonZero.csv','w')
	out.write(data)

def accuracyCalc():
	#self.predictions = pickle.load(open('predFile.pickle_old','rb'))
	data = 'priceDiff,stock,accuracy\n'
	for priceDiff in np.arange(0.01,0.06,0.01):
		#priceDiff = 0.0
		start = datetime.now()
		print 'Calculating accuracy for priceDifference : ' + str(priceDiff)
		predictor = StockPricePrediction(SP100,priceDiffThreshold = priceDiff)
		predictor.pca(500)
		predictor.calcPredictions()
		predFile = 'predFile.pickle_' + str(priceDiff) + 'pca'
		confFile = 'confFile.pickle_' + str(priceDiff) + 'pca'
		pickle.dump(predictor.predictions, open(predFile,'wb'))
		pickle.dump(predictor.confidences, open(confFile,'wb'))
		for stock in predictor.stocks:
			if (len(predictor.labels[stock])-1) != len(predictor.predictions[stock]):
				print 'Number of prediciton labels is wrong !!!'
				print str(stock),str(len(predictor.labels[stock])), str(len(predictor.predictions[stock]))
				break
			else:
				match = 0.0
				for i in xrange(len(predictor.dates)-1):
					if predictor.labels[stock][i] == predictor.predictions[stock][i]:
						match += 1
				data += str(predictor.price_diff_t) +','+ str(stock) +','+str(match / (len(predictor.labels[stock])-1)) + '\n'
		print 'Calculating accuracy for priceDifference : ' + str(priceDiff) + ' done'
		end = datetime.now() 
		diff = end - start  
		print 'Required time : ' + str(int(math.floor(diff.seconds/60))) + 'minutes'
	out = open('accuracy.csv','w')
	out.write(data)

		
if __name__ == "__main__":
	#for priceDiff in np.arange(0.0,0.06,0.01):
	predictor = StockPricePrediction(SP100)
	#predictor.pca(500)
	predictor.calcPredictions()
	#accuracyWithoutZeros()
	#accuracyCalc()