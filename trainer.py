import nltk
import random
import pickle
import requests

from statistics import mode

from bs4 import BeautifulSoup

from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC, NuSVC, LinearSVC

#----------------------------------------------------------------------------------------------------------------
class Voter(ClassifierI):
	#classifiers have to be odd so the mode can work
	def __init__(self, *classifiers):
		self.__classifiers=classifiers

	def classify(self, features):
		votes = []
		for c in self.__classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self.__classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))	
		conf = choice_votes / len(votes)
		return conf

#----------------------------------------------------------------------------------------------------------------
def check(tag):
	if tag.name in ['script','form']:
		return False
	if (tag.string==None):
		return False
	return True

#----------------------------------------------------------------------------------------------------------------
def parseHTML(url):
	r = requests.get(url.strip())
	soup = BeautifulSoup(r.content, 'lxml')
	visible_text = []

	print("Page URL: ", url)

	for tag in soup.find_all(True):
		if(check(tag)):
			#print(tag.string)
			visible_text.append(tag.string)

	return visible_text

#----------------------------------------------------------------------------------------------------------------
#find the words in the document that are also in word_features 
def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features

#----------------------------------------------------------------------------------------------------------------
#making a file with all the parsed text from the urls grabbed
def UrL2Text():
	sport_web = open("training/sport_web.txt", "r").read()
	politics_web = open("training/politics_web.txt", "r").read()

	sport = open("training/sport.txt", "a")
	for webpage in sport_web.split("\n"):
		if (len(webpage)>0):
			parsed = parseHTML(webpage)
			for p in parsed:
				sport.write(p+"\n")
	sport.close()


	politics = open("training/politics.txt", "a")
	for webpage in politics_web.split("\n"):
		if (len(webpage)>0):
			parsed = parseHTML(webpage)
			for p in parsed:
				politics.write(p+"\n")
	politics.close()

#----------------------------------------------------------------------------------------------------------------
sport = open("training/sport.txt", "r").read()
politics = open("training/politics.txt", "r").read()

full_word_list = []
documents = []

#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

#create a list of tuples ()
for p in politics.split("\n"):
	documents.append((p,"politics"))
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			full_word_list.append(w[0].lower())


for p in sport.split("\n"):
		documents.append((p,"sport"))
		words = word_tokenize(p)
		pos = nltk.pos_tag(words)
		for w in pos:
			if w[1][0] in allowed_word_types:
				full_word_list.append(w[0].lower())


save_documents = open("pickled_class/documents.pickle","wb")
pickle.dump(documents,save_documents)
save_documents.close()

full_word_list = nltk.FreqDist(full_word_list)

word_features = list(full_word_list.keys())[:5000]

save_word_features = open("pickled_class/word_features.pickle","wb")
pickle.dump(word_features,save_word_features)
save_word_features.close()

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets[:100]
testing_set = featuresets[-50:]

featuresets_f = open("pickled_class/featuresets.pickle", "wb")
pickle.dump(featuresets,featuresets_f)
featuresets_f.close()

###############
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes accuracy % :", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("pickled_class/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("pickled_class/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("pickled_class/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("pickled_class/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("pickled_class/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

save_classifier = open("pickled_class/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()


