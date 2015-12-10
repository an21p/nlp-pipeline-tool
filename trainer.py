import re
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

word_features = []

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
def splitCorpus(corpus_path):
	politics_bool = open(corpus_path, "r").read()

	politics = ""
	not_politics = ""

	for line in politics_bool.split("\n"):
		line = re.sub(r'^https?:\/\/.*[\r\n]*', '', line, flags=re.VERBOSE)
		if (len(line) > 0 and "POLIT" in line):
			politics += line.replace("POLIT\t","")+"\n"
		elif (len(line) > 0 and "NOT" in line):
			not_politics += line.replace("NOT\t","")+"\n"

	politics_txt = open("samples/politics.txt", "w+")
	politics_txt.write(politics)
	politics_txt.close()

	not_politics_txt = open("samples/not_politics.txt", "w+")
	not_politics_txt.write(not_politics)
	not_politics_txt.close()

#----------------------------------------------------------------------------------------------------------------
#making a file with all the parsed text from the urls grabbed
def UrL2Text():
	sport_web = open("samples/sport_web.txt", "r").read()
	politics_web = open("samples/politics_web.txt", "r").read()

	sport = open("samples/sport.txt", "a")
	for webpage in sport_web.split("\n"):
		if (len(webpage)>0):
			parsed = parseHTML(webpage)
			for p in parsed:
				sport.write(p+"\n")
	sport.close()


	politics = open("samples/politics.txt", "a")
	for webpage in politics_web.split("\n"):
		if (len(webpage)>0):
			parsed = parseHTML(webpage)
			for p in parsed:
				politics.write(p+"\n")
	politics.close()

#----------------------------------------------------------------------------------------------------------------
def trainCorpus(is_a_path, is_not_path):
	is_a = open(is_a_path, "r").read()
	is_not = open(is_not_path, "r").read()

	full_word_list = []
	documents = []

	#allowed_word_types = ["J","R","V"]
	allowed_word_types = ["J"]

	#create a list of tuples ()
	for a in is_a.split("\n"):
		documents.append((a,"is a"))
		words = word_tokenize(a)
		pos = nltk.pos_tag(words)
		for w in pos:
			if w[1][0] in allowed_word_types:
				full_word_list.append(w[0].lower())


	for p in is_not.split("\n"):
			documents.append((p,"is not"))
			words = word_tokenize(p)
			pos = nltk.pos_tag(words)
			for w in pos:
				if w[1][0] in allowed_word_types:
					full_word_list.append(w[0].lower())

	processDocuments(documents, full_word_list)

def processDocuments(documents, full_word_list):
	save_documents = open("pickled_class/documents.pickle","wb")
	pickle.dump(documents,save_documents)
	save_documents.close()

	full_word_list = nltk.FreqDist(full_word_list)

	global word_features
	word_features = list(full_word_list.keys())[:5000]

	save_word_features = open("pickled_class/word_features.pickle","wb")
	pickle.dump(word_features,save_word_features)
	save_word_features.close()

	featuresets = [(find_features(rev), category) for (rev, category) in documents]

	random.shuffle(featuresets)
	print(len(featuresets))

	training_index = 3*len(featuresets)//4
	testing_index = -(len(featuresets)//4-5)

	training_set = featuresets[:training_index]
	testing_set = featuresets[testing_index:]

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

def main():
	splitCorpus("samples/politics_bool.txt")

	politics = open("samples/politics_min.txt", "r").read()
	not_politics = open("samples/not_politics_min.txt", "r").read()

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


	for p in not_politics.split("\n"):
			documents.append((p,"not_politics"))
			words = word_tokenize(p)
			pos = nltk.pos_tag(words)
			for w in pos:
				if w[1][0] in allowed_word_types:
					full_word_list.append(w[0].lower())

	processDocuments(documents, full_word_list)


if __name__ == "__main__":
    main()