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
	"""
    Declaration of the Voter class functions.
    """
	#classifiers have to be odd so the mode can work
	def __init__(self, *classifiers):
		"""
        Initialiser/constructor method, assigns a copy of the classifier list
        to a member variable of the class object.

        @type  *classifiers: *list<ClassifierI>
        @param "classifiers: points to the list of Classifier objects
        """
		self.__classifiers=classifiers

	def classify(self, features):
		"""
        For each of the classifier objects, it attempts to classify based on the
        features, provided via the parameter. Each classification is expressed
        as a vote.

        @type  features: list<string>
        @param features: containts the list of features

        @rtype           string
        @return          the most popular vote
        """
		votes = []
		for c in self.__classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		"""
        Introduces a sentiment of confidence, as a parameter. Calculates the ratio
        of positive votes, for the winning vote in particular, as per the total.

        @type  features: list<string>
        @param features: containts the list of features

        @rtype           number
        @return          the ratio of the form: 3 positive / 5 total (votes)
        """
		votes = []
		for c in self.__classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))	
		conf = choice_votes / len(votes)
		return conf

#----------------------------------------------------------------------------------------------------------------
def check(tag):
	"""
	Checks whether a tag is a "script" or a "form" tag.

	@type  tag: string
	@param tag: ......

	@rtype:		boolean
	@return:    returns True if the tag is of type "script" or "form"

	"""
	if tag.name in ['script','form']:
		return False
	if (tag.string==None):
		return False
	return True

#----------------------------------------------------------------------------------------------------------------
def parseHTML(url):
	"""
	Essentially downloads a web page, from a provided URL.
	Its HyperText Markup is then parsed, by means of BeautifulSoup.
	A regex is used to clean the web page's title, and assign the
	result to the output file. All lexicon tags from the file are
	then separated, and appended to a new string, to be returned.

	@type  url: url
	@param url: the web page's uniform resource locator

	@rtype:		string
	@return:    holds the majority of words within the original text

	"""
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
	"""
    Finds the features, by tokenising the provided text string.

    @type  features: string
    @param features: containts the text document's content

    @rtype           list<string>
    @return          contains the list of features
    """
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features

#----------------------------------------------------------------------------------------------------------------
def splitCorpus(corpus_path):
	"""
	Parsing method, which allows to split the unified corpus, for our tests.
	Each line started with the annotations "POLIT" or "NOT", to indicate
	wheter politics-related or not, respectively.

	@type  corpus_path: string
	@param corpus_path: the absolute path of the unified training corpus

	@rtype void
	@return file output
	"""
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
	"""
	Parser method, utilised to concatenate all the parsed text,
	downloaded from URLs, and write it all to a file. In that older test,
	it forms training corpuses, from their corresponding URL dictionaries.
	The categories were then "sport" and "politics". It was enlightening.

	@rtype  void
	@return file output
	"""
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
	"""
	Trains, given two complementary corpuses. One of these
	includes sentences belonging to a certain category, the
	other includes sentences not belonging to that category.

	@type  is_a_path: 	  string
	@param is_a_path: 	  defined as "is_a", the sample set of
					  	  sentences, belonging to the category

    @type  is_not_a_path: string
    @param is_not_a_path: defined as "is_not", the other sample
    					  set, not belonging to the category

    @rtype  void
    @return file output, in the form of dumped pickled classes
	"""
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
	"""
	Handles the frequency distribution, saving word features, shuffling feature sets,
	training the Naive Bayes classifiers etc. It also separates the training sample
	to a subset for training, and one, reserved for testing back. Eternally, it saves
	the final .pickle file in the "pickled_class" directory.

	@type   documents: 	    string
	@param  documents:      stores the content text of the source .pickle file 

	@type   full_word_list: list<string>
	@param  full_word_list: containts all words

	@rtype  void
	@return console output
	"""
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
	"""
	The trainer's main function, when necessary to execute as a module,
	directly. Performs our test, of terms which belong to the politics
	category, and terms which do not.
	"""
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
	"""
	This is the interpreter's conventional variable check splinter.
	Should the "__name__" variable's value be "__main__", then the
	source code is indeed been executed, directly as a module.
	Since, however, it is meant to be used also as an import, the
	main() function shouldn't execute then. Should another module
	import the code, the "__name__" variable will be assigned its name.
	"""
	main()