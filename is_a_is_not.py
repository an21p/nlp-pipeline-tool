import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
    """
    Declaration of the Vote Classifier class functions.
    """
    def __init__(self, *classifiers):
        """
        Initialiser/constructor method, assigns a copy of the classifier list
        to a member variable of the class object.

        @type  *classifiers: *list<ClassifierI>
        @param "classifiers: points to the list of Classifier objects
        """
        self._classifiers = classifiers

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
        for c in self._classifiers:
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
        @return          the ratio of the form 7 positive / 12 total (votes)
        """
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents_f = open("pickled_class/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("pickled_class/word_features.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    """
    Finds the features, by tokenising the provided text string.

    @type  document: string
    @param document: containts the text document's content

    @rtype           list<string>
    @return          contains the list of features
    """
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

open_file = open("pickled_class/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_class/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_class/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_class/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_class/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_class/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


def is_a_is_not(text):
    """
    Calls the find_features(document) method, on a lower-case version of the given string

    @type   text: string
    @param  text: the given test sample text

    @rtype        tuple<string, string>
    @return       holds two strings, one expressing the vode decision, and another, the confidence ratio
    """
    feats = find_features(text.lower())
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)