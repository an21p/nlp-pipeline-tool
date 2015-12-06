import sys
import nltk
import requests
import re

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
#http://www.bogotobogo.com//python/NLTK/tf_idf_with_scikit-learn_NLTK.php
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer



def showHelp() :
	print("\n" + str(sys.argv[0]) + \
"""

	program functions:
		HTML parsing
		Pre-Processing
			Sentence Splitting, 
			Tokenization, 
			Normalization 
		Part-of-Speech Tagging 
		Stopwords
		Selecting Keywords
		Stemming or Morphological Analysis
		Print Help

	command line options:
		-h:	print this message

		-w:	parse websites listed in a file and 
			stores the pased output of each in a file, 
			each website should be in a new line
			e.g.: [PROGRAM_NAME] -w input_file 

		-pp:	pre-processes a file and stores the output in a file
			e.g.: [PROGRAM_NAME] -pp input_file 
			
		-pos:	apply Part-of-Speech Tagging on the file
			and store the output in a file
			e.g.: [PROGRAM_NAME] -pos input_file

		-k:	selects the Keywords of the file
			and store the output in a file
			e.g.: [PROGRAM_NAME] -k input_file

		-s:	applies Stemming on the file
			and store the output in a file
			e.g.: [PROGRAM_NAME] -s input_file

		-a:	goes though every step of the pipeline once
			and outputs a file for each website in the file
			e.g.: [PROGRAM_NAME] -a input_file 
		
	protip: this program can also run in interactive mode :^)
	
	""")

#----------------------------------------------------------------------------------------------------------------
def mainMenu() :
	while True :
		print("""
Main Menu:
	[1] Analyse a file
	[2] Help
	[3] Quit
		""")

		option = int(input("Enter a number : "))
		if option == 1 :
			analyseFile( \
				input("\nEnter the name of the file containing a list of websites : "))
			
		elif option == 2 : showHelp()
		elif option == 3 : sys.exit()
		else : print("Wrong Input")

#----------------------------------------------------------------------------------------------------------------
def wordTokenize(text, num) :
	file_title=str(num)+'_1_tikenized'


	sentences = nltk.sent_tokenize(text)
	tokenized = [nltk.word_tokenize(sent) for sent in sentences]

	file = open(file_title, 'w')
	print("Tokenized: ", file_title )
	for sentence in tokenized:
		for word in sentence:
			file.write(word+' ')
		file.write('\n')
	file.close()
	return tokenized

#----------------------------------------------------------------------------------------------------------------
def posTag(text, num) :
	file_title=str(num)+'_2_posTagged'


	posTagged = [nltk.pos_tag(sent) for sent in text]

	file = open(file_title, 'w')
	print("POS Tagged: ", file_title )
	for sentence in posTagged:
		for tag in sentence:
			#print(tag)
			file.write("|".join(tag)+"\n")
		file.write("\n")
	file.close()
	return posTagged #list of pos tuples (words) in list of sentences 

#----------------------------------------------------------------------------------------------------------------
def stopWords(text, num) :
	file_title=str(num)+'_3_stopwords'
	stop_words = set(stopwords.words("english"))

	filtered_sentences = []

	file = open(file_title, 'w')
	print("Removed Stopwords: ", file_title )
	for sentence in text:
		filtered_words = []
		for tpl in sentence:
			if tpl[0] not in stop_words:
				filtered_words.append(tpl)
				file.write("|".join(tpl)+"\n")
		if (len(filtered_words) > 0):
			filtered_sentences.append(filtered_words)	
	file.close()

	return filtered_sentences #list of pos tuples (words) in list of sentences 

#----------------------------------------------------------------------------------------------------------------
def traverse(tree): 
	try:
		tree.label()
	except AttributeError:
		return 
	else:
		if tree.label() == 'NP': 
			print (tree) # or do something else 
		else:
			for child in tree: traverse(child)

#----------------------------------------------------------------------------------------------------------------
def chunk(text) :

	chuckGram =r"""
		NP: {<NNP>+}                		# chunk sequences of proper nouns
			{<DT.?|PP\$>?<JJ.?>*<NN|NNS>}   # chunk determiner/possessive, adjectives and noun
	    P: {<IN>}           # Preposition
	    V: {<V.*>}          # Verb
	    PP: {<P> <NP>}      # PP -> P NP
	    VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*
	"""
	chunkParser = nltk.RegexpParser(chuckGram)
	for sentence in text:
		chunked = chunkParser.parse(sentence)
		traverse(chunked)

#----------------------------------------------------------------------------------------------------------------
def stem(text, num) :
	file_title=str(num)+'_5_stemmed'

	ps = PorterStemmer()

	for w in text:
 		print(ps.stem(w))

#----------------------------------------------------------------------------------------------------------------
def analyseFile(filename, a) :
	# initialise filename
	file_open = False
	while True :
		try :	
			f = open(filename ,'r')
			file_open = True
			break
		except IOError as err : 
			print("File does not exist")
			break

	if not file_open :
		print("File could not be opened. The program will terminate")
	else :
		websites = f.read().splitlines()
		filenum = 0
		for website in websites:
			text_content = parseHTML(website, filenum)
			text_content = " ".join(text_content)
			if (a):
				tokenized = wordTokenize(text_content, filenum)
				posTagged = posTag(tokenized, filenum)
				noStopwords = stopWords(posTagged, filenum) 
				chunked = chunk(noStopwords)
				#chunked = chunk(posTagged)
				#stemmed = stem(tokenized, filenum)
			filenum+=1
		f.close()	

#----------------------------------------------------------------------------------------------------------------
def check(tag):
	if tag.name in ['script','form']:
		return False
	if (tag.string==None):
		return False
	return True

#----------------------------------------------------------------------------------------------------------------
def parseHTML(url, num):
	r = requests.get(url.strip())
	soup = BeautifulSoup(r.content, 'lxml')
	file_title = str(num)+'_0_'+ remove(soup.title.string, '\/:*?"<>|');


	visible_text = []

	file = open(file_title, 'w')
	print("Page URL: ", url,"\nFilename: ", file_title )

	for tag in soup.find_all(True):
		if(check(tag)):
			#print(tag.string)
			visible_text.append(tag.string)
			file.write(tag.string+"\n")
	file.close()

	return visible_text

#----------------------------------------------------------------------------------------------------------------
def remove(value, deletechars):
    for c in deletechars:
        value = value.replace(c,'')
    return value;

#----------------------------------------------------------------------------------------------------------------
#main program

if len(sys.argv) == 1:
	#enter interactive mode
	mainMenu()
else:
	#enter automated mode
	if sys.argv[1] == "-h": showHelp()
	
	elif sys.argv[1] == "-a":
		if len(sys.argv) == 3:
			analyseFile( str(sys.argv[2]), True)
		else: print("invalid number of parameters for function 'analyseFile()'")
	
	sys.exit()

