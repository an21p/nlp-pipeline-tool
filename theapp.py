"""
Antonis Pishias, Alex Karavournarlis - University of Essex - Copyright 2015
"""

import sys
import nltk
import requests
import re

import trainer

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer



def showHelp() :
	"""
	Prints the help document to the console.
	Provides the key for the first parameter of the command line.
	Each parameter value, corresponds to an option, which
	represents a unique, self-sufficient function of the program.

	@rtype:		void
	@return:	console output
	"""
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
	"""
	Prints the main menu options, from which the user makes a choice.
	Presets the user interface intuitively, as the choice is represented
	by an integer, verified by the necessary controls.

	@rtype:		void
	@return:	console output
	"""
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
	"""
	Tokenises a text segment per sentence, then each sentence, on a word basis.
	Stores the output tokens in a file, predetermined by means of an index.

	@type  text: string
	@param text: holds untokenized, "raw" text
	@type   num: number
	@param  num: represents the order of the file, in the sample data set

	@rtype:		 list<string>
	@return:	 word tokens
	"""
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
	"""
	Extracts part-of-speech tags from a text segment's list of tokens.
	Stores the output tokens in a file, predetermined by means of an index.

	@type  text: string
	@param text: holds the list of word tokens
	@type   num: number
	@param  num: represents the order of the file, in the sample data set

	@rtype:		 list< < tuple < string, string > > >
	@return:	 tuples of each word and its part-of-speech tag, per sentence
	"""
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
	"""
	Filters stop words out of a text segment's list of part-of-speech tags.
	Stores the output tokens in a file, predetermined by means of an index.

	@type  text: string
	@param text: holds the list of POS tags
	@type   num: number
	@param  num: represents the order of the file, in the sample data set

	@rtype:		 list< < tuple < string, string > > >
	@return:	 tuples of each reamining word and its part-of-speech tag, per sentence
	"""
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
def stem(text, num) :
	"""
	Outputs stems from a a text segment's list of part-of-speech tags.
	Stores the output tokens in a file, predetermined by means of an index.

	@type  text: string
	@param text: holds the list of POS tags
	@type   num: number
	@param  num: represents the order of the file, in the sample data set

	@rtype:		 void
	@return:	 console output
	"""
	file_title=str(num)+'_4_stemmed'

	ps = PorterStemmer()

	# TODO:
	for w in text:
 		print(ps.stem(w))

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
	"""
	Parses a specifically determined chunk sequence and utilises it for chunking.
	It is then used to parse the text segment, provided as a parameter.
	Eternally outputs a GUI, including a graph of the chunk parser's output.
	The interface consists of a new window, per sentence.

	@type  text: string
	@param text: holds the text segment

	@rtype:		 void
	@return:	 graphical output
	"""
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
	"""
	This is effectively the main, collective method of execution.
	It initially attempts to open the file, from the provided path/name.
	The file's content is split into an array, one element per line.
	Should the user have entered the full analysis parameter "-a",
	tokenisation, POS tagging, filtering, chunking and stemming are
	performed, in this order.

	@type  filename: string
	@param filename: holds the name of the input file

	@rtype:		 	 void
	@return:	 	 file output

	@except     err: the error occurs, should the file not be found
	"""
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
def parseHTML(url, num):
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
	"""
	A helper method which statically replaces a substring's value
	with empty characters. Effectively allows its deletion.

	@type   value: string
	@param  value: the source string, to be processed

	@rtype  value: string
	@return value: the string, after processing is done
	"""
	for c in deletechars:
		value = value.replace(c,'')

	return value

#----------------------------------------------------------------------------------------------------------------
def main():
	"""
	In all its glory, that's the program's formal main function.
	Considers the command line arguments entered in the console.
	It allows for automatic/interactive mode, or a help prompt.
	Terminates execution, as soon as processing is complete.
	"""
	if len(sys.argv) == 1:
		mainMenu()
	else:
		if sys.argv[1] == "-h": showHelp()
		
		elif sys.argv[1] == "-a":
			if len(sys.argv) == 3:
				print("Analysing the file...")
				analyseFile( str(sys.argv[2]), True)
			else: print("Invalid number of parameters for the analyse 'a' command, terminating.")

		elif sys.argv[1] == '-t':
			if len(sys.argv) == 4:
				print("Training from the corpus, please be patient...")			
				trainer.trainCorpus(str(sys.argv[2]), str(sys.argv[3]))
			else: print("Invalid number of parameters for the train 't' command, terminating")
		
		sys.exit()

if __name__ == "__main__":
	"""
	This is the interpreter's conventional variable check splinter.
	Should the "__name__" variable's value be "__main__", then the
	source code is indeed been executed, directly as software.
	Since, however, it is meant to be used also as a library, the
	main() function shouldn't execute then. Should another module
	import the code, the "__name__" variable will be assigned its name.
	"""
	main()