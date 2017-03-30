# Natural Language Processing Pipeline Tool

### Modular Python-Based Implementation of Analysis Toolkit, Training and Test Modules

- Sentence Splitting,
- Tokenisation
- Normalisation Part-of-Speech Tagging
- Information Extraction
- Frequency Distribution
- Epydoc Wiki Document
- Modular Development
- Python Scripting
- HTML Parsing
- Stemming
- NLTK

## Contributors
[Antonis Pishias](https://github.com/antonispishias)
[Alexander George Karavournarlis](https://www.linkedin.com/in/alex-dj-prosgio-karavounarlis-322b5173/)

## Dependencies
The software was developed in the latest Python build, using Sublime Text 3. A python interpreter is necessary, in order to run the bundled .py scripts:

[Python 3.5.1](https://www.python.org/downloads/)

In order to execute the software correctly, it is necessary to install the following libraries, many modules of which are being imported in the main source code:

[Beautiful Soup 4](https://pypi.python.org/pypi/beautifulsoup4/4.4.1)

[Requests](http://docs.python-requests.org/en/latest/)

[Numpy](http://www.numpy.org/)

[Scipy](http://www.scipy.org/)

[NLTK](http://www.nltk.org/)

In order to install the above, further dependencies are expected to be fulfilled. Please refer to the relevant documentation, for more information. The system has been tested on an Ubuntu-based Linux PC.

## Included
**html** : a folder, containing the Epydoc wiki, built from the software’s documentation

**html/index.html** : the wiki’s homepage

**our tests** : a directory containing an automated test script, for the politics category

**pickled_class** : the directory of the .pickle files, generated during classification

**samples** : sample text files, acting as corpuses for training and testing

## Usage
_filenames of the format **n_m_restOfName.txt**_

`n` represents the ordered count of the parsed websites
`m` represents ordered count of the current process’s stage
if `m = 0`, then restOfName is the title of the submitted webpage
if `m > 0`, then restOfName, declares whether the file includes any of the following:
- The tokenized words (e.g. `0_1_tikenized`)
- The part-of-speech tags (e.g. `1_2_posTagged`)
- The insignificant stop words (e.g. `1_3_stopwords`)

#### theapp.py
The main .py script, the app executes from. Its most significant functions are as follows:

**Analyse**: Can be executed via the command parameters `-a` and `--analyse`. Can also be accessed, by means of the main menu, as option 1.

It is provided with an absolute path of a text file, containing the URLS, pointing to the websites, in need of parsing. In the command line, it is passed as the last parameter, whereas from the main menu stream, it is provided as user input, following a prompt.

**Train**: Can be executed via the terminal parameters `-t` and `--train`.
Can also be triggered, through the main menu, as option 2.
It is meant to train the system, to recognise whether a sentence belongs to a specific, unique category, or not. In order to train, it requires two corpus text files, one with sentences which belong to the category, and one with such, that do not. The absolute paths for these text files must be provided, either as command line arguments, or as user input, respectively.

**Is**: Also referred to as testing, it can be called as parameters `-i` and `--is`. The alternative way to trigger testing, is via the main menu, as option 3.
It requires an absolute path of a file, which is used as the test sample. Each of its lines contains a sentence, which will be qualified and voted for, in order to determine whether it classifies under the pre-determined category, or not. The sample files must be line-separated and as clean as possible.

**Show Help**: Output the help message to the console, it contains instructions on the current functionality, as well as the potential use of the software from the terminal. Main menu option 4, or parameter `-h` and `--help`.

**Quit**: Terminates execution of the application, from the looping main menu.


#### trainer.py
The python module, which uses the NLTK and other libraries, in order to train the system, against the provided sample.

#### is_a_is_not.py
The python module, which uses the libraries, in order to output the classification decision, as well as its confidence, for a sample of test sentences, after training.

#### required_websites.txt
The text file, necessary to test the analysis tool with, since it contains the URL of the websites requested to be parsed, in the assignment’s description.

### Example Usages
```sudo python3 theapp.py --analyse required_websites.txt```
Output is printed and stored in the n_m_restOfName files, in the root

```sudo python3 theapp.py --train samples/politics.txt samples/not_politics.txt```
Needs a few minutes to complete, .pickle files stored in picked_class folder

```sudo python3 theapp.py --is samples/politics_test.txt```
Outputs result to console

## Documentation
As a description of the architecture, functionality and implementation details, please visit the generated wiki at: [/html/index.html]

It covers the three main scripts: _theapp.py_, _trainer.py_ and _is_a_is_not.py_

All the custom-written functions have been documented
Having executed a script, you can print a particular function’s comments by running:
```print function_name.__doc__```
(within the Python interpreter)
To generate the wiki again, please execute from the linux shell:
```sudo epydoc --html theapp.py trainer.py is_a_is_not.py```
(assuming the necessary dependencies are all installed, including docstring)

### Example Training
```
Example Training/Testing Execution Output of training for politics category Training from the corpus, please be patient...
maryland 4006
Original Naive Bayes accuracy % : 71.78714859437751
Most Informative Features
stimulus = True     is a : is not = 25.9 : 1.0
tcot = True         is a : is not = 22.9 : 1.0
im = True           is not : is a = 13.9 : 1.0
global = True       is a : is not = 9.6 : 1.0
que = True          is not : is a = 9.0 : 1.0
p2 = True           is a : is not = 8.8 : 1.0
haha = True         is not : is a = 8.5 : 1.0
na = True           is not : is a = 8.1 : 1.0
wan = True          is not : is a = 7.6 : 1.0
lol = True          is not : is a = 7.6 : 1.0
teaparty = True     is a : is not = 6.8 : 1.0
soldier = True      is a : is not = 6.8 : 1.0
republican = True   is a : is not = 6.8 : 1.0
obama = True        is a : is not = 6.5 : 1.0
i = True            is a : is not = 6.5 : 1.0
MNB_classifier accuracy percent: 70.08032128514057
BernoulliNB_classifier accuracy percent: 69.77911646586345
LogisticRegression_classifier accuracy percent: 70.48192771084338
LinearSVC_classifier accuracy percent: 69.77911646586345
SGDClassifier accuracy percent: 69.37751004016064
```
