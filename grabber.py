from bs4 import BeautifulSoup
import requests
import re

#----------------------------------------------------------------------------------------------------------------
#grabbing the urls of articles only
def readSportFile(filename) :

	sport = re.compile(r"http://hubpages.com/sports/\b[^\W]+\b[^/]", flags=re.I | re.X)
	prog = re.compile(r"[\d]+")
	
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
		
		file = open("sportsArticles1", 'w')
		for website in websites:
			for p in range(1,73+1):
				current = website+"?page="+str(p)
				print(current)
				r = requests.get(current.strip())
				soup = BeautifulSoup(r.content, 'lxml')
				for link in soup.find_all('a'):
					adr = link.get('href')
					if (sport.match(adr)):
						last = adr.rsplit("/")[-1]
						if not (prog.match(last)) :
							file.write(adr+"\n")
			
		file.close()
		f.close()	

		


readSportFile("Sport")