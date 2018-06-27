import urllib.request
from bs4 import BeautifulSoup
#from selenium import webdriver

#driver = webdriver.Chrome('/path/to/chromedriver')
#driver.get
url = "https://vine.co/playlists/year4/"
#url = "https://www.youtube.com/watch?v=q6EoRBvdVPQ/"
soup = BeautifulSoup(urllib.request.urlopen(url), "lxml")
print(soup.prettify())
