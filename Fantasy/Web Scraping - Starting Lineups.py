import requests
from bs4 import BeautifulSoup

url = 'https://dailynbalineups.com/'
response = requests.get(url)
html_soup = BeautifulSoup(response.text, 'html.parser')

container = html_soup.find_all('div', class_='game-wrapper')

starters = []
for i in container:
    hometeam = i.find_all('div', class_ = 'home-team')[0].tbody.find_all('td')
    awayteam = i.find_all('div', class_ = 'away-team')[0].tbody.find_all('td')
    for i in range(0,len(hometeam),2):
        starters.append(hometeam[i].text)
    for i in range(0,len(awayteam),2):
        starters.append(awayteam[i].text)
