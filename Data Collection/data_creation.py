import pandas as pd
import numpy as np 
import lyricsgenius as lg
from lyricsgenius import Genius, OAuth2
import json
from requests.exceptions import Timeout
import re

path_to_muse = "~/Desktop/muse_v3.csv"
muse = pd.read_csv(path_to_muse, header = 0)
print(len(muse)) #should be 90001
print(muse.head())

english = re.compile('^[A-Za-z0-9\.,\' ]+$')
counter = 0
to_drop = []
while counter < len(muse):
    if english.match(muse.iloc[counter]['track']) is None or english.match(muse.iloc[counter]['artist']) is None:
        #print(f"dropping {muse.iloc[counter]['track']} {muse.iloc[counter]['artist']}")
        to_drop.append(counter)
    counter += 1
muse = muse.drop(to_drop)
print(len(muse))

muse.to_csv("new_muse.csv")


CAT = '7JcqISWAr4YdyUjqj1ov7NtCiJ_R2-9iRrMcsCGNU2FYZVJzjUQhO3c8FCOp25S-'
genius = Genius(access_token = CAT)
genius.timeout = 10
genius.sleep_time = 2 

all_lyrics = []
# data = []

muse.reset_index(drop=True)
indices = np.random.choice(66550 - 1, 6000, replace = False) # REPLACE WITH INDICES IN TXT FILE
print(indices)

data = muse.iloc[indices]

print(data.head())

for i in range(len(data)):
    #artist = genius.search_artist(, max_songs=20, sort="title")
    retries = 0
    while retries < 3:
        try:
            song = genius.search_song(data.iloc[i]["track"], data.iloc[i]["artist"])
        except Timeout as e:
            retries += 1
            pass
        if song is not None:
            try:
                all_lyrics.append(song.lyrics[song.lyrics.index('Lyrics') + 6:song.lyrics.index('You might also like')])
            except:
                all_lyrics.append("None")
        else:
            all_lyrics.append("None")
        break


data2 = data.assign(lyrics = all_lyrics)

data2.to_csv("data.csv", encoding = 'utf-8')

# f = open("test.txt", "w")
# f.write(data2.to_string())
# f.write('\n[NEW SONG]\n'.join(all_lyrics))
# f.close()

#print(data2)