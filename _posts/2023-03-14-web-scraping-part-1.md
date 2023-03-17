---
layout: post
title:  Gathering Pokémon Base Statistics
author: Michael Okuda
description: Learn how to web scrape data of Pokémon stats and put it into a pandas data frame.
image: /assets/images/pokeball.png
---

# Introduction

[Pokémon](https://www.pokemon.com/us) is a classic video game played by many people.  Personally, playing Pokémon during my teenage years helped me develop an interest and love for numbers.  I loved getting to see the stats increase each time my Pokémon would level up.  After understanding the game and beating it several times, I wondered what Pokémon would be the strongest to put in my team.

# Gathering the Data

Several websites contain data of Pokémon's base stats, including [this website](https://pokeapi.co/docs/v2#wrap), which will be used for web scraping to collect the data.  I was able to determine that it was ethical to use the data from the above website because it states that it is "free and open to use."  The repository of the data that I collected is available [here](https://github.com/mokuda2/pokemon).

To start, use a Jupyter notebook file to web scrape data from the website.  I used the following packages:

```
import pandas as pd
import numpy as np
import requests
```

We'll create a variable where the URL of the website is assigned to it.

```
poke_url = 'https://pokeapi.co/api/v2/pokemon/'
```

Using the requests library, we'll use the "get" method, which will request for the data located at the URL.

```
r2 = requests.get(poke_url)
```

Calling the json method (r2.json()) should give us a JSON file with the following output:

```
{'count': 1281,
 'next': 'https://pokeapi.co/api/v2/pokemon/?offset=20&limit=20',
 'previous': None,
 'results': [{'name': 'bulbasaur',
   'url': 'https://pokeapi.co/api/v2/pokemon/1/'},
  {'name': 'ivysaur', 'url': 'https://pokeapi.co/api/v2/pokemon/2/'},
  {'name': 'venusaur', 'url': 'https://pokeapi.co/api/v2/pokemon/3/'},
  {'name': 'charmander', 'url': 'https://pokeapi.co/api/v2/pokemon/4/'},
  {'name': 'charmeleon', 'url': 'https://pokeapi.co/api/v2/pokemon/5/'},
  {'name': 'charizard', 'url': 'https://pokeapi.co/api/v2/pokemon/6/'},
  {'name': 'squirtle', 'url': 'https://pokeapi.co/api/v2/pokemon/7/'},
  {'name': 'wartortle', 'url': 'https://pokeapi.co/api/v2/pokemon/8/'},
  {'name': 'blastoise', 'url': 'https://pokeapi.co/api/v2/pokemon/9/'},
  {'name': 'caterpie', 'url': 'https://pokeapi.co/api/v2/pokemon/10/'},
  {'name': 'metapod', 'url': 'https://pokeapi.co/api/v2/pokemon/11/'},
  {'name': 'butterfree', 'url': 'https://pokeapi.co/api/v2/pokemon/12/'},
  {'name': 'weedle', 'url': 'https://pokeapi.co/api/v2/pokemon/13/'},
  {'name': 'kakuna', 'url': 'https://pokeapi.co/api/v2/pokemon/14/'},
  {'name': 'beedrill', 'url': 'https://pokeapi.co/api/v2/pokemon/15/'},
  {'name': 'pidgey', 'url': 'https://pokeapi.co/api/v2/pokemon/16/'},
  {'name': 'pidgeotto', 'url': 'https://pokeapi.co/api/v2/pokemon/17/'},
  {'name': 'pidgeot', 'url': 'https://pokeapi.co/api/v2/pokemon/18/'},
  {'name': 'rattata', 'url': 'https://pokeapi.co/api/v2/pokemon/19/'},
  {'name': 'raticate', 'url': 'https://pokeapi.co/api/v2/pokemon/20/'}]}
```

Then we'll create two list comprehensions.  The first list comprehension is of Pokémon names.  The second list comprehension contains all base stats for all Pokémon.  Because there are 1010 Pokémon, creating these lists will take a few minutes.

```
pokemon_name_list = [requests.get(poke_url + str(i+1) + '/').json()['name'] for i in range(0, 1010)]
base_stats_list = [j['base_stat'] for i in range(0, 1010) for j in requests.get(poke_url + str(i+1)).json()['stats']]
```

We'll then take the list comprehension of base stats and break it up into different lists depending on the base stat: hp, attack, defense, special attack, special defense, and speed.

```
hp_stat = []
attack_stat = []
defense_stat = []
special_attack_stat = []
special_defense_stat = []
speed_stat = []

for i in range(len(base_stats_list)):
    if i % 6 == 0:
        hp_stat.append(base_stats_list[i])
    elif i % 6 == 1:
        attack_stat.append(base_stats_list[i])
    elif i % 6 == 2:
        defense_stat.append(base_stats_list[i])
    elif i % 6 == 3:
        special_attack_stat.append(base_stats_list[i])
    elif i % 6 == 4:
        special_defense_stat.append(base_stats_list[i])
    else:
        speed_stat.append(base_stats_list[i])
```

Now that we have the necessary data, we'll create a pandas data frame.  A dictionary is used to hold the list of Pokémon names and the lists of base stats.

```
dictionary = {"Pokemon_Name" : pokemon_name_list, "hp" : hp_stat, "attack" : attack_stat, "defense" : defense_stat,
             "special attack" : special_attack_stat, "special defense" : special_defense_stat, "speed" : speed_stat}
pokemon_df = pd.DataFrame(dictionary)
pokemon_df
```

The data frame should look something like the following:

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/pokemon_data_frame.png)

Finally, we'll save the data frame to a CSV file.  This will help us have to run this code only once and not have to worry about rerunning the list comprehensions.

```
pokemon_df.to_csv('pokemon_base_stats.csv', index=False)
```

# Conclusion

In this post, we were able to collect Pokémon base stats data.  Again, I'm interested in finding which Pokémon would have the highest overall base stats as well as the highest individual base stats.  This analysis is useful to train powerful Pokémon.  If there are any other suggestions for finding which Pokémon are most powerful, comment below!