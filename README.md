# Yelp Dish Ranker
Ranks dishes for Yelp restaurants from reviews using NLP tools (spaCy, TextBlob, Jellyfish)

Built upon data from the [Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge). 

This repo contains my contributions to my final group project for CS 378, Practical Applications of Natural Language Processing ([about the project](https://drive.google.com/file/d/1mAWVMVX-FmyH9mKzTuVpU-LFHbdkc2EF/view?usp=sharing)).

The `rank_dishes` method...
1. Recognizes dishes from Yelp text reviews using a [custom spaCy NER model](https://spacy.io/usage/training#ner) trained on collected Amazon Turk data
2. Finds the sentiment of sentences containing dishes using [TextBlob](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis) to collect ratings for dishes
3. Combines similar dishes, e.g. groups of dishes that have a high enough [Jaro-Winkler distance](https://jellyfish.readthedocs.io/en/latest/comparison.html#jaro-winkler-distance)
4. Aggregates dish ratings into a single score using [Bayesian Average Ratings](http://www.evanmiller.org/bayesian-average-ratings.html)
5. Returns dishes in an ordered list from best (highest score) to worst (lowest score)

## Run
To test the ranking algorithm, simply install the libraries found in `requirements.txt` and run `rank_dishes.py`. It will read in and execute on the sample reviews in `emerald_reviews.json`, which are real reviews for the [Emerald Chinese Restaurant](https://www.yelp.com/biz/emerald-chinese-restaurant-mississauga) in Mississauga, Ontario.

Running `rank_dishes.py` as is takes <10 seconds and prints the following output
```markdown
Emerald Chinese Restaurant - 20 best dishes:
1 - dim sum 0.598
2 - braised beef 0.292
3 - sticky rice 0.286
4 - spring rolls 0.208
5 - lobster 0.208
6 - wonton 0.208
7 - cookies 0.174
8 - bunch our staples 0.167
9 - preserved egg congee 0.167
10 - egg yolk bun 0.167
11 - leaf wrapped sausage 0.167
12 - octopus 0.167
13 - sautéed jumbo shrimp 0.13
14 - grilled vegetables 0.13
15 - fish fried rice 0.13
16 - beef sauté 0.13
17 - pork 0.125
18 - steamed fish 0.12
19 - fried rice 0.111
20 - bubble tea 0.087
```

Assuming you have the Yelp Dataset downloaded, `main.py` will run the ranking algorithm for all restaurants in the dataset and save the results to a json file. Note: This takes a **really long** time as there are over 15000 restaurants in the data.
