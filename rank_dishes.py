import spacy
import ujson
from collections import defaultdict
from jellyfish import jaro_winkler
from textblob import TextBlob


def find_dish_polarities(nlp, reviews):
    # Extract dishes from reviews and collect sentiments per dish
    dish_pols = defaultdict(list)
    for text in reviews:
        for sentence in text.split('.'):
            doc = nlp(sentence.lower())
            dishes = [ent.text for ent in doc.ents if ent.label_ == 'DISH']
            if len(dishes) == 0:
                continue
            polarity = TextBlob(sentence).sentiment.polarity
            for dish in dishes:
                if ' and ' in dish:
                    dishes.extend(dish.split(' and '))
                else:
                    dish_pols[dish].append(polarity)
    return dish_pols


def combine_similar_dishes(dish_pols, threshold):
    # Create graph where two dishes are connected if they are similar enough
    # i.e. the Jaro-Winkler distance between them are above the threshold
    graph = {dish : [d for d in dish_pols if jaro_winkler(dish, d) > threshold] 
                for dish in dish_pols}

    # Group similar dishes together by finding connected components in graph
    # Use the shortest name and aggregate polarities for each component
    def dfs(dish):
        name, pols = dish, []
        to_visit = [dish]
        seen.add(dish)
        while to_visit:
            cur = to_visit.pop()
            if len(cur) < len(name):
                name = cur
            pols.extend(dish_pols[cur])
            neis = [nei for nei in graph[cur] if nei not in seen]
            to_visit.extend(neis)
            seen.update(neis)
        return name, pols
    
    combined_dish_pols = {}
    seen = set()
    for dish in graph:
        if dish not in seen:
            dish, pols = dfs(dish)
            combined_dish_pols[dish] = pols
    
    return combined_dish_pols


def score(pols, stars, prior_votes):
    # Bayesian Average Ratings: www.evanmiller.org/bayesian-average-ratings.html
    # Round scores to nearest star rating, start with uniform neutral prior
    votes = [prior_votes] * (stars * 2 + 1)
    utilities = range(-stars, stars + 1)
    for polarity in pols:
        votes[round(polarity * stars) + stars] += 1
    return sum(v * u for v, u in zip(votes, utilities)) / sum(votes)


def rank_dishes(models_path, reviews):
    dish_pols = find_dish_polarities(spacy.load(models_path), reviews)
    dish_pols = combine_similar_dishes(dish_pols, threshold=0.88)
    dish_scores = [(dish, score(pols, stars=5, prior_votes=2)) 
                    for dish, pols in dish_pols.items()]
    return sorted(dish_scores, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    # Rank and print top 20 dishes of sample, Emerald Chinese Restaurant
    # (www.yelp.com/biz/emerald-chinese-restaurant-mississauga)
    with open('emerald_reviews.json', 'r') as f:
        sample_reviews = ujson.loads(f.read())

    dishes = rank_dishes('models', sample_reviews)

    print('Emerald Chinese Restaurant - 20 best dishes:')
    for i in range(20):
        print(i + 1, '-', dishes[i][0], round(dishes[i][1], 3))
