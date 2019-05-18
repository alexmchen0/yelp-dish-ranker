import time
import ujson
import spacy
from collections import defaultdict
from rank_dishes import rank_dishes

def get_restaurants(business_path):
    # Get business_id of all restaurants in the dataset
    restaurants = {}
    with open(business_path, 'r', encoding='utf8') as f:
        for line in f:
            jso = ujson.loads(line)
            if jso['categories'] and 'Restaurants' in jso['categories']:
                restaurants[jso['business_id']] = jso['name']
    return restaurants


def get_reviews(restaurants, review_path):
    # Get reviews for each business_id 
    reviews = defaultdict(list)
    with open(review_path, 'r', encoding='utf8') as f:
        for line in f:
            jso = ujson.loads(line)
            business_id = jso['business_id']
            if business_id in restaurants:
                reviews[business_id].append(jso['text'].replace('\n',' '))
    return reviews


if __name__ == '__main__':
    start_time = time.time()
    restaurants = get_restaurants('yelp_dataset/business.json')
    reviews_dict = get_reviews(restaurants, 'yelp_dataset/review.json')
    best_dishes = {}
    for bid, reviews in reviews_dict.items():
        dishes = rank_dishes('models', reviews)
        best_dishes[bid] = [dish for dish, _ in dishes]

    # Write rankings to a file
    max_bytes = 2**31 - 1
    dump = ujson.dumps(best_dishes)
    with open('best_dishes.json', 'w') as f:
        for i in range(0, len(dump), max_bytes):
            f.write(dump[i:i + max_bytes])

