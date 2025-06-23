import pandas as pd
import json
from app import get_movie_poster

# Load the movie dataset
csv_path = 'dataset/movies2.csv'
df = pd.read_csv(csv_path, delimiter=';', encoding='ISO-8859-1')

poster_cache = {}

for idx, row in df.iterrows():
    title = row['name']
    year = row['year']
    director = row['director'] if 'director' in row else None
    released = row['released'] if 'released' in row else None
    cache_key = f"{title}_{year}_{director}"
    if cache_key not in poster_cache:
        poster_url = get_movie_poster(title, year, director, released)
        poster_cache[cache_key] = poster_url
    print(f"Cached: {cache_key} -> {poster_cache[cache_key]}")

# Save to JSON
with open('posters_cache.json', 'w', encoding='utf-8') as f:
    json.dump(poster_cache, f, ensure_ascii=False, indent=2)

print(f"Saved {len(poster_cache)} poster URLs to posters_cache.json") 