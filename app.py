import firebase_admin
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
from tmdbv3api import TMDb, Movie
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import unquote
import json
import os
from datetime import datetime
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import MaxRetryError
from firebase_admin import credentials, auth as firebase_auth, firestore
from dotenv import load_dotenv
import sys

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'a-default-secret-key-for-development')

# Configure requests session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Initialize TMDb with your API key
tmdb = TMDb()
tmdb.api_key = os.getenv('TMDB_API_KEY')
tmdb.language = 'en'
tmdb.session = session  # Use our configured session
print(f"TMDb API Key being used: {tmdb.api_key[:8]}...")  # Only print first 8 chars for security

# Load the movie dataset
df = pd.read_csv("dataset/movies2.csv", delimiter=";", encoding='ISO-8859-1')

# Preprocess the data for content-based filtering
df['info'] = df[['name', 'genre', 'director', 'writer', 'star', 'company']].apply(lambda x: ' '.join(x.astype(str)),
                                                                                  axis=1)
countV = CountVectorizer(min_df=20, stop_words='english')
matrice_transform = countV.fit_transform(df['info'])
cosine_sim = cosine_similarity(matrice_transform)

# Load poster cache from JSON if available
poster_cache = {}
if os.path.exists('posters_cache.json'):
    try:
        with open('posters_cache.json', 'r', encoding='utf-8') as f:
            poster_cache = json.load(f)
        print(f"Loaded {len(poster_cache)} poster URLs from cache")
    except Exception as e:
        print(f"Error loading poster cache: {str(e)}")

# File to store previous positions
POSITIONS_FILE = 'movie_positions.json'

# Track number of API calls for rate limiting
api_calls = 0
last_api_call_time = datetime.now()
API_CALL_LIMIT = 40  # TMDb allows 40 requests per 10 seconds
API_WINDOW = 10  # 10 seconds window

# Firestore initialization (if not already done)
if not hasattr(app, 'firebase_initialized'):
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
    if not service_account_path or not os.path.exists(service_account_path):
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_PATH environment variable is not set or the file does not exist.")
    
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred, {
        'projectId': os.getenv('FIREBASE_PROJECT_ID'),
    })
    app.firebase_initialized = True

db = firestore.client()

def check_rate_limit():
    """Check if we're within TMDb API rate limits"""
    global api_calls, last_api_call_time
    now = datetime.now()
    time_diff = (now - last_api_call_time).total_seconds()
    
    if time_diff >= API_WINDOW:
        # Reset counter if window has passed
        api_calls = 0
        last_api_call_time = now
        return True
    
    if api_calls >= API_CALL_LIMIT:
        # Wait until window resets
        sleep_time = API_WINDOW - time_diff
        print(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
        time.sleep(sleep_time)
        api_calls = 0
        last_api_call_time = datetime.now()
        return True
    
    api_calls += 1
    return True

def load_previous_positions():
    """Load previous movie positions from file"""
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading previous positions: {str(e)}")
    return {}

def save_current_positions(positions):
    """Save current movie positions to file"""
    try:
        with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(positions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving current positions: {str(e)}")

def save_poster_cache():
    """Save the poster cache to file"""
    try:
        with open('posters_cache.json', 'w', encoding='utf-8') as f:
            json.dump(poster_cache, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(poster_cache)} poster URLs to posters_cache.json")
    except Exception as e:
        print(f"Error saving poster cache: {str(e)}")

def get_movie_poster(movie_title, movie_year=None, director_name=None, released_date=None):
    """
    Fetches the movie poster URL from TMDb based on the movie title, year, and optionally director and release date.
    """
    try:
        # Check rate limit before making API call
        if not check_rate_limit():
            return None
            
        print(f"Searching for movie: {movie_title}")
        
        # Use direct requests instead of TMDb wrapper
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': tmdb.api_key,
            'query': movie_title,
            'page': 1,
            'language': 'en'
        }
        
        try:
            response = session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json().get('results', [])
            
            if not search_results:
                print(f"No results found for {movie_title}")
                return None
                
            print(f"Found {len(search_results)} results for {movie_title}")
            
            # Filter by year if provided
            year_matches = []
            for result in search_results:
                if movie_year and result.get('release_date'):
                    result_year = result['release_date'].split('-')[0]
                    if str(result_year) == str(movie_year):
                        year_matches.append(result)
            
            # If year matches found, try to match director
            if year_matches and director_name:
                for result in year_matches:
                    # Check rate limit before making API call
                    if not check_rate_limit():
                        return None
                        
                    try:
                        # Fetch details to get director info
                        details_url = f"https://api.themoviedb.org/3/movie/{result['id']}/credits"
                        details_response = session.get(details_url, params={'api_key': tmdb.api_key}, timeout=10)
                        details_response.raise_for_status()
                        credits = details_response.json()
                        
                        crew = credits.get('crew', [])
                        for crew_member in crew:
                            if crew_member.get('job') == 'Director' and director_name.lower() in crew_member.get('name', '').lower():
                                if result.get('poster_path'):
                                    return f"https://image.tmdb.org/t/p/w500{result['poster_path']}"
                                    
                    except Exception as e:
                        print(f"Error fetching credits for '{movie_title}': {str(e)}")
                        continue
                        
                # If no director match, pick the first year match
                if year_matches[0].get('poster_path'):
                    return f"https://image.tmdb.org/t/p/w500{year_matches[0]['poster_path']}"
                    
            # If no year match, fallback to closest release date if available
            if released_date:
                try:
                    target_date = datetime.strptime(released_date, "%Y-%m-%d")
                    closest = None
                    min_diff = None
                    for result in search_results:
                        if result.get('release_date'):
                            try:
                                result_date = datetime.strptime(result['release_date'], "%Y-%m-%d")
                                diff = abs((result_date - target_date).days)
                                if min_diff is None or diff < min_diff:
                                    min_diff = diff
                                    closest = result
                            except Exception:
                                continue
                    if closest and closest.get('poster_path'):
                        return f"https://image.tmdb.org/t/p/w500{closest['poster_path']}"
                except Exception:
                    pass
                    
            # Fallback: return the first poster if no year match
            if search_results[0].get('poster_path'):
                return f"https://image.tmdb.org/t/p/w500{search_results[0]['poster_path']}"
                
        except requests.exceptions.RequestException as e:
            print(f"Network error while searching for '{movie_title}': {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error fetching poster for '{movie_title}': {str(e)}")
    return None


def film_recommander(titre, data, matrice_score, nombre=5):
    """
    Recommends movies similar to the given title based on cosine similarity.
    """
    lignes = data.index[data['name'].str.lower() == titre.lower()]
    if len(lignes) == 0:
        return [{'titre': 'Sorry ! No similar movie found.', 'poster': None}]

    ligne = lignes[0]
    if ligne >= len(matrice_score):
        return []

    films_similaires = list(enumerate(matrice_score[ligne]))
    recommandations = sorted(films_similaires, key=lambda x: x[1], reverse=True)
    top_films_recommandes = recommandations[1:nombre + 1]

    films_recommandes = []
    for i in range(len(top_films_recommandes)):
        indice_film = top_films_recommandes[i][0]
        if indice_film < len(data):
            titre_film = data.iloc[indice_film]['name']
            poster_url = get_movie_poster(titre_film)
            films_recommandes.append({'titre': titre_film, 'poster': poster_url})
    return films_recommandes


def get_movie_details(movie_title):
    """
    Fetches detailed information about a movie from TMDb based on the movie title.
    """
    try:
        print(f"Searching for movie details: {movie_title}")
        
        # Use direct requests instead of TMDb wrapper
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': tmdb.api_key,
            'query': movie_title,
            'page': 1,
            'language': 'en'
        }
        
        try:
            response = session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json().get('results', [])
            
            if not search_results:
                print(f"No results found for {movie_title}")
                return None
                
            print(f"Found {len(search_results)} results for {movie_title}")
            movie_id = search_results[0]['id']
            print(f"Selected movie ID: {movie_id}")
            
            # Get detailed movie information
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            details_response = session.get(details_url, params={'api_key': tmdb.api_key}, timeout=10)
            details_response.raise_for_status()
            details = details_response.json()
            
            # Get credits information
            credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
            credits_response = session.get(credits_url, params={'api_key': tmdb.api_key}, timeout=10)
            credits_response.raise_for_status()
            credits = credits_response.json()
            
            # Create a simple object to match the expected structure
            class MovieDetails:
                def __init__(self, data, credits_data):
                    self.title = data.get('title')
                    self.release_date = data.get('release_date')
                    self.vote_average = data.get('vote_average')
                    self.overview = data.get('overview')
                    self.poster_path = data.get('poster_path')
                    self.id = data.get('id')
                    self.original_title = data.get('original_title')
                    self.runtime = data.get('runtime')
                    self.genres = [g['name'] for g in data.get('genres', [])]
                    self.tagline = data.get('tagline')
                    self.popularity = data.get('popularity')
                    self.vote_count = data.get('vote_count')
                    self.status = data.get('status')
                    self.budget = data.get('budget')
                    self.revenue = data.get('revenue')
                    
                    # Process credits
                    self.directors = []
                    self.writers = []
                    self.cast = []
                    
                    # Get directors
                    for crew_member in credits_data.get('crew', []):
                        if crew_member.get('job') == 'Director':
                            self.directors.append({
                                'name': crew_member.get('name'),
                                'profile_path': crew_member.get('profile_path')
                            })
                    
                    # Get writers
                    for crew_member in credits_data.get('crew', []):
                        if crew_member.get('job') in ['Screenplay', 'Writer', 'Story']:
                            self.writers.append({
                                'name': crew_member.get('name'),
                                'profile_path': crew_member.get('profile_path')
                            })
                    
                    # Get top 6 cast members
                    for cast_member in credits_data.get('cast', [])[:6]:
                        self.cast.append({
                            'name': cast_member.get('name'),
                            'character': cast_member.get('character'),
                            'profile_path': cast_member.get('profile_path')
                        })
            
            movie_details = MovieDetails(details, credits)
            
            # Log the details we received
            print(f"Movie details received:")
            print(f"- Title: {movie_details.title}")
            print(f"- Release Date: {movie_details.release_date}")
            print(f"- Vote Average: {movie_details.vote_average}")
            print(f"- Overview: {movie_details.overview[:100]}...")  # First 100 chars of overview
            print(f"- Poster Path: {movie_details.poster_path}")
            
            # Ensure we have all required fields
            if not all([movie_details.title, movie_details.release_date, movie_details.vote_average, movie_details.overview]):
                print("Warning: Some required fields are missing")
                
            return movie_details
            
        except requests.exceptions.RequestException as e:
            print(f"Network error while searching for '{movie_title}': {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error fetching details for '{movie_title}': {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    return None


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Renders the index page where users can input a movie title for recommendations.
    """
    return render_template('index.html')


@app.route('/recommendation', methods=['POST'])
def recommendation():
    """
    Handles the movie recommendation logic and renders the recommendations page.
    """
    utilisateur = request.form['titre']
    recommandations = film_recommander(titre=utilisateur, data=df, matrice_score=cosine_sim, nombre=20)
    movie_details = get_movie_details(utilisateur)  # Fetch details for the searched movie

    # Determine if movie details were found
    movie_exists = movie_details is not None
    # Determine if recommendations are empty
    recommendations_empty = len(recommandations) == 0 or recommandations[0][
        'titre'] == 'Sorry ! No similar movie found.'

    return render_template('recommendation.html',
                           recommandations=recommandations,
                           last_query=utilisateur,
                           movie_details=movie_details,
                           movie_exists=movie_exists,
                           recommendations_empty=recommendations_empty)

@app.route('/movie/<title>')
def movie_detail(title):
    """
    Renders the movie details page for a specific movie title.
    """
    title_decoded = unquote(title)
    print(f"Fetching details for movie: {title_decoded}")
    
    # Fetch TMDb details
    details = get_movie_details(title_decoded)
    
    # Fetch local dataset details
    movie_row = df[df['name'].str.lower() == title_decoded.lower()]
    dataset_details = None
    if not movie_row.empty:
        dataset_details = movie_row.iloc[0].to_dict()
        # Convert score to float with one decimal place
        if 'score' in dataset_details:
            try:
                score = float(str(dataset_details['score']).replace(',', '.'))
                dataset_details['score'] = f"{score:.1f}"
            except (ValueError, TypeError):
                pass
        print(f"Found local dataset details for {title_decoded}")
    else:
        print(f"No local dataset details found for {title_decoded}")
    
    # Check if we have the minimum required data
    if not details:
        print(f"Error: No TMDb details found for {title_decoded}")
        return render_template(
            'movie_details.html',
            details=None,
            dataset_details=dataset_details,
            error="Movie details not found in TMDb database"
        )
    
    # Ensure we have all required fields
    if not all([details.title, details.release_date, details.vote_average, details.overview]):
        print(f"Warning: Some required fields are missing for {title_decoded}")
    
    return render_template(
        'movie_details.html',
        details=details,
        dataset_details=dataset_details,
        error=None
    )


@app.route('/genres')
def all_movies_by_genre():
    search_query = request.args.get('search', '').strip()
    page = int(request.args.get('page', 1))
    per_page = 3  # Limit to 3 genres per page

    # Prepare all genres and genre counts
    all_genres = []
    for genre_str in df['genre'].dropna().unique():
        genres = [g.strip() for g in genre_str.split(',')]
        all_genres.extend(genres)
    unique_genres = sorted(list(set(all_genres)))
    genre_counts = {}
    for genre in unique_genres:
        genre_counts[genre] = len(df[df['genre'].str.contains(genre, na=False)])

    # Paginate genres for movie sections
    total_genres = len(unique_genres)
    total_pages = (total_genres + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_genres = unique_genres[start:end]
    has_next = end < total_genres
    has_prev = start > 0

    if search_query:
        if len(search_query) < 3:
            return render_template('genres.html', error="You should input at least 3 characters!", genres=unique_genres, genre_counts=genre_counts, request=request, page=page, total_pages=total_pages, has_next=has_next, has_prev=has_prev)
        # Search for movies by title (case-insensitive, partial match)
        search_df = df[df['name'].str.contains(search_query, case=False, na=False)].copy()
        
        # Check if no movies were found
        if search_df.empty:
            return render_template('genres.html', error=f"No movies found for '{search_query}'. Try a different search term.", genres=unique_genres, genre_counts=genre_counts, request=request, page=page, total_pages=total_pages, has_next=has_next, has_prev=has_prev)
        
        # Sort by year and release date (newest first)
        search_df['released_dt'] = pd.to_datetime(search_df['released'], format='%Y-%m-%d', errors='coerce')
        search_df = search_df.sort_values(by=['year', 'released_dt', 'name'], ascending=[False, False, True])
        # Limit search results to 20 movies for performance
        search_df = search_df.head(20)
        search_results = []
        for _, movie in search_df.iterrows():
            title = movie['name']
            year = movie['year']
            director = movie['director'] if 'director' in movie else None
            released = movie['released'] if 'released' in movie else None
            cache_key = f"{title}_{year}_{director}"
            if cache_key in poster_cache:
                poster_url = poster_cache[cache_key]
            else:
                poster_url = get_movie_poster(title, year, director, released)
                if poster_url:  # Only cache if we got a valid URL
                    poster_cache[cache_key] = poster_url
                    save_poster_cache()  # Save cache after adding new poster
            search_results.append({
                'titre': title,
                'poster': poster_url,
                'year': year,
                'rating': movie['rating']
            })
        return render_template('genres.html', search_results=search_results, genres=unique_genres, genre_counts=genre_counts, request=request, page=page, total_pages=total_pages, has_next=has_next, has_prev=has_prev)

    # Normal genre view
    movies_by_genre = {}
    for genre in paginated_genres:
        movies_in_genre = df[df['genre'].str.contains(genre, na=False)].copy()
        # Add released_dt for sorting
        movies_in_genre['released_dt'] = pd.to_datetime(movies_in_genre['released'], format='%Y-%m-%d', errors='coerce')
        # Sort by year (desc), released_dt (desc), name (asc)
        movies_in_genre = movies_in_genre.sort_values(
            by=['year', 'released_dt', 'name'],
            ascending=[False, False, True]
        )
        movies_in_genre = movies_in_genre.head(12)  # 12 movies per genre
        genre_movies = []
        for _, movie in movies_in_genre.iterrows():
            title = movie['name']
            year = movie['year']
            director = movie['director'] if 'director' in movie else None
            released = movie['released'] if 'released' in movie else None
            cache_key = f"{title}_{year}_{director}"
            if cache_key in poster_cache:
                poster_url = poster_cache[cache_key]
            else:
                poster_url = get_movie_poster(title, year, director, released)
                if poster_url:  # Only cache if we got a valid URL
                    poster_cache[cache_key] = poster_url
                    save_poster_cache()  # Save cache after adding new poster
            genre_movies.append({
                'titre': title,
                'poster': poster_url,
                'year': year,
                'rating': movie['rating']
            })
        movies_by_genre[genre] = genre_movies
    return render_template('genres.html', movies_by_genre=movies_by_genre, genres=unique_genres, genre_counts=genre_counts, request=request, page=page, total_pages=total_pages, has_next=has_next, has_prev=has_prev)

@app.route('/genre/<genre_name>')
def genre_movies(genre_name):
    from flask import request
    genre = unquote(genre_name)
    page = int(request.args.get('page', 1))
    per_page = 24
    movies_in_genre = df[df['genre'].str.contains(genre, na=False)].copy()

    # Try to parse the 'released' column to datetime, fallback to NaT if parsing fails
    movies_in_genre['released_dt'] = pd.to_datetime(movies_in_genre['released'], format='%Y-%m-%d', errors='coerce')
    # Sort by year (desc), then by released_dt (desc), then by name (asc)
    movies_in_genre = movies_in_genre.sort_values(
        by=['year', 'released_dt', 'name'],
        ascending=[False, False, True]
    )

    total_movies = len(movies_in_genre)
    total_pages = (total_movies + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    movies_in_genre = movies_in_genre.iloc[start:end]
    genre_movies = []
    for _, movie in movies_in_genre.iterrows():
        title = movie['name']
        year = movie['year']
        director = movie['director'] if 'director' in movie else None
        released = movie['released'] if 'released' in movie else None
        cache_key = f"{title}_{year}_{director}"
        if cache_key in poster_cache:
            poster_url = poster_cache[cache_key]
        else:
            poster_url = get_movie_poster(title, year, director, released)
            if poster_url:  # Only cache if we got a valid URL
                poster_cache[cache_key] = poster_url
                save_poster_cache()  # Save cache after adding new poster
        genre_movies.append({
            'titre': title,
            'poster': poster_url,
            'year': year,
            'rating': movie['rating']
        })
    has_next = end < total_movies
    has_prev = start > 0
    return render_template(
        'genre_movies.html',
        genre=genre,
        movies=genre_movies,
        page=page,
        has_next=has_next,
        has_prev=has_prev,
        total_pages=total_pages
    )

def convert_minutes_to_hours_minutes(minutes):
    """
    Converts minutes to hours and minutes format (e.g., 125 -> 2h 5m)
    """
    if not minutes:
        return None
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if hours > 0:
        return f"{hours}h {remaining_minutes}m"
    return f"{remaining_minutes}m"

@app.route('/top-movies')
def top_movies():
    top_movies = df.copy()

    # Convert score to float
    def convert_score(x):
        try:
            x_str = str(x).strip().replace(',', '.')
            return float(x_str)
        except (ValueError, TypeError):
            return None

    top_movies['score'] = top_movies['score'].apply(convert_score)
    top_movies = top_movies.dropna(subset=['score'])

    # 1. Sort by score and take top 100 first
    top_movies = top_movies.sort_values(
        by=['score'],
        ascending=[False]
    ).head(100)

    # 2. Now apply filters to these 100
    from flask import request
    genre = request.args.get('genre', '').strip()
    year_from = request.args.get('year_from', type=int)
    year_to = request.args.get('year_to', type=int)
    rating_from = request.args.get('rating_from', type=float)
    rating_to = request.args.get('rating_to', type=float)
    votes_from = request.args.get('votes_from', type=int)
    votes_to = request.args.get('votes_to', type=int)

    if genre:
        top_movies = top_movies[top_movies['genre'].str.contains(genre, na=False)]
    if year_from:
        top_movies = top_movies[top_movies['year'] >= year_from]
    if year_to:
        top_movies = top_movies[top_movies['year'] <= year_to]
    if rating_from:
        top_movies = top_movies[top_movies['score'] >= rating_from]
    if rating_to:
        top_movies = top_movies[top_movies['score'] <= rating_to]
    if votes_from:
        top_movies = top_movies[top_movies['votes'] >= votes_from]
    if votes_to:
        top_movies = top_movies[top_movies['votes'] <= votes_to]

    # Calculate genre counts for the filtered top 100 movies
    genre_counts = {}
    for genres in top_movies['genre'].dropna():
        for genre_item in genres.split(','):
            genre_item = genre_item.strip()
            genre_counts[genre_item] = genre_counts.get(genre_item, 0) + 1

    # Process top movies
    top_results = []
    for _, movie_data in top_movies.iterrows():
        title = movie_data['name']
        year = movie_data['year']
        director = movie_data['director'] if 'director' in movie_data else None
        released = movie_data['released'] if 'released' in movie_data else None
        cache_key = f"{title}_{year}_{director}"
        
        # Get poster URL
        if cache_key in poster_cache:
            poster_url = poster_cache[cache_key]
        else:
            poster_url = get_movie_poster(title, year, director, released)
            if poster_url:  # Only cache if we got a valid URL
                poster_cache[cache_key] = poster_url
                save_poster_cache()  # Save cache after adding new poster
            
        # Remove duration fetching to speed up loading
        # Duration will be fetched only when viewing individual movie details
            
        top_results.append({
            'titre': title,
            'poster': poster_url,
            'year': year,
            'rating': f"{movie_data['score']:.1f}",  # Format to exactly 1 decimal place
            'votes': movie_data['votes'],
            'content_rating': movie_data['rating'],  # Get content rating from the dataset
            'duration': None,  # Set to None instead of fetching
            'genres': movie_data['genre'].split(',') if pd.notna(movie_data['genre']) else []  # Add genres
        })

    return render_template('top_movies.html', movies=top_results, genre_counts=genre_counts)

def calculate_trending_score(movie_data):
    """
    Calculate a trending score based on recency, popularity, and quality.
    """
    # Get current year
    current_year = pd.Timestamp.now().year
    
    # Recency score (0-1): More recent movies get higher scores
    year_diff = current_year - movie_data['year']
    recency_score = 1 / (1 + year_diff)  # Decay factor
    
    # Popularity score (0-1): Based on number of votes
    max_votes = df['votes'].max()
    popularity_score = movie_data['votes'] / max_votes if max_votes > 0 else 0
    
    # Quality score (0-1): Based on rating
    quality_score = float(str(movie_data['score']).replace(',', '.')) / 10
    
    # Weighted combination
    trending_score = (
        0.4 * recency_score +  # 40% weight to recency
        0.4 * popularity_score +  # 40% weight to popularity
        0.2 * quality_score  # 20% weight to quality
    )
    
    return trending_score

@app.route('/trending')
def trending_movies():
    # Get all movies
    trending_movies = df.copy()
    
    # Calculate trending score for each movie
    trending_movies['trending_score'] = trending_movies.apply(calculate_trending_score, axis=1)
    
    # Sort by trending score (highest first)
    trending_movies = trending_movies.sort_values(
        by=['trending_score'],
        ascending=[False]
    ).head(100)

    # Apply filters to these 100 movies
    genre = request.args.get('genre', '').strip()
    year_from = request.args.get('year_from', type=int)
    year_to = request.args.get('year_to', type=int)
    rating_from = request.args.get('rating_from', type=float)
    rating_to = request.args.get('rating_to', type=float)
    votes_from = request.args.get('votes_from', type=int)
    votes_to = request.args.get('votes_to', type=int)

    if genre:
        trending_movies = trending_movies[trending_movies['genre'].str.contains(genre, na=False)]
    if year_from:
        trending_movies = trending_movies[trending_movies['year'] >= year_from]
    if year_to:
        trending_movies = trending_movies[trending_movies['year'] <= year_to]
    if rating_from:
        trending_movies = trending_movies[trending_movies['score'] >= rating_from]
    if rating_to:
        trending_movies = trending_movies[trending_movies['score'] <= rating_to]
    if votes_from:
        trending_movies = trending_movies[trending_movies['votes'] >= votes_from]
    if votes_to:
        trending_movies = trending_movies[trending_movies['votes'] <= votes_to]

    # Calculate genre counts for the filtered trending movies
    genre_counts = {}
    for genres in trending_movies['genre'].dropna():
        for genre_item in genres.split(','):
            genre_item = genre_item.strip()
            genre_counts[genre_item] = genre_counts.get(genre_item, 0) + 1

    # Load previous positions
    previous_positions = load_previous_positions()
    
    # Create current positions dictionary
    current_positions = {}
    for idx, movie_data in trending_movies.iterrows():
        current_positions[movie_data['name']] = idx + 1  # +1 because we want 1-based indexing
    
    # Save current positions for next time
    save_current_positions(current_positions)

    # Process trending movies
    trending_results = []
    for idx, movie_data in trending_movies.iterrows():
        title = movie_data['name']
        year = movie_data['year']
        director = movie_data['director'] if 'director' in movie_data else None
        released = movie_data['released'] if 'released' in movie_data else None
        cache_key = f"{title}_{year}_{director}"
        
        # Calculate position change
        current_pos = idx + 1
        previous_pos = previous_positions.get(title)
        position_change = None if previous_pos is None else previous_pos - current_pos
        
        # Get poster URL
        if cache_key in poster_cache:
            poster_url = poster_cache[cache_key]
        else:
            poster_url = get_movie_poster(title, year, director, released)
            if poster_url:  # Only cache if we got a valid URL
                poster_cache[cache_key] = poster_url
                save_poster_cache()  # Save cache after adding new poster
            
        # Remove duration fetching to speed up loading
        # Duration will be fetched only when viewing individual movie details
            
        trending_results.append({
            'titre': title,
            'poster': poster_url,
            'year': year,
            'rating': f"{movie_data['score']:.1f}",
            'votes': movie_data['votes'],
            'content_rating': movie_data['rating'],
            'duration': None,  # Set to None instead of fetching
            'trending_score': f"{movie_data['trending_score']:.2f}",
            'position_change': position_change,
            'genres': movie_data['genre'].split(',') if pd.notna(movie_data['genre']) else []  # Add genres
        })

    return render_template('trending_movies.html', movies=trending_results, genre_counts=genre_counts, request=request)

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html')

@app.route('/auth_page')
def auth_page():
    """
    Renders the authentication page for login and registration.
    """
    return render_template('auth.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/watchlist')
def watchlist():
    return render_template('watchlist.html')

@app.route('/api/movie/<int:movie_id>')
def get_movie_by_id(movie_id):
    try:
        movie = Movie()
        movie_details = movie.details(movie_id)
        
        return jsonify({
            'id': movie_details.id,
            'title': movie_details.title,
            'poster_path': movie_details.poster_path,
            'backdrop_path': movie_details.backdrop_path,
            'release_date': movie_details.release_date,
            'vote_average': movie_details.vote_average,
            'overview': movie_details.overview,
            'runtime': movie_details.runtime
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-firebase')
def test_firebase():
    """Test Firebase connection and data access"""
    try:
        # Test basic connection
        favorites_ref = db.collection('favorites')
        docs = list(favorites_ref.limit(1).stream())
        
        result = {
            'firebase_connected': True,
            'favorites_collection_accessible': True,
            'sample_docs_found': len(docs),
            'dataset_loaded': len(df) if 'df' in globals() else 0
        }
        
        if docs:
            sample_data = docs[0].to_dict()
            result['sample_favorites'] = len(sample_data.get('movies', []))
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'firebase_connected': False,
            'error': str(e)
        }), 500

@app.route('/api/movie_by_title')
def api_movie_by_title():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'Missing title'}), 400
    details = get_movie_details(title)
    if not details:
        return jsonify({'error': 'Movie not found'}), 404
    return jsonify({
        'title': details.title,
        'poster_path': details.poster_path,
        'release_date': details.release_date,
        'vote_average': details.vote_average,
        'overview': details.overview,
        'id': details.id
    })

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        print(f"=== Recommendations API called ===")
        
        # Check if dataset is loaded
        if 'df' not in globals():
            print("❌ Dataset not loaded!")
            return jsonify({'error': 'Dataset not loaded', 'recommendations': []}), 500
        
        print(f"✅ Dataset available: {len(df)} movies")
        
        # Get the request data
        request_data = request.get_json()
        if not request_data:
            print("❌ No JSON data in request")
            return jsonify({'error': 'No JSON data provided', 'recommendations': []}), 400
        
        id_token = request_data.get('idToken')
        if not id_token:
            print("❌ No idToken provided")
            return jsonify({'error': 'No idToken provided', 'recommendations': []}), 400
        
        print(f"🔑 ID Token provided: {id_token[:20]}...")
        
        try:
            decoded_token = firebase_auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            print(f"✅ User authenticated: {uid}")
        except Exception as auth_error:
            print(f"❌ Authentication error: {auth_error}")
            return jsonify({'error': 'Invalid authentication token', 'recommendations': []}), 401
        
        # Get favorite movies from Firestore favorites collection
        try:
            favorites_ref = db.collection('favorites').document(uid)
            favorites_doc = favorites_ref.get()
            
            print(f"Favorites doc exists: {favorites_doc.exists}")
            if favorites_doc.exists:
                data = favorites_doc.to_dict()
                print(f"Favorites data: {data}")
            
            if not favorites_doc.exists or 'movies' not in favorites_doc.to_dict() or not favorites_doc.to_dict()['movies']:
                print("No favorites found - returning empty recommendations")
                return jsonify({'recommendations': []})
            
            favorite_titles = favorites_doc.to_dict()['movies']
            print(f"Found {len(favorite_titles)} favorite movies for user {uid}")
            
        except Exception as firestore_error:
            print(f"❌ Firestore error: {firestore_error}")
            return jsonify({'error': 'Database error', 'recommendations': []}), 500
        
        # Get watched movies to exclude from recommendations
        watched_ref = db.collection('watchlists').document(uid)
        watched_doc = watched_ref.get()
        watched_ids = set()
        if watched_doc.exists and 'watched' in watched_doc.to_dict():
            watched_ids = set(str(mid) for mid in watched_doc.to_dict()['watched'])
        
        # Find favorite movies in dataset
        favorite_movies = df[df['name'].str.lower().isin([title.lower() for title in favorite_titles])]
        
        if favorite_movies.empty:
            print("No favorite movies found in dataset")
            return jsonify({'recommendations': []})
        
        print(f"Found {len(favorite_movies)} favorite movies in dataset")
        
        # Analyze favorite movies to extract preferences
        liked_genres = set()
        liked_directors = set()
        liked_stars = set()
        liked_countries = set()
        total_score = 0
        count = 0
        
        for _, row in favorite_movies.iterrows():
            # Extract genres
            if row['genre'] is not None and str(row['genre']).strip() and str(row['genre']) != 'nan':
                genres = [g.strip() for g in str(row['genre']).split(',') if g.strip()]
                liked_genres.update(genres)
            
            # Extract directors
            if row['director'] is not None and str(row['director']).strip() and str(row['director']) != 'nan':
                directors = [d.strip() for d in str(row['director']).split(',') if d.strip()]
                liked_directors.update(directors)
            
            # Extract stars
            if row['star'] is not None and str(row['star']).strip() and str(row['star']) != 'nan':
                stars = [s.strip() for s in str(row['star']).split(',') if s.strip()]
                liked_stars.update(stars)
            
            # Extract countries
            if row['country'] is not None and str(row['country']).strip() and str(row['country']) != 'nan':
                countries = [c.strip() for c in str(row['country']).split(',') if c.strip()]
                liked_countries.update(countries)
            
            # Calculate average score preference
            if row['score'] is not None and str(row['score']).strip() and str(row['score']) != 'nan':
                try:
                    score = float(str(row['score']).replace(',', '.'))
                    total_score += score
                    count += 1
                except (ValueError, TypeError):
                    pass
        
        avg_liked_score = total_score / count if count > 0 else 0
        print(f"User preferences - Genres: {len(liked_genres)}, Directors: {len(liked_directors)}, Stars: {len(liked_stars)}")
        
        # Content-based scoring function
        def calculate_recommendation_score(row):
            score = 0
            
            # Genre matching (highest weight)
            if row['genre'] is not None and str(row['genre']).strip() and str(row['genre']) != 'nan':
                movie_genres = set([g.strip() for g in str(row['genre']).split(',') if g.strip()])
                genre_overlap = len(movie_genres & liked_genres)
                score += genre_overlap * 4
            
            # Director matching
            if row['director'] is not None and str(row['director']).strip() and str(row['director']) != 'nan':
                movie_directors = set([d.strip() for d in str(row['director']).split(',') if d.strip()])
                director_overlap = len(movie_directors & liked_directors)
                score += director_overlap * 3
            
            # Star matching
            if row['star'] is not None and str(row['star']).strip() and str(row['star']) != 'nan':
                movie_stars = set([s.strip() for s in str(row['star']).split(',') if s.strip()])
                star_overlap = len(movie_stars & liked_stars)
                score += star_overlap * 2
            
            # Country matching
            if row['country'] is not None and str(row['country']).strip() and str(row['country']) != 'nan':
                movie_countries = set([c.strip() for c in str(row['country']).split(',') if c.strip()])
                country_overlap = len(movie_countries & liked_countries)
                score += country_overlap * 1
            
            # Score similarity bonus
            if row['score'] is not None and str(row['score']).strip() and str(row['score']) != 'nan' and avg_liked_score > 0:
                try:
                    movie_score = float(str(row['score']).replace(',', '.'))
                    # Bonus for movies with similar or higher scores
                    if movie_score >= avg_liked_score - 0.5:
                        score += 2
                    if movie_score >= avg_liked_score:
                        score += 1
                except (ValueError, TypeError):
                    pass
            
            # Popularity bonus (for movies with good vote counts)
            if row['votes'] is not None and str(row['votes']).strip() and str(row['votes']) != 'nan':
                try:
                    votes = int(row['votes'])
                    if votes > 10000:  # Popular movies get small bonus
                        score += 0.5
                except (ValueError, TypeError):
                    pass
            
            return score
        
        # Filter out already favorited movies and watched movies
        candidate_movies = df[
            ~df['name'].str.lower().isin([title.lower() for title in favorite_titles])
        ].copy()
        
        # Calculate recommendation scores
        candidate_movies['rec_score'] = candidate_movies.apply(calculate_recommendation_score, axis=1)
        
        # Get top recommendations (only movies with score > 0)
        top_recommendations = candidate_movies[
            candidate_movies['rec_score'] > 0
        ].sort_values(['rec_score', 'score'], ascending=[False, False]).head(12)
        
        print(f"Generated {len(top_recommendations)} recommendations")
        
        # Prepare response with poster URLs
        recommendations = []
        for _, row in top_recommendations.iterrows():
            # Get poster URL
            cache_key = f"{row['name']}_{row.get('year', '')}_{row.get('director', '')}"
            poster_url = poster_cache.get(cache_key)
            
            if not poster_url:
                poster_url = get_movie_poster(
                    row['name'], 
                    row.get('year'), 
                    row.get('director'), 
                    row.get('released')
                )
                if poster_url:
                    poster_cache[cache_key] = poster_url
                    save_poster_cache()
            
            # Format score safely
            score_formatted = 'N/A'
            if row['score'] is not None and str(row['score']).strip() and str(row['score']) != 'nan':
                try:
                    score_formatted = f"{float(str(row['score']).replace(',', '.')):.1f}"
                except (ValueError, TypeError):
                    score_formatted = 'N/A'
            
            recommendations.append({
                'title': row['name'],
                'year': row.get('year', ''),
                'genres': row.get('genre', ''),
                'score': score_formatted,
                'director': row.get('director', ''),
                'poster_url': poster_url or '/static/favicon.ico',
                'rec_score': f"{row['rec_score']:.1f}",
                'votes': row.get('votes', 0)
            })
        
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        print(f"❌ Error in get_recommendations: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        # Return more detailed error for debugging
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'recommendations': []
        }), 500

@app.route('/firebase-config.js')
def firebase_config_js():
    firebase_config = {
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID")
    }
    js_content = f"const firebaseConfig = {json.dumps(firebase_config)};"
    return Response(js_content, mimetype='application/javascript')

if __name__ == '__main__':
    app.run(debug=True)