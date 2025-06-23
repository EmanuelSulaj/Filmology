import requests
import time

def test_page_load_time(url, page_name):
    """Test how long it takes to load a page"""
    print(f"Testing {page_name}...")
    start_time = time.time()
    
    try:
        response = requests.get(url, timeout=30)
        end_time = time.time()
        load_time = end_time - start_time
        
        if response.status_code == 200:
            print(f"✅ {page_name} loaded successfully in {load_time:.2f} seconds")
            return load_time
        else:
            print(f"❌ {page_name} failed with status code {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error loading {page_name}: {str(e)}")
        return None

def main():
    base_url = "http://localhost:5000"
    
    print("=== Performance Test Results ===")
    print("Testing page load times after removing duration fetching...")
    print()
    
    # Test top movies page
    top_movies_time = test_page_load_time(f"{base_url}/top-movies", "Top Movies")
    
    print()
    
    # Test trending movies page
    trending_time = test_page_load_time(f"{base_url}/trending", "Trending Movies")
    
    print()
    print("=== Summary ===")
    if top_movies_time:
        print(f"Top Movies: {top_movies_time:.2f} seconds")
    if trending_time:
        print(f"Trending Movies: {trending_time:.2f} seconds")
    
    print()
    print("Note: Duration information is now set to 'None' and will only be")
    print("fetched when viewing individual movie details for better performance.")

if __name__ == "__main__":
    main() 