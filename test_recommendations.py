import requests
import json

def test_recommendations_endpoint():
    """Test the recommendations endpoint to see what's happening"""
    
    # Test the endpoint without authentication first
    print("1. Testing recommendations endpoint without auth...")
    try:
        response = requests.post('http://localhost:5000/api/recommendations', 
                               json={'idToken': 'invalid_token'})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test if the Flask app is running
    print("\n2. Testing if Flask app is running...")
    try:
        response = requests.get('http://localhost:5000/')
        print(f"Status: {response.status_code}")
        print("Flask app is running!")
    except Exception as e:
        print(f"Error: {e}")
        print("Flask app is not running. Please start it with: python app.py")

if __name__ == "__main__":
    test_recommendations_endpoint() 