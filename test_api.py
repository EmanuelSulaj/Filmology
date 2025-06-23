import requests
import json

def test_recommendations_api():
    print("=== Testing Recommendations API ===")
    
    # Test the Firebase test endpoint first
    try:
        response = requests.get('http://127.0.0.1:5000/api/test-firebase')
        print(f"Firebase test status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Firebase test result: {data}")
        else:
            print(f"Firebase test error: {response.text}")
    except Exception as e:
        print(f"Error testing Firebase: {e}")
    
    print("\n" + "="*50)
    print("Please check your Flask terminal for detailed error messages")
    print("Look for lines starting with ❌ or 'Error in get_recommendations'")
    print("="*50)

if __name__ == "__main__":
    test_recommendations_api() 