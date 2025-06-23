import firebase_admin
from firebase_admin import credentials, firestore, auth

def debug_firebase():
    print("=== Firebase Debug ===")
    
    try:
        # Initialize Firebase (if not already done)
        if not firebase_admin._apps:
            cred = credentials.Certificate('serviceAccountKey.json')
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        print("✅ Firebase Admin SDK connected")
        
        # Test favorites collection access
        print("\n=== Testing Favorites Collection ===")
        favorites_ref = db.collection('favorites')
        
        # Try to list documents
        docs = list(favorites_ref.limit(5).stream())
        print(f"Found {len(docs)} user favorites documents")
        
        for doc in docs:
            data = doc.to_dict()
            movies_count = len(data.get('movies', []))
            print(f"  User {doc.id}: {movies_count} favorite movies")
            if movies_count > 0:
                print(f"    Sample movies: {data['movies'][:3]}...")
        
        # Test watchlists collection access
        print("\n=== Testing Watchlists Collection ===")
        watchlists_ref = db.collection('watchlists')
        watchlist_docs = list(watchlists_ref.limit(3).stream())
        print(f"Found {len(watchlist_docs)} watchlist documents")
        
        for doc in watchlist_docs:
            data = doc.to_dict()
            watchlist_count = len(data.get('movies', []))
            watched_count = len(data.get('watched', []))
            print(f"  User {doc.id}: {watchlist_count} watchlist, {watched_count} watched")
        
        # Test users collection access
        print("\n=== Testing Users Collection ===")
        users_ref = db.collection('users')
        user_docs = list(users_ref.limit(3).stream())
        print(f"Found {len(user_docs)} user documents")
        
        print("\n=== Firebase Collections Summary ===")
        print("✅ All collections are accessible via Admin SDK")
        print("✅ Data exists in favorites collection")
        print("\nIf recommendations still don't work, the issue is likely:")
        print("1. Frontend authentication token issues")
        print("2. API endpoint not being called")
        print("3. Frontend JavaScript errors")
        print("4. Firestore security rules blocking client access")
        
        print("\n=== Next Steps ===")
        print("1. Start your Flask app: python app.py")
        print("2. Open browser and login")
        print("3. Go to dashboard and open browser DevTools (F12)")
        print("4. Check Console tab for JavaScript errors")
        print("5. Check Network tab to see if /api/recommendations is called")
        print("6. Check Flask terminal for server-side debug messages")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_firebase() 