import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path to import app functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions to test
from app import film_recommander, convert_minutes_to_hours_minutes, check_rate_limit

class TestAppFunctions:
    """Simple unit tests for app.py core functions"""
    
    def test_convert_minutes_to_hours_minutes(self):
        """Test the minutes to hours conversion function"""
        assert convert_minutes_to_hours_minutes(90) == "1h 30m"
        assert convert_minutes_to_hours_minutes(60) == "1h 0m"
        assert convert_minutes_to_hours_minutes(45) == "0h 45m"
        assert convert_minutes_to_hours_minutes(120) == "2h 0m"
        assert convert_minutes_to_hours_minutes(0) == "0h 0m"
    
    def test_check_rate_limit(self):
        """Test that rate limit function returns True"""
        result = check_rate_limit()
        assert result is True
    
    def test_film_recommander_with_sample_data(self):
        """Test the recommendation function with sample data"""
        # Create sample movie data
        sample_data = pd.DataFrame({
            'name': ['Movie A', 'Movie B', 'Movie C'],
            'genre': ['Action', 'Drama', 'Comedy'],
            'director': ['Director A', 'Director B', 'Director C'],
            'writer': ['Writer A', 'Writer B', 'Writer C'],
            'star': ['Star A', 'Star B', 'Star C'],
            'company': ['Company A', 'Company B', 'Company C']
        })
        
        # Create simple similarity matrix
        similarity_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        
        # Test recommendation
        recommendations = film_recommander(
            titre="Movie A",
            data=sample_data,
            matrice_score=similarity_matrix,
            nombre=2
        )
        
        # Should return 2 recommendations
        assert len(recommendations) == 2
        
        # Each recommendation should have required keys
        for rec in recommendations:
            assert 'titre' in rec
            assert 'poster' in rec
    
    def test_film_recommander_non_existent_movie(self):
        """Test recommendation function with non-existent movie"""
        sample_data = pd.DataFrame({
            'name': ['Movie A', 'Movie B'],
            'genre': ['Action', 'Drama'],
            'director': ['Director A', 'Director B'],
            'writer': ['Writer A', 'Writer B'],
            'star': ['Star A', 'Star B'],
            'company': ['Company A', 'Company B']
        })
        
        similarity_matrix = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ])
        
        recommendations = film_recommander(
            titre="Non Existent Movie",
            data=sample_data,
            matrice_score=similarity_matrix,
            nombre=5
        )
        
        # Should return error message
        assert len(recommendations) == 1
        assert recommendations[0]['titre'] == 'Sorry ! No similar movie found.'
    
    def test_data_file_exists(self):
        """Test that the required data file exists"""
        csv_path = "dataset/movies2.csv"
        assert os.path.exists(csv_path), "movies2.csv file should exist"
    
    def test_csv_can_be_loaded(self):
        """Test that the CSV file can be loaded"""
        csv_path = "dataset/movies2.csv"
        try:
            df = pd.read_csv(csv_path, delimiter=";", encoding='ISO-8859-1')
            assert len(df) > 0, "CSV should contain data"
            assert 'name' in df.columns, "CSV should have 'name' column"
        except Exception as e:
            pytest.fail(f"Failed to load CSV: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 