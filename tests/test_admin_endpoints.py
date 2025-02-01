import os
import time
import logging
import requests
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
API_URL = os.getenv("API_URL")
API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")

def get_auth_token():
    """Get authentication token for API requests"""
    auth_response = requests.post(
        f"{API_URL}/token",
        data={"username": API_USERNAME, "password": API_PASSWORD}
    )
    if auth_response.status_code != 200:
        raise Exception("Authentication failed")
    return auth_response.json()["access_token"]

def make_authenticated_request(endpoint, method="POST"):
    """Make an authenticated request to the specified endpoint"""
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    if method == "POST":
        response = requests.post(f"{API_URL}{endpoint}", headers=headers)
    elif method == "DELETE":
        response = requests.delete(f"{API_URL}{endpoint}", headers=headers)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")
    
    return response

def test_database_operations():
    """Test database creation and deletion"""
    logger.info("Testing database operations...")
    
    # Create database
    response = make_authenticated_request("/admin/create_database")
    assert response.status_code == 200
    logger.info("Database created successfully")
    
    # Delete database
    response = make_authenticated_request("/admin/delete_database", method="DELETE")
    assert response.status_code == 200
    logger.info("Database deleted successfully")
    
    # Recreate database for further tests
    response = make_authenticated_request("/admin/create_database")
    assert response.status_code == 200
    logger.info("Database recreated successfully")

def test_table_operations():
    """Test table creation and deletion"""
    logger.info("Testing table operations...")
    
    # Create tables
    response = make_authenticated_request("/admin/create_tables")
    assert response.status_code == 200
    logger.info("Tables created successfully")
    
    # Delete tables
    response = make_authenticated_request("/admin/delete_tables", method="DELETE")
    assert response.status_code == 200
    logger.info("Tables deleted successfully")
    
    # Recreate tables for further tests
    response = make_authenticated_request("/admin/create_tables")
    assert response.status_code == 200
    logger.info("Tables recreated successfully")

def test_data_operations():
    """Test data deletion"""
    logger.info("Testing data operations...")
    
    # Delete all data
    response = make_authenticated_request("/admin/delete_data", method="DELETE")
    assert response.status_code == 200
    logger.info("Data deleted successfully")

def run_all_tests():
    """Run all administrative endpoint tests"""
    try:
        # Add delay between operations to avoid potential race conditions
        test_database_operations()
        time.sleep(2)
        
        test_table_operations()
        time.sleep(2)
        
        test_data_operations()
        logger.info("All administrative tests completed successfully!")
        
    except AssertionError as e:
        logger.error("Test failed: %s", str(e))
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))

if __name__ == "__main__":
    run_all_tests()
