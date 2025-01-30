"""
Integration tests for the company database API.
Includes functions for obtaining a token, 
loading test data, storing data, normalizing data, and searching data.
"""
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
API_URL = os.getenv('API_URL')
TEST_FILE = os.getenv('TEST_FILE')
USERNAME = os.getenv('API_USERNAME')
PASSWORD = os.getenv('API_PASSWORD')

# Validate required environment variables
if not all([API_URL, TEST_FILE, USERNAME, PASSWORD]):
    raise ValueError("Missing required environment variables. Please check .env file.")

def get_token():
    """ Получаем JWT-токен """
    url = f"{API_URL}/token"
    credentials = {"username": USERNAME, "password": PASSWORD}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(url, data=credentials, headers=headers, timeout=60)
    if response.status_code == 200:
        auth_token = response.json().get("access_token")
        print(f"✅ Токен получен: {auth_token}\n")
        return auth_token
    else:
        print(f"❌ Ошибка получения токена: {response.text}")
        return None

def load_test_data():
    """ Загружаем тестовые данные из JSON-файла """
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def store_data(auth_token, companies_data):
    """ Отправляем данные в API (сохранение в базу) """
    url = f"{API_URL}/store"
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

    for company in companies_data:
        payload = {
            "source": "linkedin",
            "company_id": company["id"],
            "data": company
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            print(f"✅ Данные успешно сохранены: {company['id']}\n")
        else:
            print(f"❌ Ошибка сохранения данных: {response.text}")

def normalize_data(auth_token, company_id):
    """ Вызывает API нормализации """
    url = f"{API_URL}/normalize/{company_id}"
    headers = {"Authorization": f"Bearer {auth_token}"}

    response = requests.post(url, headers=headers, timeout=60)
    if response.status_code == 200:
        print(f"✅ Данные нормализованы для: {company_id}\n")
    else:
        print(f"❌ Ошибка нормализации: {response.text}")

def search_data(auth_token, query):
    """ Выполняем поиск по базе """
    url = f"{API_URL}/search?query={query}"
    headers = {"Authorization": f"Bearer {auth_token}"}

    response = requests.get(url, headers=headers, timeout=60)
    if response.status_code == 200:
        print(f"✅ Результаты поиска: {response.json()}\n")
    else:
        print(f"❌ Ошибка поиска: {response.text}")

if __name__ == "__main__":
    token = get_token()
    if token:
        data = load_test_data()
        store_data(token, data)

        # Вызываем нормализацию для всех загруженных данных
        for item in data:
            normalize_data(token, item["id"])

        # Выполняем поиск
        search_data(token, "Head of Product")
