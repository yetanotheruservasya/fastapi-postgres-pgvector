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
load_dotenv(dotenv_path=".env")

# Get configuration from environment variables
API_URL = os.getenv('API_URL')
TEST_FILE = os.getenv('TEST_FILE')
USERNAME = os.getenv('API_USERNAME')
PASSWORD = os.getenv('API_PASSWORD')

# Validate required environment variables
if not all([API_URL, TEST_FILE, USERNAME, PASSWORD]):
    raise ValueError("Missing required environment variables. Please check .env file.")

# Используем сессию для повторного использования HTTP-соединений
session = requests.Session()

def get_token():
    """ Получаем JWT-токен """
    url = f"{API_URL}/token"
    credentials = {"username": USERNAME, "password": PASSWORD}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = session.post(url, data=credentials, headers=headers, timeout=60)
    # Если статус не 200, выбрасываем исключение
    response.raise_for_status()
    auth_token = response.json().get("access_token")
    if not auth_token:
        raise ValueError("Token not found in the response.")
    print(f"✅ Токен получен: {auth_token}\n")
    return auth_token

def load_test_data():
    """ Загружаем тестовые данные из JSON-файла """
    print(f"Загрузка тестовых данных из файла: {TEST_FILE}\n")
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def store_data(auth_token, companies_data):
    """ Отправляем данные в API (сохранение в базу) """
    url = f"{API_URL}/store"
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

    for company in companies_data:
        company_id = company.get("id")
        if not company_id:
            print("❌ Пропущена запись без поля 'id'")
            continue

        payload = {
            "source": "linkedin",
            "company_id": company_id,
            "data": company
        }

        response = session.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            print(f"✅ Данные успешно сохранены для компании: {company_id}\n")
        else:
            print(f"❌ Ошибка сохранения данных для компании {company_id}: {response.text}")

def normalize_data(auth_token, company_id):
    """ Вызывает API нормализации """
    url = f"{API_URL}/normalize/{company_id}"
    headers = {"Authorization": f"Bearer {auth_token}"}

    response = session.post(url, headers=headers, timeout=60)
    if response.status_code == 200:
        print(f"✅ Данные нормализованы для компании: {company_id}\n")
    else:
        print(f"❌ Ошибка нормализации для компании {company_id}: {response.text}")

def search_data(auth_token, query):
    """ Выполняем поиск по базе """
    url = f"{API_URL}/search"
    headers = {"Authorization": f"Bearer {auth_token}"}
    params = {"query": query}  # Используем параметр для корректного формирования URL

    response = session.get(url, headers=headers, params=params, timeout=60)
    if response.status_code == 200:
        results = response.json()
        print(f"✅ Результаты поиска:\n{json.dumps(results, ensure_ascii=False, indent=2)}\n")
    else:
        print(f"❌ Ошибка поиска: {response.text}")

if __name__ == "__main__":
    try:
        token = get_token()
    except Exception as e:
        print(f"❌ Не удалось получить токен: {e}")
        exit(1)

    data = load_test_data()
    store_data(token, data)

    # Вызываем нормализацию для всех загруженных компаний
    for company in data:
        company_id = company.get("id")
        if company_id:
            normalize_data(token, company_id)
        else:
            print("❌ Пропущена нормализация для записи без 'id'")

    # Выполняем поиск по базе
    search_data(token, "Head of Product")
