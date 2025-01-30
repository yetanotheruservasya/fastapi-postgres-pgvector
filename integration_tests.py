import requests
import json
import os

# Константы
API_URL = "http://44.210.89.103:8000"
TEST_FILE = "linkedin_profile.json"  # Имя JSON-файла с тестовыми данными
USERNAME = "admin"
PASSWORD = "password"

def get_token():
    """ Получаем JWT-токен """
    url = f"{API_URL}/token"
    data = {"username": USERNAME, "password": PASSWORD}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(url, data=data, headers=headers)
    if response.status_code == 200:
        token = response.json().get("access_token")
        print(f"✅ Токен получен: {token}\n")
        return token
    else:
        print(f"❌ Ошибка получения токена: {response.text}")
        return None

def load_test_data():
    """ Загружаем тестовые данные из JSON-файла """
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def store_data(token, data):
    """ Отправляем данные в API (сохранение в базу) """
    url = f"{API_URL}/store"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    for item in data:
        payload = {
            "source": "linkedin",
            "company_id": item["id"],
            "data": item
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print(f"✅ Данные успешно сохранены: {item['id']}\n")
        else:
            print(f"❌ Ошибка сохранения данных: {response.text}")

def normalize_data(token, company_id):
    """ Вызывает API нормализации """
    url = f"{API_URL}/normalize/{company_id}"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        print(f"✅ Данные нормализованы для: {company_id}\n")
    else:
        print(f"❌ Ошибка нормализации: {response.text}")

def search_data(token, query):
    """ Выполняем поиск по базе """
    url = f"{API_URL}/search?query={query}"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)
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
