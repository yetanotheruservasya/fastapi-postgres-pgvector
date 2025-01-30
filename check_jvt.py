import requests

# Адрес API (замени на свой)
API_URL = "http://44.210.89.103:8000"

# Данные для авторизации
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

def test_protected_request(token):
    """ Делаем запрос к защищённому эндпоинту """
    url = f"{API_URL}/search?query=test"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print(f"✅ Запрос успешен! Ответ:\n{response.json()}")
    else:
        print(f"❌ Ошибка запроса: {response.text}")

if __name__ == "__main__":
    token = get_token()
    if token:
        test_protected_request(token)
