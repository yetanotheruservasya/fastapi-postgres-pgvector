# FastAPI + PostgreSQL + pgVector + pgAdmin

🚀 **Полностью контейнеризированный стек для хранения, нормализации и поиска данных компаний**, с поддержкой JSONB, структурированных данных и векторного поиска.

---

## 🎯 **Какие бизнес-задачи решает система?**

### 1️⃣ **Интеграция данных из разных источников**  
- Поддержка хранения структурированных и неструктурированных данных (JSONB).  
- Автоматическое объединение информации о компаниях из разных API.  

### 2️⃣ **Нормализация и приведение к единому формату**  
- Данные автоматически разбираются и приводятся к реляционной структуре.  
- Возможность добавлять **вычисляемые поля** и пересчитывать по ним векторные представления.  

### 3️⃣ **Поиск по тексту и семантике**  
- Векторное представление (pgVector) позволяет находить **похожие компании** по описанию и другим параметрам.  
- Интеграция с AI (OpenAI API) для **поиска по смыслу**.  

### 4️⃣ **Готовность к использованию в RAG**  
- Можно использовать данные для **обогащения AI-моделей** в Retrieval-Augmented Generation (RAG).  
- Простая интеграция с ML-системами.  

---

## 📦 **Установка и запуск**

### 1️⃣ **Клонировать репозиторий**
```bash
git clone https://github.com/yetanotheruservasya/fastapi-postgres-pgvector.git
cd fastapi-postgres-pgvector
```

### 2️⃣ **Создать `.env` файл**
```bash
cp .env.example .env
```
Затем открыть `.env` и задать свои значения.

### 3️⃣ **Запустить систему**
```bash
docker-compose up -d
```

### 4️⃣ **Проверить, что всё работает**
- **API:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **pgAdmin:** [http://localhost:5050](http://localhost:5050)  
  - **Логин:** `admin@example.com`  
  - **Пароль:** `adminpassword`  

---

## 📡 **API Эндпоинты**

| Метод    | URL                        | Описание                                   |
|----------|----------------------------|--------------------------------------------|
| **POST** | `/store`                    | Сохранить JSONB-данные компании            |
| **POST** | `/normalize/{company_id}`   | Нормализовать данные компании              |
| **GET**  | `/search?query=...`         | Поиск компаний по векторному представлению |

---

## 🛠 **Технологии**
- Python 3.10 + FastAPI
- PostgreSQL 16 + pgVector
- Docker + Docker Compose
- pgAdmin 4
- OpenAI API (для эмбеддингов)

---

## 🤝 **Авторы**
Создан пользователем [yetanotheruservasya](https://github.com/yetanotheruservasya).

---

## ⚖️ **Лицензия**
Этот проект распространяется под лицензией MIT.

