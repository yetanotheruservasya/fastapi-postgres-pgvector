import os
import json
import logging
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from typing import List, Dict, Any

import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем переменные окружения и настраиваем логирование
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Конфигурация безопасности и JWT
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Настройки базы данных
POSTGRES_USER = os.getenv("POSTGRES_USER", "---")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "---")
POSTGRES_DB = os.getenv("POSTGRES_DB", "---")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "---")
DATABASE_URL = f"dbname={POSTGRES_DB} user={POSTGRES_USER} password={POSTGRES_PASSWORD} host={POSTGRES_HOST}"

# Настройки хэширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Данные администратора (например, для тестирования)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "---")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "---")
ADMIN_FULL_NAME = os.getenv("ADMIN_FULL_NAME", "---")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "---")

# "Фейковая" база пользователей (можно заменить на реальную)
fake_users_db: Dict[str, Dict[str, Any]] = {
    ADMIN_USERNAME: {
        "username": ADMIN_USERNAME,
        "full_name": ADMIN_FULL_NAME,
        "email": ADMIN_EMAIL,
        "hashed_password": pwd_context.hash(ADMIN_PASSWORD),
        "disabled": False
    }
}

app = FastAPI()

# --- Работа с OpenAI ---

class OpenAIClientSingleton:
    """
    Singleton для клиента OpenAI.
    """
    _instance: OpenAI = None

    @classmethod
    def get_client(cls) -> OpenAI:
        if cls._instance is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            cls._instance = OpenAI(api_key=api_key)
            logging.info(f"Initialized OpenAI client with API key: {api_key[:5]}...")
        return cls._instance

def get_openai_client() -> OpenAI:
    """
    Возвращает экземпляр клиента OpenAI.
    """
    return OpenAIClientSingleton.get_client()

def get_embedding(text: str) -> List[float]:
    """
    Получает векторное представление текста с помощью модели OpenAI.
    """
    client = get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

def generate_description(entity_text: str) -> str:
    """
    Генерирует описание для сущности с помощью модели GPT-4o.
    """
    client = get_openai_client()
    prompt = f"Generate a short description for the entity following context: {entity_text}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that summarizes company information."
            },
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# --- Аутентификация и JWT ---

def authenticate_user(username: str, password: str) -> Any:
    """
    Проверяет корректность имени пользователя и пароля.
    """
    user = fake_users_db.get(username)
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """
    Создаёт JWT токен.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Получает пользователя по JWT токену.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = fake_users_db.get(username)
        if user is None:
            raise credentials_exception
        return user
    except JWTError as exc:
        raise credentials_exception from exc

async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Проверяет, что пользователь активен (не отключён).
    """
    if current_user.get("disabled"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is disabled"
        )
    return current_user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Эндпоинт для получения JWT токена.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

# --- Работа с базой данных ---

@contextmanager
def get_db():
    """
    Контекстный менеджер для подключения к базе данных.
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()

# Модель данных компании
class CompanyData(BaseModel):
    source: str
    company_id: str
    data: dict

@app.post("/store")
def store_data(
    company: CompanyData,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Эндпоинт для сохранения данных компании (JSONB) в БД.
    """
    if not company.source or not company.company_id or not company.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields: source, company_id, or data"
        )
    logging.info(f"Storing data for company_id: {company.company_id} from source: {company.source}")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO raw_data (source, company_id, data) VALUES (%s, %s, %s)",
                (company.source, company.company_id, json.dumps(company.data))
            )
            conn.commit()
    return {"message": "Data stored successfully"}

@app.post("/normalize/{company_id}")
def normalize_data(
    company_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Эндпоинт для нормализации данных компании.
    Доступен только для аутентифицированных пользователей.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM raw_data WHERE company_id = %s", (company_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Company not found")
            data = row[0]
            name = data.get("name")
            industry = data.get("industry")
            description = data.get("description", "")
            vector = get_embedding(description)
            cur.execute(
                "INSERT INTO companies (name, industry, description, vector) VALUES (%s, %s, %s, %s)",
                (name, industry, description, vector)
            )
            conn.commit()
    return {"message": "Data normalized successfully"}

@app.get("/search")
def search_companies(
    query: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Эндпоинт для поиска компаний с использованием векторного представления запроса.
    """
    vector = get_embedding(query)
    # Приводим вектор к списку значений float32
    vector = np.array(vector, dtype=np.float32).tolist()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM companies ORDER BY vector <-> %s::vector LIMIT 5",
                (json.dumps(vector),)
            )
            column_names = [desc[0] for desc in cur.description]
            results = cur.fetchall()

    search_results = []
    for res in results:
        entity_data = dict(zip(column_names, res))
        entity_data_text = json.dumps(entity_data, indent=2)
        enhanced_description = generate_description(entity_data_text)
        entity_data["enhanced_description"] = enhanced_description
        search_results.append(entity_data)
    return {"results": search_results}
