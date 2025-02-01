"""
This module implements the main API,
handling authentication, OpenAI communications, and database operations.
"""

# =====================
# Imports and Global Setup
# =====================
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
from dotenv import load_dotenv
from openai import OpenAI

from models.main_models import EntityData, EntityNormalizedData
from models.config_models import EntityConfig

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Security and JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database settings
POSTGRES_USER = os.getenv("POSTGRES_USER", "---")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "---")
POSTGRES_DB = os.getenv("POSTGRES_DB", "---")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "---")
# Break long DATABASE_URL assignment into multiple lines
DATABASE_URL = (
    f"dbname={POSTGRES_DB} user={POSTGRES_USER} "
    f"password={POSTGRES_PASSWORD} host={POSTGRES_HOST}"
)

# Password hashing settings
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Admin data (e.g., for testing)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "---")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "---")
ADMIN_FULL_NAME = os.getenv("ADMIN_FULL_NAME", "---")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "---")

# "Fake" user database (can be replaced with a real one)
fake_users_db: Dict[str, Dict[str, Any]] = {
    ADMIN_USERNAME: {
        "username": ADMIN_USERNAME,
        "full_name": ADMIN_FULL_NAME,
        "email": ADMIN_EMAIL,
        "hashed_password": pwd_context.hash(ADMIN_PASSWORD),
        "disabled": False
    }
}

ENTITY_CONFIG_FILE = os.getenv("ENTITY_CONFIG_FILE", "---")

app = FastAPI()

# =====================
# Utility Functions
# =====================

def load_entity_config(config_path: str = ENTITY_CONFIG_FILE) -> EntityConfig:
    """
    Load and validate entity configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file. Defaults to ENTITY_CONFIG_FILE.

    Returns:
        EntityConfig: Validated configuration object.

    Raises:
        FileNotFoundError: If configuration file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    # Create the EntityConfig object; validation happens here
    entity_config = EntityConfig.model_validate(raw_config)
    return entity_config

def extract_field(data: dict, field_path: str):
    """
    Extracts the value from a nested dictionary using a dot-separated path.
    
    For example, given data = {"data": {"name": "Acme Corp"}} and field_path = "data.name",
    it returns "Acme Corp".
    """
    keys = field_path.split(".")
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value

def get_normalized_table_name(entity_name: str) -> str:
    """
    Forms a table name for normalized data based on the entity name.
    If the name ends with 'y', replaces 'y' with 'ies', otherwise adds 's'.
    """
    if entity_name.endswith("y"):
        return entity_name[:-1] + "ies"
    return entity_name + "s"

@contextmanager
def get_db():
    """
    Context manager for connecting to the database.
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()

# Function to obtain a super connection to the postgres database (for creating/deleting databases)
def get_super_db_connection():
    """
    Returns a connection to the 'postgres' database for administrative operations.
    """
    conn = psycopg2.connect(
        dbname="postgres",
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST
    )
    return conn

try:
    config = load_entity_config()
    logging.info("Loaded entity config: %s", config.json())
except Exception as e:
    logging.error("Error loading entity config: %s", e)
    raise e

# =====================
# OpenAI Integration
# =====================

class OpenAIClientSingleton:
    """
    Singleton for OpenAI client.
    """
    _instance: OpenAI = None

    @classmethod
    def get_client(cls) -> OpenAI:
        """
        Returns the singleton instance of the OpenAI client.
        """
        if cls._instance is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            cls._instance = OpenAI(api_key=api_key)
            logging.info("Initialized OpenAI client with API key: %s...", api_key[:5])
        return cls._instance

def get_openai_client() -> OpenAI:
    """
    Returns an instance of the OpenAI client.
    """
    return OpenAIClientSingleton.get_client()

def get_embedding(text: str) -> List[float]:
    """
    Gets the vector representation of the text using the OpenAI model.
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
    Generates a description for the entity using the GPT-4o model.
    """
    client = get_openai_client()
    prompt = f"Generate a short description for the entity following context: {entity_text}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that summarizes entity information."
            },
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# =====================
# Authentication and JWT
# =====================

def authenticate_user(username: str, password: str) -> Any:
    """
    Verifies the correctness of the username and password.
    """
    user = fake_users_db.get(username)
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """
    Creates a JWT token.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Gets the user by JWT token.
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
    Checks that the user is active (not disabled).
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
    Endpoint for obtaining a JWT token.
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

# =====================
# API Endpoints
# =====================

@app.post("/store")
def store_data(
    entity: EntityData,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Endpoint for saving entity data (JSONB) to the database.
    """
    # Load the entity configuration
    entity_config = load_entity_config()

    normalized_data = {}
    # Check required fields according to configuration and prepare normalized data
    for field_name, field_conf in entity_config.fields.items():
        value = extract_field(entity.data, field_conf.source_field)
        if field_conf.required and value is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Missing required field: '{field_name}' "
                    f"(expected at '{field_conf.source_field}')"
                )
            )
        normalized_data[field_name] = value

    # Use the full model dump so that all EntityData values are saved,
    # and update/merge with normalized data.
    data_to_insert = entity.model_dump()
    data_to_insert.update(normalized_data)

    # Convert any dictionary or list values to JSON strings,
    # so that psycopg2 can adapt them properly.
    for key, val in data_to_insert.items():
        if isinstance(val, (dict, list)):
            data_to_insert[key] = json.dumps(val)

    # Dynamically build the SQL query for inserting data
    columns = list(data_to_insert.keys())
    values = list(data_to_insert.values())
    columns_str = ', '.join(columns)
    placeholders = ', '.join(['%s'] * len(values))
    sql = f"INSERT INTO raw_data ({columns_str}) VALUES ({placeholders})"

    logging.info("Storing data for entity: %s", data_to_insert)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, values)
            conn.commit()

    return {"message": "Data stored successfully"}

@app.post("/normalize/{entity_id}")
def normalize_data(
    entity_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Endpoint for normalizing entity data.
    Available only to authenticated users.
    """
    # Загружаем конфигурацию сущности
    entity_config = load_entity_config()
    normalized_table = get_normalized_table_name(entity_config.entity_name)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM raw_data WHERE entity_id = %s", (entity_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Entity not found in raw_data")
            raw_data = row[0]

    # Нормализуем данные согласно конфигурации
    normalized_data = {}
    for field_name, field_conf in entity_config.fields.items():
        value = extract_field(raw_data, field_conf.source_field)
        if field_conf.required and value is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Missing required field: '{field_name}' "
                    f"(expected at '{field_conf.source_field}')"
                )
            )
        normalized_data[field_name] = value

    # Векторизация (если задана в конфигурации)
    if entity_config.vector_settings:
        vector_field = entity_config.vector_settings.vector_field
        text_for_vector = normalized_data.get(vector_field)
        if text_for_vector:
            vector = get_embedding(text_for_vector)
            normalized_data["vector"] = vector
        else:
            logging.warning("No value found for vectorization in field '%s'", vector_field)

    # Валидируем данные по модели нормализованных данных
    try:
        normalized_entity = EntityNormalizedData(**normalized_data)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error in normalized data: {e}"
        ) from e

    # Динамическая вставка нормализованных данных в вычисляемую таблицу
    data_to_insert = normalized_entity.model_dump()
    columns = list(data_to_insert.keys())
    values = list(data_to_insert.values())
    columns_str = ', '.join(columns)
    placeholders = ', '.join(['%s'] * len(values))
    sql = f"INSERT INTO {normalized_table} ({columns_str}) VALUES ({placeholders})"

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, values)
            conn.commit()

    return {"message": "Data normalized successfully"}

@app.get("/search")
def search_companies(
    query: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Endpoint for searching entities using the vector representation of the query.
    """
    # Получаем имя нормализованной таблицы из конфигурации
    local_config = load_entity_config()
    normalized_table = get_normalized_table_name(local_config.entity_name)

    # Проверка существования нормализованной таблицы
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename = %s
                );
            """, (normalized_table,))
            table_exists = cur.fetchone()[0]
            if not table_exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"{normalized_table} table does not exist. Please create tables first."
                )

    vector = get_embedding(query)
    # Приводим вектор к списку float32
    vector = np.array(vector, dtype=np.float32).tolist()
    query_sql = f"SELECT * FROM {normalized_table} ORDER BY vector <-> %s::vector LIMIT 5"
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query_sql, (json.dumps(vector),))
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

#####################################
# 1. Deleting Data from Tables
#####################################
@app.delete("/admin/delete_data")
def delete_data(current_user: dict = Depends(get_current_active_user)):
    """
    Deletes all data from existing tables (raw_data and companies).
    Skips tables that don't exist.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            # Check if tables exist before truncating
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('raw_data', 'companies');
            """)
            existing_tables = [row[0] for row in cur.fetchall()]
            
            if existing_tables:
                # Build TRUNCATE query only for existing tables
                tables_to_truncate = ', '.join(existing_tables)
                cur.execute(f"TRUNCATE TABLE {tables_to_truncate} RESTART IDENTITY;")
                conn.commit()
                return {"message": f"Data deleted successfully from tables: {tables_to_truncate}"}
            else:
                return {"message": "No tables found to delete data from"}

#####################################
# 2. Dropping Tables and Databases
#####################################
@app.delete("/admin/delete_tables")
def delete_tables(current_user: dict = Depends(get_current_active_user)):
    """
    Drops the raw_data and normalized tables from the database.
    """
    # Получаем имя нормализованной таблицы из конфигурации
    local_config = load_entity_config()
    normalized_table = get_normalized_table_name(local_config.entity_name)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS raw_data CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {normalized_table} CASCADE;")
            conn.commit()
    return {"message": f"Tables raw_data and {normalized_table} dropped successfully"}

@app.delete("/admin/delete_database")
def delete_database(current_user: dict = Depends(get_current_active_user)):
    """
    Deletes the database if necessary.
    WARNING: This is a destructive operation!
    """
    conn = get_super_db_connection()
    try:
        # Set autocommit to True to execute DROP DATABASE
        conn.set_session(autocommit=True)
        
        with conn.cursor() as cur:
            # Terminate all connections to the target database
            cur.execute("""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid();
            """, (POSTGRES_DB,))
            # Drop the database
            cur.execute(f"DROP DATABASE IF EXISTS {POSTGRES_DB}")
    finally:
        conn.close()
    
    return {"message": f"Database {POSTGRES_DB} deleted successfully"}

#####################################
# 3. Creating Tables and Databases Based on Configuration
#####################################
@app.post("/admin/create_database")
def create_database(current_user: dict = Depends(get_current_active_user)):
    """
    Creates the database if it does not exist.
    """
    conn = get_super_db_connection()
    try:
        # Set autocommit to True to execute CREATE DATABASE
        conn.set_session(autocommit=True)
        
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (POSTGRES_DB,))
            if not cur.fetchone():
                cur.execute(
                    f"CREATE DATABASE {POSTGRES_DB} "
                    f"OWNER {POSTGRES_USER} ENCODING 'UTF8'"
                )
    finally:
        conn.close()
    
    return {"message": f"Database {POSTGRES_DB} created successfully (if it did not exist)"}

@app.post("/admin/create_tables")
def create_tables(current_user: dict = Depends(get_current_active_user)):
    """
    Creates the raw_data and normalized data tables based on the configuration.
    First creates the pgvector extension if needed.
    """
    # Создание расширения pgvector и таблицы raw_data (фиксированная схема)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS raw_data (
                    id SERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_entity_id ON raw_data (entity_id);")
            conn.commit()

    # Генерация SQL для создания нормализованной таблицы на основе конфигурации
    local_config = load_entity_config()
    normalized_table = get_normalized_table_name(local_config.entity_name)
    columns = ["entity_id TEXT"]  # обязательное поле идентификатора
    for field_name in local_config.fields.keys():
        columns.append(f"{field_name} TEXT")
    if local_config.vector_settings:
        # Фиксированная размерность для модели text-embedding-ada-002, можно параметризовать
        columns.append("vector VECTOR(1536)")
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {normalized_table} (
            id SERIAL PRIMARY KEY,
            {', '.join(columns)}
        );
    """
    index_entity_sql = f"CREATE INDEX IF NOT EXISTS idx_{normalized_table}_entity_id ON {normalized_table} (entity_id);"
    index_vector_sql = ""
    if local_config.vector_settings:
        index_vector_sql = f"CREATE INDEX IF NOT EXISTS idx_{normalized_table}_vector ON {normalized_table} USING ivfflat (vector);"

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            cur.execute(index_entity_sql)
            if index_vector_sql:
                cur.execute(index_vector_sql)
            conn.commit()

    return {"message": "Tables created successfully based on configuration"}