"""
Main module for working with the company database.
Contains functions for authentication, data storage, normalization, and search.
"""
import os
import json
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import psycopg2
from openai import OpenAI
import numpy as np
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Security settings for JWT
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database settings
POSTGRES_USER = os.getenv("POSTGRES_USER", "myuser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mypassword")
POSTGRES_DB = os.getenv("POSTGRES_DB", "companies")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
DATABASE_URL = f"dbname={POSTGRES_DB} user={POSTGRES_USER} password={POSTGRES_PASSWORD} host={POSTGRES_HOST}"

# Password hashing settings
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load user credentials from environment variables
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")
ADMIN_FULL_NAME = os.getenv("ADMIN_FULL_NAME", "Admin User")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")

# User database (can be replaced with a real database)
fake_users_db = {
    ADMIN_USERNAME: {
        "username": ADMIN_USERNAME,
        "full_name": ADMIN_FULL_NAME,
        "email": ADMIN_EMAIL,
        "hashed_password": pwd_context.hash(ADMIN_PASSWORD),
        "disabled": False
    }
}

app = FastAPI()

# Function to compute embeddings
class OpenAIClientSingleton:
    """
    Singleton class to manage the OpenAI client instance.
    Ensures that only one instance of the client is created.
    """
    _instance = None

    @classmethod
    def get_client(cls):
        """
        Returns OpenAI client instance with API key from environment.
        Uses singleton pattern to avoid multiple client creations.
        """
        if cls._instance is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            cls._instance = OpenAI(api_key=api_key)
            print(f"Using OpenAI API key: {api_key[:5]}...")
        return cls._instance

def get_openai_client():
    """
    Returns the OpenAI client instance.
    """
    return OpenAIClientSingleton.get_client()

def get_embedding(text):
    """
    Gets the vector representation of text using the OpenAI model.
    
    :param text: Text to be converted.
    :return: Vector representation of the text.
    """
    client = get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

# Generate entity description using GPT-4o
def generate_description(entity_text):
    """
    Generates a description of an entity using the GPT-4o model.
    
    :param entity_text: Text of the entity.
    :return: Generated description.
    """
    client = get_openai_client()
    prompt = f"Generate a short description for the entity following context: {entity_text}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are an AI assistant that summarizes company information."},
            {"role": "user",
             "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Authentication and JWT handling
def authenticate_user(username: str, password: str):
    """
    Authenticates a user by username and password.
    
    :param username: Username.
    :param password: User password.
    :return: User data or False if authentication failed.
    """
    user = fake_users_db.get(username)
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    """
    Creates a JWT access token.
    
    :param data: Data to include in the token.
    :param expires_delta: Token expiration time.
    :return: Generated JWT token.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Gets the current authenticated user by JWT token.
    
    :param token: JWT token.
    :return: User data.
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

# Endpoint for obtaining token
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Endpoint for obtaining a JWT token.
    
    :param form_data: Form data for authentication.
    :return: Access token and token type.
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

# Data model
class CompanyData(BaseModel):
    """
    Company data model.
    """
    source: str
    company_id: str
    data: dict

# Endpoint for storing JSONB data (protected)
@app.post("/store")
def store_data(
    company: CompanyData,
    current_user: dict = Depends(get_current_user)
    ):
    """
    Endpoint for storing company data in JSONB format.
    
    :param company: Company data.
    :param current_user: Current authenticated user.
    :return: Message about successful data storage.
    """
    print(f"DATABASE_URL: {DATABASE_URL}")  # Add logging to verify DATABASE_URL
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    # Validate user authentication
    if not current_user or current_user.get("disabled"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is not authorized or disabled"
        )

    # Validate input data
    if not company.source or not company.company_id or not company.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields: source, company_id, or data"
        )
    cur.execute(
        "INSERT INTO raw_data (source, company_id, data) VALUES (%s, %s, %s)",
        (company.source, company.company_id, json.dumps(company.data))
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"message": "Data stored successfully"}

# Handler for normalizing data
@app.post("/normalize/{company_id}")
def normalize_data(company_id: str):
    """
    Handler for normalizing company data.
    
    :param company_id: Company identifier.
    :return: Message about successful data normalization.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

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
    cur.close()
    conn.close()
    return {"message": "Data normalized successfully"}

# Vector search (protected)
@app.get("/search")
def search_companies(query: str, current_user: dict = Depends(get_current_user)):
    """
    Endpoint for searching companies by vector representation of the query.
    
    :param query: Search query.
    :param current_user: Current authenticated user.
    :return: Search results with enhanced descriptions.
    """
    if not current_user or current_user.get("disabled"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is not authorized or disabled"
        )
    vector = get_embedding(query)
    vector = np.array(vector, dtype=np.float32).tolist()  # Convert to float32 list

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM companies ORDER BY vector <-> %s::vector LIMIT 5",
        (json.dumps(vector),))

    column_names = [desc[0] for desc in cur.description]
    results = cur.fetchall()
    cur.close()
    conn.close()

    # Generate enhanced descriptions
    search_results = []
    for res in results:
        entity_data = dict(zip(column_names, res))
        entity_data_text = json.dumps(entity_data, indent=2)
        enhanced_description = generate_description(entity_data_text)
        entity_data["enhanced_description"] = enhanced_description
        search_results.append(entity_data["enhanced_description"])

    return {"results": search_results}
