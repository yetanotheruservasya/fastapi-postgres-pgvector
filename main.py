from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import psycopg2
import openai
import os
import json
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Security settings for JWT
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "dbname=companies user=myuser password=mypassword host=db")

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
def get_embedding(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

# Authentication and JWT handling
def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
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
    except JWTError:
        raise credentials_exception

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
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
    source: str
    company_id: str
    data: dict

# Endpoint for storing JSONB data (protected)
@app.post("/store")
def store_data(company: CompanyData, current_user: dict = Depends(get_current_user)):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO raw_data (source, company_id, data) VALUES (%s, %s, %s)",
        (company.source, company.company_id, json.dumps(company.data))
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"message": "Data stored successfully"}

# Endpoint for normalizing data (protected)
@app.post("/normalize/{company_id}")
def normalize_data(company_id: str, current_user: dict = Depends(get_current_user)):
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
    vector = get_embedding(query)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT id, name, industry FROM companies ORDER BY vector <-> %s LIMIT 5", (vector,))
    results = cur.fetchall()
    cur.close()
    conn.close()

    return {"results": results}
