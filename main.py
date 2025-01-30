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
    response = openai.Embedding.create(model="text-embedding-ada-002", input=[text])
    return response["data"][0]["embedding"]

# Generate entity description using GPT-4o
def generate_description(entity_text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Generate a short description: for the entity folowing context: {entity_text}"
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that summarizes company information."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

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

# Endpoint for obtaining token
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

# Vector search (protected)
@app.get("/search")
def search_companies(query: str, current_user: dict = Depends(get_current_user)):
    vector = get_embedding(query)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT * FROM companies ORDER BY vector <-> %s LIMIT 5", (vector,))
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
