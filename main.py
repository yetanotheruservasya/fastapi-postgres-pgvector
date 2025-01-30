from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import openai
import os
import json

app = FastAPI()

DATABASE_URL = "dbname=companies user=myuser password=mypassword host=localhost"

# Функция для вычисления эмбеддингов (используем OpenAI, можно заменить)
def get_embedding(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

# Модель данных
class CompanyData(BaseModel):
    source: str
    company_id: str
    data: dict

# Эндпоинт для сохранения JSONB
@app.post("/store")
def store_data(company: CompanyData):
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

# Обработчик для нормализации данных
@app.post("/normalize/{company_id}")
def normalize_data(company_id: str):
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

# Поиск по вектору
@app.get("/search")
def search_companies(query: str):
    vector = get_embedding(query)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT id, name, industry FROM companies ORDER BY vector <-> %s LIMIT 5", (vector,))
    results = cur.fetchall()
    cur.close()
    conn.close()

    return {"results": results}
