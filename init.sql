-- Создаём базы, если их нет
DO
$$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'companies') THEN
        EXECUTE 'CREATE DATABASE companies OWNER myuser ENCODING ''UTF8'';';
    END IF;
END
$$;

-- Подключаемся к основной базе
\c companies;

-- Устанавливаем расширение pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Таблица для хранения JSONB-данных
CREATE TABLE IF NOT EXISTS raw_data (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    company_id TEXT NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Таблица для нормализованных данных компаний
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    name TEXT,
    industry TEXT,
    description TEXT,
    address TEXT,
    phone TEXT,
    email TEXT,
    vector VECTOR(1536)  -- Векторное представление
);

-- Индексы для ускорения поиска
CREATE INDEX IF NOT EXISTS idx_companies_vector ON companies USING ivfflat (vector);
CREATE INDEX IF NOT EXISTS idx_raw_data_company_id ON raw_data (company_id);
