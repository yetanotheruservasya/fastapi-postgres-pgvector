# FastAPI + PostgreSQL + pgVector + pgAdmin

🚀 **A fully containerized stack for storing, normalizing, and searching company data**, with support for JSONB, structured data, and vector search.

---

## 🎯 **What Business Problems Does This System Solve?**

### 1️⃣ **Integration of Data from Different Sources**  
- Support for storing structured and unstructured data (JSONB).  
- Automatic merging of information from various APIs.  

### 2️⃣ **Normalization and Standardization**  
- Data is automatically parsed and structured into a relational format.  
- Ability to add **computed fields** and recalculate vector representations.  

### 3️⃣ **Text and Semantic Search**  
- Vector representation (pgVector) allows finding **similar companies** based on descriptions and other parameters.  
- AI integration (OpenAI API) for **semantic search**.  

### 4️⃣ **Ready for RAG (Retrieval-Augmented Generation)**  
- Can be used to **enhance AI models** in Retrieval-Augmented Generation (RAG).  
- Simple integration with ML systems.  

---

## 📦 **Installation and Deployment**

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yetanotheruservasya/fastapi-postgres-pgvector.git
cd fastapi-postgres-pgvector
```

### 2️⃣ **Create `.env` File**
```bash
cp .env.example .env
```
Then open `.env` and set your values.

### 3️⃣ **Start the System**
```bash
docker-compose up -d
```

### 4️⃣ **Verify Everything is Working**
- **API:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **pgAdmin:** [http://localhost:5050](http://localhost:5050)  
  - **Login:** `admin@example.com`  
  - **Password:** `adminpassword`  

---

## 📡 **API Endpoints**

| Method  | URL                         | Description                                |
|---------|-----------------------------|--------------------------------------------|
| **POST** | `/store`                     | Save company JSONB data                    |
| **POST** | `/normalize/{company_id}`    | Normalize company data                     |
| **GET**  | `/search?query=...`          | Search for companies via vector embedding  |

---

## 🛠 **Technologies**
- Python 3.12 + FastAPI
- PostgreSQL 16 + pgVector
- Docker + Docker Compose
- pgAdmin 4
- OpenAI API (for embeddings)

---

## 🤝 **Authors**
Created by [yetanotheruservasya](https://github.com/yetanotheruservasya).

---

## ⚖️ **License**
This project is distributed under the MIT License.

