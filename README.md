# FastAPI + PostgreSQL + pgVector + pgAdmin

🚀 **A fully containerized stack for storing, normalizing, and searching entities**, with support for JSONB, structured data, and vector search.

---

## 🎯 **What Business Problems Does This System Solve?**

### 1️⃣ **Integration of Data from Different Sources**  
- Support for storing structured and unstructured data (JSONB).  
- Automatic merging of information from various APIs.  

### 2️⃣ **Normalization and Standardization**  
- Data is automatically parsed and structured into a relational format.  
- Ability to add **computed fields** and recalculate vector representations.  

### 3️⃣ **Text and Semantic Search**  
- Vector representation (pgVector) allows finding **similar entities** based on descriptions and other parameters.  
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
Required environment variables:
- `POSTGRES_*`: Database credentials
- `OPENAI_API_KEY`: Your OpenAI API key
- `SECRET_KEY`: JWT secret key
- `ADMIN_USERNAME`: Admin login
- `ADMIN_PASSWORD`: Admin password
- `ADMIN_EMAIL`: Admin email

### 3️⃣ **Start the System**
```bash
docker-compose up -d
```

### 4️⃣ **Initialize the Database**
```bash
curl -X POST "http://localhost:8000/admin/create_database" -H "Authorization: Bearer YOUR_TOKEN"
curl -X POST "http://localhost:8000/admin/create_tables" -H "Authorization: Bearer YOUR_TOKEN"
```

### 5️⃣ **Verify Everything is Working**
- **API:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **pgAdmin:** [http://localhost:5050](http://localhost:5050)  
  - **Login:** `admin@example.com`  
  - **Password:** `adminpassword`  

---

## 🔧 **Configuration Parameters**

### Environment Variables

#### Database Configuration
- `POSTGRES_HOST`: Database host (default: "db")
- `POSTGRES_USER`: Database username
- `POSTGRES_PASSWORD`: Database password
- `POSTGRES_DB`: Main database name
- `ANOTHER_DB`: Additional database name (optional)

#### pgAdmin Configuration
- `PGADMIN_DEFAULT_EMAIL`: Admin email for pgAdmin interface
- `PGADMIN_DEFAULT_PASSWORD`: Admin password for pgAdmin
- `PGADMIN_LISTEN_PORT`: Port for pgAdmin (default: 5050)
- `PGADMIN_LISTEN_ADDRESS`: Listen address (default: 0.0.0.0)

#### API Security
- `SECRET_KEY`: JWT secret key for token generation
- `ADMIN_USERNAME`: Admin username for API access
- `ADMIN_PASSWORD`: Admin password for API access
- `ADMIN_FULL_NAME`: Admin's full name
- `ADMIN_EMAIL`: Admin's email address

#### OpenAI Integration
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings generation

#### File Paths
- `ENTITY_CONFIG_FILE`: Path to entity configuration JSON file

### Example of Entity Configuration (entity_config.json)

```json
{
    "entity_name": "company",
    "fields": {
        "name": {
            "source_field": "name",
            "required": true
        },
        "industry": {
            "source_field": "industries",
            "required": false
        },
        "description": {
            "source_field": "description",
            "required": false,
            "vectorize": true
        }
    },
    "vector_settings": {
        "vector_field": "description",
        "model": "text-embedding-ada-002"
    }
}
```

#### Configuration Fields
- `entity_name`: Name of the entity (used for table naming)
- `fields`: Mapping of target fields to source data paths
  - `source_field`: JSON path to source data
  - `required`: Whether the field is required
- `vector_settings`: Configuration for vector embeddings
  - `vector_field`: Field to use for generating embeddings

---

## 🔐 **Authentication**

1. **Get Access Token**
```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your_username&password=your_password"
```

2. **Use Token in Requests**
```bash
curl -X POST "http://localhost:8000/store" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"your": "data"}'
```

## 📡 **API Endpoints**

### Data Operations
| Method   | URL                         | Description                                |
|----------|-----------------------------|--------------------------------------------|
| **POST** | `/store`                   | Save entity raw data (JSONB)                    |
| **POST** | `/normalize/{entity_id}`  | Normalize entity data                     |
| **GET**  | `/search?query=...`        | Search for entities via vector embedding  |

### Admin Operations
| Method     | URL                         | Description                           |
|------------|----------------------------|---------------------------------------|
| **POST**   | `/admin/create_database`   | Initialize database                   |
| **POST**   | `/admin/create_tables`     | Create required tables                |
| **DELETE** | `/admin/delete_data`       | Clear all data from tables            |
| **DELETE** | `/admin/delete_tables`     | Drop all tables                       |
| **DELETE** | `/admin/delete_database`   | Delete entire database                |

---

## 🛠 **Technologies**
- Python 3.12 + FastAPI
- PostgreSQL 16 + pgVector
- JWT Authentication
- Docker + Docker Compose
- pgAdmin 4
- OpenAI API (for embeddings)

---

## 🤝 **Authors**
Created by [yetanotheruservasya](https://github.com/yetanotheruservasya).

---

## ⚖️ **License**
This project is distributed under the MIT License.

