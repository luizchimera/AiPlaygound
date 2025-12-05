# hgTransformer

**Intelligent Document Processing Pipeline for Financial Documents**

Extract structured table data from financial PDFs using Hugging Face Transformers and custom financial tokenizers.

## Features

- **Table Detection**: Uses Microsoft's Table Transformer for detecting tables in documents
- **Table Structure Recognition**: Identifies rows, columns, and cells within tables
- **Financial Entity Extraction**: Custom tokenizer for financial terms, currencies, percentages
- **Async Processing**: Background document processing with status tracking
- **REST API**: FastAPI-based API with automatic OpenAPI documentation
- **PostgreSQL Storage**: Extracted tables stored with full structure metadata

## Architecture

```
PDF Upload → Text/Layout Extraction → Table Detection (Transformer) → Structure Recognition → Database Storage
     ↑                                                                                              ↓
  FastAPI  ←←←←←←←←←←←←←←←←←←←←←←←←←←← JSON Response ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

## Quick Start

### Using Docker (Recommended)

```bash
# Clone and navigate to the project
cd hgTransformer

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

The API will be available at `http://localhost:8000`

### Local Development

1. **Prerequisites**
   - Python 3.11+
   - PostgreSQL
   - poppler-utils (for PDF to image conversion)

2. **Install dependencies**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Copy example env file
   cp .env.example .env

   # Edit .env with your database credentials
   ```

4. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload a PDF for processing |
| GET | `/api/v1/documents/{id}` | Get document metadata |
| GET | `/api/v1/documents/{id}/status` | Check processing status |
| GET | `/api/v1/documents/{id}/tables` | Get extracted tables |
| GET | `/api/v1/documents/{id}/tables/{table_id}` | Get specific table |
| DELETE | `/api/v1/documents/{id}` | Delete document |
| GET | `/api/v1/documents` | List all documents |
| GET | `/api/v1/health` | Health check |

## Usage Example

```python
import requests

# Upload a document
with open("financial_report.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/documents/upload",
        files={"file": ("report.pdf", f, "application/pdf")}
    )
    document_id = response.json()["id"]

# Check processing status
status = requests.get(
    f"http://localhost:8000/api/v1/documents/{document_id}/status"
).json()

print(f"Status: {status['status']}, Tables: {status['tables_extracted']}")

# Get extracted tables
tables = requests.get(
    f"http://localhost:8000/api/v1/documents/{document_id}/tables"
).json()

for table in tables["tables"]:
    print(f"Page {table['page_number']}: {table['num_rows']}x{table['num_columns']} table")
    print(table["table_data"])
```

## Project Structure

```
hgTransformer/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── documents.py      # Document endpoints
│   │   │   └── health.py         # Health check
│   │   └── dependencies.py       # DI dependencies
│   ├── core/
│   │   ├── config.py             # Settings
│   │   └── database.py           # DB connection
│   ├── models/
│   │   └── schemas.py            # Pydantic & SQLAlchemy models
│   ├── services/
│   │   ├── pdf_extractor.py      # PDF processing
│   │   ├── table_detector.py     # Table Transformer
│   │   ├── tokenizer.py          # Financial tokenizer
│   │   └── pipeline.py           # Orchestration
│   └── main.py                   # FastAPI app
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection URL |
| `DEBUG` | `false` | Enable debug mode |
| `DEVICE` | `cpu` | Device for ML models (`cpu` or `cuda`) |
| `DETECTION_THRESHOLD` | `0.7` | Confidence threshold for table detection |
| `STRUCTURE_THRESHOLD` | `0.6` | Confidence threshold for structure recognition |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size |

## Models Used

- **Table Detection**: `microsoft/table-transformer-detection`
- **Structure Recognition**: `microsoft/table-transformer-structure-recognition`
- **Base Tokenizer**: `bert-base-uncased` (extended with financial vocabulary)

## License

MIT License

