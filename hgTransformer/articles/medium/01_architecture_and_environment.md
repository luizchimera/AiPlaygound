Building an Intelligent Document Processing Pipeline with Hugging Face Transformers — Part 1: Architecture & Environment Setup

Extract tables from financial PDFs using state-of-the-art ML models

---

Financial documents are everywhere—invoices, bank statements, annual reports, SEC filings. Yet extracting structured data from these documents remains a significant challenge for organizations. Tables buried within PDFs contain critical information, but manual extraction is time-consuming and error-prone.

In this series, I'll walk you through building **hgTransformer**, an intelligent document processing pipeline that leverages Hugging Face Transformers to automatically extract tables from financial documents and store them in a structured database.

By the end of this series, you'll have a production-ready REST API that can:

- Accept PDF uploads
- Detect tables using state-of-the-art ML models
- Extract structured data with financial entity recognition
- Store results in PostgreSQL
- Deploy via Docker containers

---

## The Architecture

Let's start with a bird's-eye view of what we're building. The data flows like this:

**PDF Upload → Text/Layout Extraction → Table Detection (Transformer) → Structure Recognition → Database Storage**

The FastAPI server receives requests and returns JSON responses, while all the heavy processing happens in the background.

### Component Breakdown

**1. FastAPI REST Service**

The entry point for our pipeline. FastAPI provides async request handling, automatic OpenAPI documentation, and excellent performance for ML workloads.

**2. PDF Extraction Layer**

Using `pdfplumber` for native PDFs, we extract raw text content, page layout information, initial table detection, and page images for transformer processing.

**3. Table Detection (ML Layer)**

Microsoft's Table Transformer models handle table detection (finding tables in document images) and structure recognition (identifying rows, columns, and cell boundaries).

**4. Financial Tokenizer**

A custom tokenizer that understands financial domain vocabulary: currency formats ($1,234.56, €500), percentages (15.5%), accounting notation ((1,000) for negatives), and financial metrics (EBITDA, ROI, P/E).

**5. PostgreSQL Storage**

Three-table schema capturing document metadata, extracted tables with JSONB structure, and individual cells with position data.

---

## Project Structure

Here's how we organize our code:

```
hgTransformer/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── documents.py
│   │   │   └── health.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py
│   │   └── database.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── pdf_extractor.py
│   │   ├── table_detector.py
│   │   ├── tokenizer.py
│   │   └── pipeline.py
│   └── main.py
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

This structure follows clean architecture principles:

- **api/** — HTTP layer, routes, and dependencies
- **core/** — Configuration and infrastructure
- **models/** — Data schemas (Pydantic + SQLAlchemy)
- **services/** — Business logic and ML components

---

## Environment Configuration

### The Settings Class

We use Pydantic Settings for central, type-safe configuration:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "hgTransformer"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/hgtransformer"

    # Model Configuration
    table_detection_model: str = "microsoft/table-transformer-detection"
    table_structure_model: str = "microsoft/table-transformer-structure-recognition"
    device: str = "cpu"

    # Processing Thresholds
    detection_threshold: float = 0.7
    structure_threshold: float = 0.6
```

Let me break down what each part does:

**Imports and class definition:**

`from pydantic_settings import BaseSettings, SettingsConfigDict` imports Pydantic's settings management classes for type-safe configuration. The `class Settings(BaseSettings)` creates a settings class that automatically reads from environment variables.

**Model configuration:**

The `model_config = SettingsConfigDict(...)` section configures how settings are loaded. `env_file=".env"` reads variables from a .env file in the project root. `env_file_encoding="utf-8"` ensures proper character encoding. `case_sensitive=False` allows both `DATABASE_URL` or `database_url` in env files.

**Application settings:**

`app_name: str = "hgTransformer"` sets the application name with a type hint and default value. `debug: bool = False` sets debug mode disabled by default for safety.

**Database configuration:**

`database_url: str = "postgresql+asyncpg://..."` defines the async PostgreSQL connection string using the asyncpg driver.

**Model configuration:**

`table_detection_model` contains the HuggingFace model ID for detecting tables in images. `table_structure_model` is the HuggingFace model ID for recognizing table structure (rows/columns). `device: str = "cpu"` sets the PyTorch device — change to "cuda" for GPU acceleration.

**Processing thresholds:**

`detection_threshold: float = 0.7` sets minimum confidence (0-1) for table detection — higher means fewer false positives. `structure_threshold: float = 0.6` sets minimum confidence for row/column detection, slightly lower since structure is harder to detect.

---

## Key Configuration Decisions

**Async Database URL**

We use `postgresql+asyncpg` for non-blocking database operations, critical for handling concurrent document processing.

**Device Selection**

Default to CPU for broad compatibility, but easily switch to CUDA for GPU acceleration in production.

**Confidence Thresholds**

Detection threshold at 0.7 means tables must have 70%+ confidence to be extracted. Structure threshold at 0.6 is the row/column detection threshold. These can be tuned based on your document types.

---

## Dependencies

Our core dependencies include:

**Core ML/Transformers:** torch>=2.0.0, transformers>=4.35.0, timm>=0.9.0

**PDF Processing:** pdfplumber>=0.10.0, pdf2image>=1.16.0

**API Framework:** fastapi>=0.104.0, uvicorn[standard]>=0.24.0

**Database:** sqlalchemy>=2.0.0, asyncpg>=0.29.0

**Utilities:** pydantic>=2.5.0, pydantic-settings>=2.1.0, aiofiles>=23.2.0

### Why These Choices?

**torch + transformers** is the backbone of modern NLP/CV, required for Table Transformer.

**pdfplumber** is best-in-class for native PDF text extraction.

**pdf2image** converts PDF pages to images for transformer processing.

**asyncpg** is the fastest async PostgreSQL driver for Python.

**SQLAlchemy 2.0** is a modern ORM with full async support.

---

## What's Next?

In Part 2, we'll dive deep into the FastAPI application structure, core database setup with async SQLAlchemy, API endpoint design and implementation, and dependency injection patterns.

Stay tuned!

---

*Have you built document processing pipelines? What challenges did you face? Let me know in the comments!*

