Building an Intelligent Document Processing Pipeline — Part 5: Testing Strategy

Unit tests, integration tests, and mocking ML models

---

We've built a complete document processing pipeline. But how do we know it works? How do we prevent regressions? How do we test ML components that are inherently non-deterministic?

In this final article, we'll establish a comprehensive testing strategy covering unit tests for individual services, integration tests for the pipeline, API endpoint testing, and ML model mocking strategies.

A well-tested system is a trustworthy system.

---

## Testing Philosophy

### The Testing Pyramid

At the base, we have many **unit tests** — fast and isolated. In the middle, we have some **integration tests** — moderate speed. At the top, we have few **E2E tests** — slow but high confidence.

For ML systems, we add **model tests** to verify model loading and basic inference, **data tests** to validate input/output schemas, and **regression tests** to catch accuracy degradation.

---

## Project Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_tokenizer.py
│   ├── test_pdf_extractor.py
│   └── test_schemas.py
├── integration/
│   ├── __init__.py
│   ├── test_pipeline.py
│   └── test_database.py
├── api/
│   ├── __init__.py
│   ├── test_documents.py
│   └── test_health.py
└── fixtures/
    └── sample.pdf
```

---

## Shared Test Configuration

### conftest.py

```python
import asyncio
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.database import Base, get_db

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/hgtransformer_test"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine):
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()
```

Let me explain each part:

`import asyncio` is for event loop management in async tests.

`import pytest` is the testing framework.

`from httpx import AsyncClient` is an async HTTP client for API testing.

`TEST_DATABASE_URL = "..."` is a separate test database to avoid polluting production.

`@pytest.fixture(scope="session")` creates a fixture shared across all tests in the session.

`def event_loop()` creates an event loop for async tests. `asyncio.get_event_loop_policy().new_event_loop()` creates a new loop. `yield loop` provides it to tests. `loop.close()` cleans up after all tests.

`async def test_engine()` creates the test database engine. `engine = create_async_engine(TEST_DATABASE_URL, echo=False)` creates the engine without SQL logging. `async with engine.begin() as conn` starts a transaction. `await conn.run_sync(Base.metadata.drop_all)` drops existing tables using `run_sync()` to run sync SQLAlchemy method in async context, ensuring a clean slate for tests. `await conn.run_sync(Base.metadata.create_all)` creates fresh tables. `yield engine` provides engine to tests. `await engine.dispose()` closes all connections after tests.

`@pytest.fixture` with default scope (function) means a new session per test.

`async def db_session(test_engine)` creates an isolated database session. `async_session = sessionmaker(...)` creates the session factory. `async with async_session() as session` creates the session. `yield session` provides it to the test. `await session.rollback()` undoes all changes after the test for isolation.

`async def client(db_session)` creates an API test client with DB override. `async def override_get_db(): yield db_session` is the replacement dependency. `app.dependency_overrides[get_db] = override_get_db` injects the test DB. `async with AsyncClient(app=app, base_url="http://test") as ac` creates the test client. `app=app` does direct ASGI mounting (no network). `base_url="http://test"` is the base URL for requests. `yield ac` provides the client to the test. `app.dependency_overrides.clear()` resets overrides after the test.

---

## Unit Tests

### Testing the Financial Tokenizer

```python
import pytest
from app.services.tokenizer import FinancialTokenizer


class TestFinancialTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return FinancialTokenizer()

    def test_currency_detection(self, tokenizer):
        text = "Total revenue was $1,234,567.89"
        _, entities = tokenizer.preprocess(text)
        
        currency_entities = [e for e in entities if e["type"] == "currency"]
        assert len(currency_entities) == 1
        assert currency_entities[0]["value"] == "$1,234,567.89"
        assert currency_entities[0]["normalized"] == 1234567.89

    def test_percentage_detection(self, tokenizer):
        text = "Growth rate: 15.5%"
        _, entities = tokenizer.preprocess(text)
        
        pct_entities = [e for e in entities if e["type"] == "percentage"]
        assert len(pct_entities) == 1
        assert pct_entities[0]["normalized"] == 0.155

    def test_accounting_negative_numbers(self, tokenizer):
        text = "Net loss: (500,000)"
        _, entities = tokenizer.preprocess(text)
        
        acct_entities = [e for e in entities if e["type"] == "accounting_number"]
        assert len(acct_entities) == 1
        assert acct_entities[0]["normalized"] == -500000.0

    def test_multiple_entities(self, tokenizer):
        text = "Q2 2024: Revenue $500M, up 12.5% YoY"
        _, entities = tokenizer.preprocess(text)
        
        assert len(entities) >= 3
        types = {e["type"] for e in entities}
        assert "fiscal_period" in types
        assert "percentage" in types
```

Breaking this down:

`import pytest` imports the testing framework.

`from app.services.tokenizer import FinancialTokenizer` imports the class under test.

`class TestFinancialTokenizer` is a test class that groups related tests.

`@pytest.fixture` creates a tokenizer instance for each test.

`def tokenizer(self): return FinancialTokenizer()` provides a fresh tokenizer per test.

`def test_currency_detection(self, tokenizer)` is a test method that receives the fixture.

`text = "Total revenue was $1,234,567.89"` is input with currency.

`_, entities = tokenizer.preprocess(text)` uses underscore to ignore processed text.

`currency_entities = [e for e in entities if e["type"] == "currency"]` filters to currency only.

`assert len(currency_entities) == 1` verifies exactly one currency found.

`assert currency_entities[0]["value"] == "$1,234,567.89"` verifies the original string is preserved.

`assert currency_entities[0]["normalized"] == 1234567.89` verifies correct parsing to float.

`def test_percentage_detection(...)` tests percentage recognition.

`assert pct_entities[0]["normalized"] == 0.155` verifies 15.5% becomes 0.155.

`def test_accounting_negative_numbers(...)` tests parenthetical negatives.

`assert acct_entities[0]["normalized"] == -500000.0` verifies (500,000) becomes -500000.

`def test_multiple_entities(...)` tests complex input.

`types = {e["type"] for e in entities}` creates a set comprehension for unique types.

`assert "fiscal_period" in types` verifies Q2 2024 was detected.

`assert "percentage" in types` verifies 12.5% was detected.

---

### Testing PDF Extraction

```python
import pytest
from pathlib import Path
from app.services.pdf_extractor import PDFExtractor, TableBBox


class TestPDFExtractor:
    @pytest.fixture
    def extractor(self):
        return PDFExtractor(dpi=150)

    def test_extract_basic_pdf(self, extractor, sample_pdf_path):
        result = extractor.extract(
            sample_pdf_path,
            extract_images=False,
            extract_tables=True
        )
        
        assert result.filename == sample_pdf_path.name
        assert result.page_count >= 1
        assert len(result.pages) == result.page_count

    def test_extract_nonexistent_file(self, extractor):
        with pytest.raises(FileNotFoundError):
            extractor.extract(Path("/nonexistent/file.pdf"))


class TestTableDataCleaning:
    @pytest.fixture
    def extractor(self):
        return PDFExtractor()

    def test_clean_table_removes_empty_rows(self, extractor):
        dirty_data = [
            ["Header 1", "Header 2"],
            [None, None],
            ["Value 1", "Value 2"],
            ["", ""],
        ]
        
        cleaned = extractor._clean_table_data(dirty_data)
        
        assert len(cleaned) == 2
        assert cleaned[0] == ["Header 1", "Header 2"]
        assert cleaned[1] == ["Value 1", "Value 2"]

    def test_header_detection(self, extractor):
        header_row = ["Date", "Description", "Amount"]
        data_row = ["2024-01-01", "Payment", "1234.56"]
        
        assert extractor._is_header_row(header_row) is True
        assert extractor._is_header_row(data_row) is False
```

Here's the explanation:

`class TestPDFExtractor` contains tests for the PDF extraction service.

`def extractor(self): return PDFExtractor(dpi=150)` uses lower DPI for faster tests.

`def test_extract_basic_pdf(self, extractor, sample_pdf_path)` tests basic extraction. `sample_pdf_path` is a fixture providing a test PDF file.

`assert result.filename == sample_pdf_path.name` verifies filename matches.

`assert result.page_count >= 1` verifies at least one page.

`assert len(result.pages) == result.page_count` verifies the pages list matches the count.

`def test_extract_nonexistent_file(...)` tests error handling.

`with pytest.raises(FileNotFoundError)` expects a specific exception.

`class TestTableDataCleaning` contains tests for the data cleaning helper.

`def test_clean_table_removes_empty_rows(...)` tests empty row removal.

`dirty_data = [...]` is input with None rows and empty strings.

`assert len(cleaned) == 2` verifies only valid rows remain.

`def test_header_detection(...)` tests the header heuristic.

`assert extractor._is_header_row(header_row) is True` verifies a text row is detected as header.

`assert extractor._is_header_row(data_row) is False` verifies a numeric row is detected as data.

---

## Integration Tests

### Testing the Pipeline

```python
import pytest
from uuid import uuid4
from pathlib import Path

from app.services.pipeline import DocumentPipeline, PipelineConfig
from app.models.schemas import ProcessingStatus, Document


class TestDocumentPipeline:
    @pytest.fixture
    def pipeline(self):
        config = PipelineConfig(
            use_transformer_detection=False,
            use_pdfplumber_extraction=True,
            extract_entities=True,
        )
        return DocumentPipeline(config=config)

    @pytest.mark.asyncio
    async def test_process_document_success(self, pipeline, sample_pdf_path, db_session):
        document_id = uuid4()
        
        doc = Document(
            id=document_id,
            filename="test.pdf",
            original_filename="test.pdf",
            file_path=str(sample_pdf_path),
            file_size=1024,
        )
        db_session.add(doc)
        await db_session.commit()

        result = await pipeline.process_document(
            document_id=document_id,
            pdf_path=sample_pdf_path,
            session=db_session,
        )

        assert result.status == ProcessingStatus.COMPLETED
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_document_file_not_found(self, pipeline, db_session):
        document_id = uuid4()
        
        doc = Document(
            id=document_id,
            filename="missing.pdf",
            original_filename="missing.pdf",
            file_path="/nonexistent/path.pdf",
            file_size=0,
        )
        db_session.add(doc)
        await db_session.commit()

        result = await pipeline.process_document(
            document_id=document_id,
            pdf_path=Path("/nonexistent/path.pdf"),
            session=db_session,
        )

        assert result.status == ProcessingStatus.FAILED
        assert "not found" in result.error_message.lower()
```

Breaking this down:

`class TestDocumentPipeline` contains integration tests for the full pipeline.

`def pipeline(self)` creates a pipeline with specific config.

`use_transformer_detection=False` skips ML for speed since transformer models are slow to load. Tests focus on orchestration, not ML accuracy.

`@pytest.mark.asyncio` marks the test as async.

`async def test_process_document_success(...)` tests the happy path.

`document_id = uuid4()` generates a unique ID.

`doc = Document(...)` creates a test document record.

`db_session.add(doc)` stages for insertion.

`await db_session.commit()` persists before processing.

`result = await pipeline.process_document(...)` runs the full pipeline.

`assert result.status == ProcessingStatus.COMPLETED` verifies success.

`assert result.processing_time_ms > 0` verifies timing was recorded.

`async def test_process_document_file_not_found(...)` tests the error case.

`file_path="/nonexistent/path.pdf"` is an invalid path.

`assert result.status == ProcessingStatus.FAILED` verifies proper failure marking.

`assert "not found" in result.error_message.lower()` verifies a meaningful error message.

---

## API Tests

### Testing Document Endpoints

```python
import pytest
from uuid import uuid4


class TestDocumentAPI:
    @pytest.mark.asyncio
    async def test_upload_document(self, client, sample_pdf_path):
        with open(sample_pdf_path, "rb") as f:
            response = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
            )

        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_upload_invalid_file_type(self, client):
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, client):
        fake_id = uuid4()
        response = await client.get(f"/api/v1/documents/{fake_id}")

        assert response.status_code == 404
```

Here's the explanation:

`class TestDocumentAPI` contains API endpoint tests.

`async def test_upload_document(self, client, sample_pdf_path)` tests file upload.

`with open(sample_pdf_path, "rb") as f` opens PDF in binary mode.

`response = await client.post(...)` POSTs to the upload endpoint. `files={"file": ("test.pdf", f, "application/pdf")}` is multipart file upload with tuple format: (filename, file_object, content_type).

`assert response.status_code == 202` verifies accepted for processing.

`assert "id" in data` verifies document ID returned.

`assert data["status"] == "pending"` verifies not processed yet.

`async def test_upload_invalid_file_type(...)` tests validation.

`files={"file": ("test.txt", b"not a pdf", "text/plain")}` is a non-PDF file.

`assert response.status_code == 400` verifies bad request.

`assert "PDF" in response.json()["detail"]` verifies helpful error message.

`async def test_get_document_not_found(...)` tests 404 handling.

`fake_id = uuid4()` is an ID that doesn't exist.

`assert response.status_code == 404` verifies not found.

---

## Mocking ML Models

For fast tests, mock the heavy ML components:

```python
@pytest.fixture
def mock_table_detector(mocker):
    from app.services.table_detector import DetectedTable, TableStructure
    
    mock = mocker.patch("app.services.pipeline.TableDetector")
    instance = mock.return_value
    
    instance.detect_tables.return_value = [
        DetectedTable(
            bbox=(10, 20, 200, 300),
            confidence=0.95,
            label="table",
        )
    ]
    
    instance.recognize_structure.return_value = TableStructure(
        rows=[{"bbox": (10, 20, 200, 50), "confidence": 0.9}],
        columns=[{"bbox": (10, 20, 100, 300), "confidence": 0.9}],
        cells=[],
        spanning_cells=[],
    )
    
    return instance
```

Breaking this down:

`@pytest.fixture` creates a reusable mock fixture.

`def mock_table_detector(mocker)` receives `mocker` which is pytest-mock fixture.

`from app.services.table_detector import DetectedTable, TableStructure` imports return types.

`mock = mocker.patch("app.services.pipeline.TableDetector")` replaces the class in the pipeline module. It patches where it's used, not where defined.

`instance = mock.return_value` gets the instance when the class is instantiated.

`instance.detect_tables.return_value = [...]` configures the method return with a list containing one detected table. `bbox=(10, 20, 200, 300)` uses realistic coordinates. `confidence=0.95` is high confidence.

`instance.recognize_structure.return_value = TableStructure(...)` configures the structure method, returning simple structure with one row and one column.

`return instance` provides the mock for assertions.

---

## Series Conclusion

Over these five articles, we've built a complete intelligent document processing pipeline:

1. **Architecture & Environment** — Project structure and configuration
2. **API & Core** — FastAPI, async database, and schemas
3. **Hugging Face Transformers** — Table detection deep dive
4. **Services & Pipeline** — Orchestration and business logic
5. **Testing** — Comprehensive test strategy

### Key Takeaways

**Clean Architecture** — Separation of concerns makes testing easier.

**Async Everything** — Non-blocking I/O is crucial for ML workloads.

**Lazy Loading** — Load heavy models only when needed.

**Test Pyramid** — Many unit tests, fewer integration tests.

**Mock ML Models** — Keep tests fast by mocking expensive operations.

---

*What testing strategies work best for your ML projects? Let's discuss in the comments!*

---

*Thank you for following along this series! If you found it valuable, please share it with others who might benefit.*

