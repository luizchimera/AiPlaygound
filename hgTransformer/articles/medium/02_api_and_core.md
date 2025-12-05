Building an Intelligent Document Processing Pipeline — Part 2: API & Core Infrastructure

FastAPI, async database connections, and dependency injection patterns

---

In Part 1, we explored the architecture of hgTransformer—our intelligent document processing pipeline. Now, let's build the foundation: the FastAPI application and core infrastructure.

A solid foundation is crucial. We need async database connections that scale, clean API design with proper error handling, dependency injection for testability, and proper lifecycle management.

Let's dive in.

---

## Database Layer

### Async SQLAlchemy Setup

Modern Python applications demand async I/O. Here's our database module:

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)
```

Let me explain each line:

**Imports:**

`from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine` imports SQLAlchemy's async components for non-blocking database operations. `from sqlalchemy.orm import DeclarativeBase` imports the base class for defining ORM models.

**Engine creation:**

`engine = create_async_engine(...)` creates the async database engine. `settings.database_url` is the connection string from our config (e.g., `postgresql+asyncpg://...`). `echo=settings.debug` logs all SQL queries to console when True — useful for debugging. `pool_pre_ping=True` tests connections before use, preventing "connection closed" errors after idle periods. `pool_size=5` maintains 5 persistent connections in the pool. `max_overflow=10` allows up to 10 additional connections during high load (total max: 15).

**Session factory:**

`async_session_maker = async_sessionmaker(...)` creates a factory for database sessions. `engine` uses our async engine. `class_=AsyncSession` creates AsyncSession instances (not regular Sessions). `expire_on_commit=False` keeps objects accessible after commit, preventing extra queries when returning data. `autocommit=False` requires explicit commits, giving us transaction control. `autoflush=False` disables automatic flushing — we control when changes hit the database.

---

### Session Management

Here's the pattern for dependency injection:

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

Breaking this down:

`async def get_db() -> AsyncGenerator[AsyncSession, None]` is an async generator function that yields database sessions. The return type indicates it yields AsyncSession objects.

`async with async_session_maker() as session` creates a new session using the factory. The `async with` ensures proper cleanup.

`try:` begins the exception handling block.

`yield session` pauses execution and returns the session to the caller (FastAPI route).

`await session.commit()` commits any pending changes after the route completes successfully.

`except Exception:` catches any error that occurred in the route.

`await session.rollback()` undoes all uncommitted changes on error.

`raise` re-raises the exception so FastAPI can handle it and return an error response.

`finally:` runs regardless of success or failure.

`await session.close()` releases the connection back to the pool.

---

## Data Models

### SQLAlchemy ORM Models

Our schema captures documents, tables, and cells:

```python
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus), default=ProcessingStatus.PENDING
    )
    
    tables: Mapped[list["ExtractedTable"]] = relationship(
        "ExtractedTable", back_populates="document", cascade="all, delete-orphan"
    )
```

Here's what each line does:

`class Document(Base)` defines a Document model inheriting from SQLAlchemy's declarative base.

`__tablename__ = "documents"` sets the PostgreSQL table name.

`id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)` creates the UUID primary key. `Mapped[UUID]` is the type hint for Python UUID type. `PG_UUID(as_uuid=True)` uses PostgreSQL-native UUID column and returns Python UUID objects (not strings). `primary_key=True` makes this the table's primary key. `default=uuid4` automatically generates a new UUID for each record.

`filename: Mapped[str] = mapped_column(String(255), nullable=False)` creates a required string column with max 255 characters.

`original_filename: Mapped[str]` stores the user's original filename (preserved for display).

`file_path: Mapped[str]` stores where the file is saved on disk.

`status: Mapped[ProcessingStatus] = mapped_column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)` handles processing state. `Enum(ProcessingStatus)` maps Python enum to PostgreSQL enum type. `default=ProcessingStatus.PENDING` means new documents start as "pending".

`tables: Mapped[list["ExtractedTable"]] = relationship(...)` creates a one-to-many relationship. The string `"ExtractedTable"` is a forward reference avoiding circular imports. `back_populates="document"` makes it bidirectional — ExtractedTable has a `document` attribute. `cascade="all, delete-orphan"` automatically deletes all tables when a document is deleted.

---

### Pydantic Schemas

API request/response models separate from ORM:

```python
class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    original_filename: str
    status: ProcessingStatus
    created_at: datetime
```

Breaking this down:

`class DocumentResponse(BaseModel)` creates a Pydantic model for API responses that handles validation and serialization.

`model_config = ConfigDict(from_attributes=True)` enables ORM mode, allowing you to create this model directly from SQLAlchemy objects using `DocumentResponse.model_validate(orm_object)`.

`id: UUID` is a required UUID field — Pydantic validates the type.

`filename: str` is a required string field.

`original_filename: str` is the name the user uploaded the file with.

`status: ProcessingStatus` is an enum field that serializes to string in JSON.

`created_at: datetime` is a timestamp that Pydantic converts to ISO format in JSON.

---

## FastAPI Application

### Application Factory Pattern

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    yield
    await close_db()

def create_app() -> FastAPI:
    app = FastAPI(
        title="hgTransformer",
        description="Intelligent document processing pipeline...",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)

    return app
```

Here's the explanation:

`from contextlib import asynccontextmanager` imports the decorator to create async context managers.

`@asynccontextmanager` marks the function as an async context manager.

`async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]` handles app startup and shutdown.

`await init_db()` runs on startup — creates database tables if they don't exist.

`yield` suspends here while the application runs. Everything before is startup, everything after is shutdown.

`await close_db()` runs on shutdown — closes database connections cleanly.

`def create_app() -> FastAPI` is a factory function that creates and configures the app.

`app = FastAPI(...)` creates the FastAPI application. `title="hgTransformer"` shows in OpenAPI docs. `description="..."` provides API description for documentation. `version="0.1.0"` displays API version in docs. `lifespan=lifespan` attaches our startup/shutdown handler.

`app.add_middleware(CORSMiddleware, ...)` enables Cross-Origin Resource Sharing. `allow_origins=settings.cors_origins` defines which domains can call this API. `allow_credentials=True` allows cookies/auth headers. `allow_methods=["*"]` allows all HTTP methods (GET, POST, etc.). `allow_headers=["*"]` allows all request headers.

`app.include_router(health.router, prefix=settings.api_prefix)` mounts health routes under `/api/v1/`.

`app.include_router(documents.router, prefix=settings.api_prefix)` mounts document routes.

`return app` returns the configured application.

---

## API Endpoints

### Document Upload

The core endpoint with background processing:

```python
@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_database),
    settings: Settings = Depends(get_app_settings),
) -> UploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum: {settings.max_file_size_mb}MB",
        )

    document_id = uuid4()
    file_path = settings.upload_dir / f"{document_id}.pdf"
    
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    document = Document(id=document_id, ...)
    db.add(document)
    await db.commit()

    background_tasks.add_task(process_document_background, document_id, file_path)

    return UploadResponse(
        id=document_id,
        status=ProcessingStatus.PENDING,
        message="Processing will begin shortly.",
    )
```

Let me walk through each part:

`@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)` creates a POST endpoint returning 202 (accepted for processing, not yet complete).

`async def upload_document(...)` is the async route handler with dependency injection. `background_tasks: BackgroundTasks` is FastAPI's background task queue. `file: UploadFile = File(...)` is the uploaded file where `File(...)` means required. `db: AsyncSession = Depends(get_database)` is the injected database session. `settings: Settings = Depends(get_app_settings)` is the injected configuration.

`if not file.filename.lower().endswith(".pdf")` validates file extension (case-insensitive).

`raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, ...)` returns a 400 error with message.

`content = await file.read()` reads the entire file into memory asynchronously.

`if len(content) > settings.max_file_size_mb * 1024 * 1024` checks file size in bytes.

`raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, ...)` returns 413 if too large.

`document_id = uuid4()` generates a unique ID for this document.

`file_path = settings.upload_dir / f"{document_id}.pdf"` constructs the save path using pathlib.

`async with aiofiles.open(file_path, "wb") as f` opens the file for async binary writing.

`await f.write(content)` writes content without blocking the event loop.

`document = Document(id=document_id, ...)` creates the ORM object.

`db.add(document)` stages it for insertion.

`await db.commit()` persists to database.

`background_tasks.add_task(process_document_background, document_id, file_path)` schedules processing after the response is sent.

`return UploadResponse(...)` returns immediately while processing happens in background.

---

## Dependency Injection

Clean dependencies for testability:

```python
async def get_database() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db():
        yield session

def get_app_settings() -> Settings:
    return get_settings()

def get_pipeline() -> DocumentPipeline:
    config = PipelineConfig()
    return DocumentPipeline(config=config)
```

Here's what each does:

`async def get_database() -> AsyncGenerator[AsyncSession, None]` is an async generator that yields database sessions.

`async for session in get_db()` iterates over the get_db generator (which yields exactly once).

`yield session` passes the session to the route handler.

`def get_app_settings() -> Settings` is a simple function returning the settings singleton.

`return get_settings()` returns the cached Settings instance (uses `@lru_cache`).

`def get_pipeline() -> DocumentPipeline` creates a new pipeline for each request.

`config = PipelineConfig()` creates default configuration.

`return DocumentPipeline(config=config)` returns the configured pipeline instance.

The benefits are clear: easy to mock in tests by overriding dependencies, configuration centralized in one place, and clear function signatures showing what each route needs.

---

## What We've Built

We now have an async PostgreSQL database layer with connection pooling, clean SQLAlchemy ORM models with Pydantic schemas, a FastAPI application with proper lifecycle management, RESTful endpoints with background processing, and dependency injection for clean, testable code.

---

## Coming Up: Part 3

In the next article, we'll explore the heart of our system: **Hugging Face Transformers for Table Detection**. We'll cover Table Transformer architecture, detection vs. structure recognition models, and optimizing inference for production.

See you there!

---

*Questions about async Python or FastAPI patterns? Drop them in the comments!*

