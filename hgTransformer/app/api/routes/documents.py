"""Document processing API endpoints."""

import shutil
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.dependencies import get_database, get_app_settings, get_pipeline
from app.core.config import Settings
from app.models.schemas import (
    Document,
    DocumentResponse,
    DocumentStatusResponse,
    DocumentTablesResponse,
    ErrorResponse,
    ExtractedTable,
    ExtractedTableResponse,
    ProcessingStatus,
    UploadResponse,
)
from app.services.pipeline import DocumentPipeline, process_document_background

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_database),
    settings: Settings = Depends(get_app_settings),
    pipeline: DocumentPipeline = Depends(get_pipeline),
) -> UploadResponse:
    """
    Upload a PDF document for table extraction.

    The document will be processed asynchronously. Use the status endpoint
    to check processing progress.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    # Validate content type
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type. Expected application/pdf",
        )

    # Read file to check size
    content = await file.read()
    file_size = len(content)

    max_size = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB",
        )

    # Generate unique filename
    document_id = uuid4()
    safe_filename = f"{document_id}.pdf"
    file_path = settings.upload_dir / safe_filename

    # Save file
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Create document record
    document = Document(
        id=document_id,
        filename=safe_filename,
        original_filename=file.filename,
        file_path=str(file_path),
        file_size=file_size,
        mime_type="application/pdf",
        status=ProcessingStatus.PENDING,
    )
    db.add(document)
    await db.commit()

    # Schedule background processing
    background_tasks.add_task(process_document_background, document_id, file_path)

    return UploadResponse(
        id=document_id,
        filename=file.filename,
        status=ProcessingStatus.PENDING,
        message="Document uploaded successfully. Processing will begin shortly.",
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_database),
) -> DocumentResponse:
    """Get document metadata by ID."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def get_document_status(
    document_id: UUID,
    db: AsyncSession = Depends(get_database),
) -> DocumentStatusResponse:
    """Check the processing status of a document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Count extracted tables
    count_result = await db.execute(
        select(func.count(ExtractedTable.id)).where(
            ExtractedTable.document_id == document_id
        )
    )
    tables_count = count_result.scalar() or 0

    return DocumentStatusResponse(
        id=document.id,
        status=document.status,
        error_message=document.error_message,
        tables_extracted=tables_count,
    )


@router.get(
    "/{document_id}/tables",
    response_model=DocumentTablesResponse,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def get_document_tables(
    document_id: UUID,
    page: Optional[int] = None,
    db: AsyncSession = Depends(get_database),
) -> DocumentTablesResponse:
    """
    Get all extracted tables from a document.

    Optionally filter by page number.
    """
    # Get document with tables
    query = (
        select(Document)
        .options(selectinload(Document.tables))
        .where(Document.id == document_id)
    )
    result = await db.execute(query)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Filter tables by page if specified
    tables = document.tables
    if page is not None:
        tables = [t for t in tables if t.page_number == page]

    # Convert to response format
    table_responses = [
        ExtractedTableResponse(
            id=t.id,
            page_number=t.page_number,
            table_index=t.table_index,
            num_rows=t.num_rows,
            num_columns=t.num_columns,
            confidence=t.confidence,
            table_data=t.table_data,
            headers=t.headers,
        )
        for t in sorted(tables, key=lambda x: (x.page_number, x.table_index))
    ]

    return DocumentTablesResponse(
        document_id=document.id,
        filename=document.original_filename,
        status=document.status,
        tables=table_responses,
    )


@router.get(
    "/{document_id}/tables/{table_id}",
    response_model=ExtractedTableResponse,
    responses={404: {"model": ErrorResponse, "description": "Table not found"}},
)
async def get_table(
    document_id: UUID,
    table_id: UUID,
    db: AsyncSession = Depends(get_database),
) -> ExtractedTableResponse:
    """Get a specific extracted table by ID."""
    result = await db.execute(
        select(ExtractedTable).where(
            ExtractedTable.id == table_id,
            ExtractedTable.document_id == document_id,
        )
    )
    table = result.scalar_one_or_none()

    if not table:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table {table_id} not found in document {document_id}",
        )

    return ExtractedTableResponse(
        id=table.id,
        page_number=table.page_number,
        table_index=table.table_index,
        num_rows=table.num_rows,
        num_columns=table.num_columns,
        confidence=table.confidence,
        table_data=table.table_data,
        headers=table.headers,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_database),
    settings: Settings = Depends(get_app_settings),
) -> None:
    """Delete a document and all its extracted data."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Delete the file
    file_path = Path(document.file_path)
    if file_path.exists():
        file_path.unlink()

    # Delete the database record (cascade will delete tables and cells)
    await db.delete(document)
    await db.commit()


@router.get(
    "",
    response_model=list[DocumentResponse],
)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    status_filter: Optional[ProcessingStatus] = None,
    db: AsyncSession = Depends(get_database),
) -> list[DocumentResponse]:
    """List all documents with optional filtering."""
    query = select(Document).order_by(Document.created_at.desc())

    if status_filter:
        query = query.where(Document.status == status_filter)

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    documents = result.scalars().all()

    return [DocumentResponse.model_validate(doc) for doc in documents]

