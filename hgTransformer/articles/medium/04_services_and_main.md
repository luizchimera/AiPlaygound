Building an Intelligent Document Processing Pipeline — Part 4: Services & Pipeline Orchestration

PDF extraction, financial tokenization, and bringing it all together

---

We have our API layer. We have our ML models. Now it's time to connect everything into a cohesive pipeline. In this article, we'll build the PDF Extractor for text and table extraction from native PDFs, the Financial Tokenizer for domain-specific entity recognition, and the Pipeline Orchestrator to bring it all together.

This is where the magic happens.

---

## PDF Extraction Service

### The Challenge

Native PDFs contain text, but extracting tables requires understanding character positions and groupings, line detection, cell boundaries, and distinguishing table content from non-table content.

### Implementation with pdfplumber

```python
class PDFExtractor:
    def __init__(self, dpi: int = 200):
        self.dpi = dpi

    def extract(
        self,
        pdf_path: Path,
        extract_images: bool = True,
        extract_tables: bool = True,
    ) -> PDFExtractionResult:
        pages: list[PageData] = []

        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata or {}
            
            page_images = []
            if extract_images:
                page_images = self._convert_to_images(pdf_path)

            for page_idx, page in enumerate(pdf.pages):
                page_data = self._extract_page(
                    page=page,
                    page_number=page_idx + 1,
                    extract_tables=extract_tables,
                    image=page_images[page_idx] if page_images else None,
                )
                pages.append(page_data)

        return PDFExtractionResult(
            filename=pdf_path.name,
            page_count=len(pages),
            pages=pages,
            metadata=metadata,
        )
```

Let me explain each part:

`class PDFExtractor` is the service class for PDF processing.

`def __init__(self, dpi: int = 200)` is the constructor with configurable DPI. `dpi=200` sets the resolution for rendering pages to images — higher means better quality but slower.

`self.dpi = dpi` stores DPI for later use in image conversion.

`def extract(self, pdf_path, extract_images=True, extract_tables=True)` is the main extraction method with options. `extract_images` determines whether to render pages for ML processing. `extract_tables` determines whether to run table extraction.

`pages: list[PageData] = []` is the accumulator for page data.

`with pdfplumber.open(pdf_path) as pdf` opens the PDF with a context manager that automatically closes the file when done and provides access to pages, metadata, etc.

`metadata = pdf.metadata or {}` gets document metadata (author, title, etc.) or an empty dict if none exists.

`page_images = []` initializes the image list.

`if extract_images: page_images = self._convert_to_images(pdf_path)` renders all pages to PIL Images using pdf2image internally.

`for page_idx, page in enumerate(pdf.pages)` iterates through PDF pages with index.

`page_data = self._extract_page(...)` processes a single page. `page` is the pdfplumber page object. `page_number=page_idx + 1` is 1-indexed. `image=page_images[page_idx] if page_images else None` is the corresponding rendered image.

`pages.append(page_data)` adds to results.

`return PDFExtractionResult(...)` returns the structured result with all extracted data.

---

### Table Extraction from PDFs

pdfplumber has excellent built-in table detection:

```python
def _extract_tables_from_page(
    self, page: pdfplumber.page.Page, page_number: int
) -> list[ExtractedTableData]:
    tables = []
    found_tables = page.find_tables()

    for table_idx, table in enumerate(found_tables):
        table_data = table.extract()
        
        if not table_data:
            continue

        cleaned_data = self._clean_table_data(table_data)
        
        headers = None
        if cleaned_data and self._is_header_row(cleaned_data[0]):
            headers = [str(cell) if cell else "" for cell in cleaned_data[0]]

        tables.append(ExtractedTableData(
            page_number=page_number,
            table_index=table_idx,
            bbox=TableBBox(*table.bbox) if table.bbox else None,
            data=cleaned_data,
            headers=headers,
        ))

    return tables
```

Here's the breakdown:

`def _extract_tables_from_page(self, page, page_number)` extracts all tables from one page.

`tables = []` is the accumulator for extracted tables.

`found_tables = page.find_tables()` uses pdfplumber's table detection with visual cues (lines, whitespace patterns) and returns Table objects with bounding boxes.

`for table_idx, table in enumerate(found_tables)` processes each found table.

`table_data = table.extract()` extracts cell contents as a 2D list returning `[[row1_cells], [row2_cells], ...]` where cells are strings or None.

`if not table_data: continue` skips empty tables.

`cleaned_data = self._clean_table_data(table_data)` normalizes the data.

`headers = None` initializes headers.

`if cleaned_data and self._is_header_row(cleaned_data[0])` checks if the first row looks like headers.

`headers = [str(cell) if cell else "" for cell in cleaned_data[0]]` converts the header row to strings, handling None cells.

`tables.append(ExtractedTableData(...))` creates the result object. `bbox=TableBBox(*table.bbox) if table.bbox else None` unpacks the tuple into the dataclass.

---

### Data Cleaning

Real-world PDFs are messy:

```python
def _clean_table_data(self, table_data: list[list]) -> list[list]:
    cleaned = []
    
    for row in table_data:
        if row is None:
            continue
            
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            elif isinstance(cell, str):
                cleaned_row.append(" ".join(cell.split()))
            else:
                cleaned_row.append(cell)
        
        if any(cell for cell in cleaned_row):
            cleaned.append(cleaned_row)

    return cleaned
```

Breaking this down:

`def _clean_table_data(self, table_data)` normalizes raw table data.

`cleaned = []` is the result accumulator.

`for row in table_data` processes each row.

`if row is None: continue` skips null rows (sometimes returned by pdfplumber).

`cleaned_row = []` builds the cleaned row.

`for cell in row` processes each cell.

`if cell is None: cleaned_row.append("")` replaces None with empty string.

`elif isinstance(cell, str)` checks if it's a string.

`cleaned_row.append(" ".join(cell.split()))` normalizes whitespace. `cell.split()` splits on any whitespace and removes empty strings. `" ".join(...)` rejoins with single spaces. The effect: "  Multiple   spaces  " becomes "Multiple spaces".

`else: cleaned_row.append(cell)` keeps non-string values as-is.

`if any(cell for cell in cleaned_row)` checks if the row has content. `any()` returns True if any cell is truthy (non-empty), filtering out completely empty rows.

`cleaned.append(cleaned_row)` adds valid rows.

---

## Custom Financial Tokenizer

### Why Custom Tokenization?

Standard tokenizers struggle with financial text. For example, `$1,234.56` gets split into `['$', '1', ',', '234', '.', '56']` by a standard tokenizer, but our tokenizer recognizes it as a `currency` entity. Similarly, `(15.5%)` becomes `['(', '15', '.', '5', '%', ')']` normally, but we recognize it as a `percentage` in accounting negative format. And `Q3 2024` becomes `['Q', '3', '20', '24']` normally, but we recognize it as a `fiscal_period`.

### Entity Patterns

```python
PATTERNS = {
    "currency": re.compile(
        r"[$€£¥₹]\s*[\d,]+\.?\d*|\d+\.?\d*\s*(?:USD|EUR|GBP|JPY)",
        re.IGNORECASE,
    ),
    "percentage": re.compile(r"[\d,]+\.?\d*\s*%"),
    "date": re.compile(
        r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    "fiscal_period": re.compile(
        r"\b(?:Q[1-4]|FY|H[1-2])\s*\d{2,4}\b",
        re.IGNORECASE,
    ),
    "accounting_number": re.compile(r"\([\d,]+\.?\d*\)"),
    "large_number": re.compile(
        r"[\d,]+\.?\d*\s*(?:million|billion|mn|bn|M|B)\b",
        re.IGNORECASE,
    ),
}
```

Let me explain each pattern:

`PATTERNS = {...}` is a dictionary of compiled regex patterns by entity type.

**Currency pattern:** `[$€£¥₹]` matches currency symbols. `\s*` matches optional whitespace after the symbol. `[\d,]+` matches one or more digits/commas. `\.?\d*` matches an optional decimal part. The `|` means OR. `\d+\.?\d*\s*(?:USD|EUR|GBP|JPY)` matches a number followed by currency code. `re.IGNORECASE` enables case-insensitive matching.

**Percentage pattern:** `[\d,]+\.?\d*` matches a number with optional decimals/commas. `\s*%` matches optional space then percent sign.

**Date pattern:** `\d{1,2}[-/]\d{1,2}[-/]\d{2,4}` matches numeric formats like 12/31/2024 or 31-12-24. `(?:Jan|Feb|...)` matches month abbreviations. `\b` creates word boundaries to prevent partial matches.

**Fiscal period pattern:** `Q[1-4]` matches Quarter 1-4. `FY` matches Fiscal Year. `H[1-2]` matches Half 1 or 2. `\d{2,4}` matches 2 or 4 digit year.

**Accounting number pattern:** `\(` matches opening paren (escaped). `[\d,]+\.?\d*` matches the number. `\)` matches closing paren. In accounting, (1,234) means -1,234.

**Large number pattern:** `[\d,]+\.?\d*` matches the number part. `\s*` matches optional space. `(?:million|billion|mn|bn|M|B)` matches the scale indicator.

---

### Value Normalization

Convert strings to structured data:

```python
def _normalize_value(self, entity_type: str, value: str) -> Any:
    if entity_type == "currency":
        cleaned = re.sub(r"[^\d.\-]", "", value.replace(",", ""))
        return float(cleaned)

    elif entity_type == "percentage":
        cleaned = re.sub(r"[^\d.\-]", "", value)
        return float(cleaned) / 100

    elif entity_type == "accounting_number":
        cleaned = re.sub(r"[^\d.\-]", "", value.replace("(", "-").replace(")", ""))
        return float(cleaned)

    elif entity_type == "large_number":
        multipliers = {"million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}
        for name, mult in multipliers.items():
            if name in value.lower():
                num = float(re.sub(r"[^\d.]", "", value))
                return num * mult
    
    return value
```

Here's the detailed explanation:

`def _normalize_value(self, entity_type: str, value: str)` converts a string to a typed value.

**Currency handling:** `.replace(",", "")` removes thousands separators. `re.sub(r"[^\d.\-]", "", ...)` keeps only digits, decimal, and minus. "$1,234.56" becomes "1234.56". `return float(cleaned)` converts to float.

**Percentage handling:** `re.sub(r"[^\d.\-]", "", value)` extracts the number. `return float(cleaned) / 100` converts to decimal, so "15.5%" becomes 0.155, useful for calculations.

**Accounting number handling:** `.replace("(", "-").replace(")", "")` converts parens to minus, so "(1,234)" becomes "-1234". `return float(cleaned)` returns the negative number.

**Large number handling:** `multipliers = {"million": 1e6, ...}` maps words to values. `for name, mult in multipliers.items()` checks each multiplier. `if name in value.lower()` does case-insensitive check. `num = float(re.sub(r"[^\d.]", "", value))` extracts the base number. `return num * mult` applies the multiplier, so "1.5 billion" becomes 1,500,000,000.

`return value` is the fallback returning the original string.

---

## Pipeline Orchestration

### Main Processing Flow

```python
async def process_document(
    self,
    document_id: UUID,
    pdf_path: Path,
) -> ProcessingResult:
    start_time = time.time()

    try:
        await self._update_document_status(document_id, ProcessingStatus.PROCESSING)

        extraction_result = await asyncio.to_thread(
            self.pdf_extractor.extract,
            pdf_path,
            extract_images=self.config.use_transformer_detection,
            extract_tables=self.config.use_pdfplumber_extraction,
        )

        tables = await self._process_tables(extraction_result)

        await self._store_results(document_id, extraction_result, tables)

        return ProcessingResult(
            document_id=document_id,
            status=ProcessingStatus.COMPLETED,
            tables_extracted=len(tables),
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        await self._update_document_status(
            document_id, ProcessingStatus.FAILED, str(e)
        )
        return ProcessingResult(
            document_id=document_id,
            status=ProcessingStatus.FAILED,
            error_message=str(e),
        )
```

Breaking this down:

`async def process_document(self, document_id, pdf_path)` is the async method for document processing.

`start_time = time.time()` records the start for timing.

`try:` begins error handling.

`await self._update_document_status(document_id, ProcessingStatus.PROCESSING)` updates the DB status to show users that processing has started. Await ensures it completes before continuing.

`extraction_result = await asyncio.to_thread(...)` runs sync code in a thread pool. `asyncio.to_thread()` runs blocking code without blocking the event loop. PDF extraction is CPU-bound and would block async otherwise. `self.pdf_extractor.extract` is the sync method to call with parameters passed after.

`tables = await self._process_tables(extraction_result)` processes extracted tables, may run ML models, and enriches with financial entities.

`await self._store_results(document_id, extraction_result, tables)` persists to database.

`return ProcessingResult(...)` builds the success response. `status=ProcessingStatus.COMPLETED` marks it complete. `tables_extracted=len(tables)` is the count of tables found. `processing_time_ms=(time.time() - start_time) * 1000` is duration in milliseconds.

`except Exception as e` catches any error.

`await self._update_document_status(..., ProcessingStatus.FAILED, str(e))` marks as failed with error message.

`return ProcessingResult(..., status=ProcessingStatus.FAILED, error_message=str(e))` returns the error result.

---

### Database Storage

```python
async def _store_results(
    self,
    document_id: UUID,
    extraction_result: PDFExtractionResult,
    tables: list[ExtractedTableData],
) -> None:
    async with get_db_context() as session:
        document = await session.get(Document, document_id)
        document.page_count = extraction_result.page_count
        document.status = ProcessingStatus.COMPLETED

        for table_data in tables:
            extracted_table = ExtractedTable(
                document_id=document_id,
                page_number=table_data.page_number,
                table_data=table_data.data,
                headers=table_data.headers,
                num_rows=table_data.num_rows,
                num_columns=table_data.num_columns,
            )
            session.add(extracted_table)

            for row_idx, row in enumerate(table_data.data):
                for col_idx, value in enumerate(row):
                    cell = TableCell(
                        table_id=extracted_table.id,
                        row_index=row_idx,
                        column_index=col_idx,
                        value=str(value) if value else None,
                        is_header=row_idx == 0 and table_data.headers is not None,
                    )
                    session.add(cell)

        await session.commit()
```

Here's the explanation:

`async def _store_results(...)` persists all extracted data.

`async with get_db_context() as session` gets a database session with auto-commit/rollback.

`document = await session.get(Document, document_id)` fetches the document by primary key.

`document.page_count = extraction_result.page_count` updates page count.

`document.status = ProcessingStatus.COMPLETED` marks processing complete.

`for table_data in tables` processes each extracted table.

`extracted_table = ExtractedTable(...)` creates the table record. `document_id` is the foreign key to parent document. `table_data=table_data.data` stores the 2D list as JSONB. `headers` is the optional header row. `num_rows, num_columns` are dimensions for quick queries.

`session.add(extracted_table)` stages for insertion.

`for row_idx, row in enumerate(table_data.data)` iterates rows with index.

`for col_idx, value in enumerate(row)` iterates cells with column index.

`cell = TableCell(...)` creates the cell record. `table_id=extracted_table.id` — SQLAlchemy handles this FK automatically. `row_index, column_index` is the grid position. `value=str(value) if value else None` converts to string or null. `is_header=row_idx == 0 and table_data.headers is not None` is True if first row with headers.

`session.add(cell)` stages cell for insertion.

`await session.commit()` writes all changes to database in a single transaction.

---

## What's Next?

In our final article, **Part 5: Testing Strategy**, we'll cover unit tests for each service, integration tests for the pipeline, API endpoint testing, and mocking ML models.

Almost there!

---

*Building data pipelines? What orchestration patterns work best for you? Share in the comments!*

