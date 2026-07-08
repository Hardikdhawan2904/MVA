"""
PostgreSQL database service.
Manages the dataset_metadata table for storing dataset metadata.
"""

import json
import logging
from typing import Any, Optional

import psycopg2
import psycopg2.extras

from app.config import settings
from app.models.schemas import DatasetMetadata

logger = logging.getLogger(__name__)

# SQL for creating the metadata table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dataset_metadata (
    dataset_id TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    upload_timestamp TEXT NOT NULL,
    business_domain TEXT DEFAULT 'Pending',
    sub_domain TEXT DEFAULT 'General',
    dataset_summary TEXT DEFAULT '',
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    column_names TEXT DEFAULT '[]',
    column_data_types TEXT DEFAULT '[]',
    column_descriptions TEXT DEFAULT '{}',
    sample_data TEXT DEFAULT '[]',
    processing_status TEXT DEFAULT 'Processing',
    quality_report TEXT DEFAULT NULL,
    quality_score REAL DEFAULT NULL
);
"""

INSERT_METADATA_SQL = """
INSERT INTO dataset_metadata (
    dataset_id, dataset_name, file_type, upload_timestamp,
    business_domain, sub_domain, dataset_summary, row_count, column_count,
    column_names, column_data_types, column_descriptions, sample_data, processing_status,
    quality_report, quality_score
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

UPDATE_METADATA_SQL = """
UPDATE dataset_metadata
SET business_domain = %s,
    sub_domain = %s,
    dataset_summary = %s,
    column_descriptions = %s,
    quality_report = %s,
    quality_score = %s,
    processing_status = %s
WHERE dataset_id = %s;
"""

UPDATE_AFTER_APPEND_SQL = """
UPDATE dataset_metadata
SET row_count = %s,
    column_count = %s,
    column_names = %s,
    column_data_types = %s,
    upload_timestamp = %s,
    sample_data = %s,
    processing_status = %s
WHERE dataset_id = %s;
"""

SELECT_METADATA_SQL = """
SELECT * FROM dataset_metadata WHERE dataset_id = %s;
"""

SELECT_ALL_SQL = """
SELECT * FROM dataset_metadata ORDER BY upload_timestamp DESC;
"""

GET_NEXT_ID_SQL = """
SELECT COUNT(*) AS count FROM dataset_metadata;
"""


def _get_connection() -> psycopg2.extensions.connection:
    """Create a new PostgreSQL connection."""
    conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        cursor_factory=psycopg2.extras.RealDictCursor,
        options="-c search_path=agent1",
    )
    return conn


def init_db() -> None:
    """Initialize the database and create tables if they don't exist."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("CREATE SCHEMA IF NOT EXISTS agent1")
        cur.execute(CREATE_TABLE_SQL)
        # Migrations: add columns to existing tables that predate them
        cur.execute("ALTER TABLE dataset_metadata ADD COLUMN IF NOT EXISTS sub_domain TEXT DEFAULT 'General'")
        cur.execute("ALTER TABLE dataset_metadata ADD COLUMN IF NOT EXISTS column_names TEXT DEFAULT '[]'")
        cur.execute("ALTER TABLE dataset_metadata ADD COLUMN IF NOT EXISTS column_data_types TEXT DEFAULT '[]'")
        cur.execute("ALTER TABLE dataset_metadata ADD COLUMN IF NOT EXISTS quality_report TEXT DEFAULT NULL")
        cur.execute("ALTER TABLE dataset_metadata ADD COLUMN IF NOT EXISTS quality_score REAL DEFAULT NULL")
        conn.commit()
        cur.close()
        conn.close()
        logger.info(
            f"Database initialized at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_next_dataset_id() -> str:
    """
    Generate the next sequential dataset ID.

    Returns:
        A string like 'DS_001', 'DS_002', etc.
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(GET_NEXT_ID_SQL)
    count = cur.fetchone()["count"]
    cur.close()
    conn.close()
    return f"DS_{count + 1:03d}"


def insert_metadata(metadata: DatasetMetadata) -> None:
    """
    Insert a new dataset metadata record into the database.

    Args:
        metadata: The DatasetMetadata object to store.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        mva_json = None
        if metadata.quality_report:
            if hasattr(metadata.quality_report, "model_dump"):
                mva_json = json.dumps(metadata.quality_report.model_dump())
            elif isinstance(metadata.quality_report, dict):
                mva_json = json.dumps(metadata.quality_report)
            else:
                mva_json = json.dumps(dict(metadata.quality_report))
        score_val = float(metadata.quality_score) if metadata.quality_score is not None else None
        cur.execute(INSERT_METADATA_SQL, (
            metadata.dataset_id,
            metadata.dataset_name,
            metadata.file_type,
            metadata.upload_timestamp,
            metadata.business_domain,
            metadata.sub_domain,
            metadata.dataset_summary,
            metadata.row_count,
            metadata.column_count,
            json.dumps(metadata.column_names),
            json.dumps(metadata.column_data_types),
            json.dumps(metadata.column_descriptions),
            json.dumps(metadata.sample_data, default=str),
            metadata.processing_status,
            mva_json,
            score_val,
        ))
        conn.commit()
        cur.close()
        logger.info(f"Inserted metadata for dataset {metadata.dataset_id}")
    except Exception as e:
        logger.error(f"Failed to insert metadata for {metadata.dataset_id}: {e}")
        raise
    finally:
        conn.close()


def update_metadata_after_classification(
    dataset_id: str,
    business_domain: str,
    sub_domain: str,
    dataset_summary: str,
    column_descriptions: dict[str, str],
    quality_score: Optional[float] = None,
    quality_report: Optional[Any] = None,
    processing_status: str = "Completed",
) -> None:
    """
    Update metadata after LLM classification and analysis completes.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        mva_json = None
        if quality_report:
            if hasattr(quality_report, "model_dump"):
                mva_json = json.dumps(quality_report.model_dump())
            elif isinstance(quality_report, dict):
                mva_json = json.dumps(quality_report)
            else:
                mva_json = json.dumps(dict(quality_report))

        score_val = float(quality_score) if quality_score is not None else None

        cur.execute(UPDATE_METADATA_SQL, (
            business_domain,
            sub_domain,
            dataset_summary,
            json.dumps(column_descriptions),
            mva_json,
            score_val,
            processing_status,
            dataset_id,
        ))
        conn.commit()
        cur.close()
        logger.info(f"Updated metadata for dataset {dataset_id}: domain={business_domain}, sub_domain={sub_domain}")
    except Exception as e:
        logger.error(f"Failed to update metadata for {dataset_id}: {e}")
        raise
    finally:
        conn.close()


def _row_to_metadata(row: dict) -> DatasetMetadata:
    """Convert a database row to a DatasetMetadata object."""
    from app.models.schemas import QualityReport

    mva_raw = row.get("quality_report")
    quality_report = None
    if mva_raw:
        try:
            quality_report = QualityReport(**json.loads(mva_raw))
        except Exception as e:
            logger.warning(f"Failed to deserialize quality_report for dataset {row['dataset_id']}: {e}")

    score_val = row.get("quality_score")

    return DatasetMetadata(
        dataset_id=row["dataset_id"],
        dataset_name=row["dataset_name"],
        file_type=row["file_type"],
        upload_timestamp=row["upload_timestamp"],
        business_domain=row["business_domain"],
        sub_domain=row["sub_domain"] if row["sub_domain"] else "General",
        dataset_summary=row["dataset_summary"],
        row_count=row["row_count"],
        column_count=row["column_count"],
        column_names=json.loads(row["column_names"]) if row["column_names"] else [],
        column_data_types=json.loads(row["column_data_types"]) if row["column_data_types"] else [],
        column_descriptions=json.loads(row["column_descriptions"]),
        sample_data=json.loads(row["sample_data"]),
        processing_status=row["processing_status"],
        quality_score=score_val,
        quality_report=quality_report,
    )


def get_metadata(dataset_id: str) -> Optional[DatasetMetadata]:
    """
    Retrieve metadata for a specific dataset.

    Args:
        dataset_id: The dataset identifier.

    Returns:
        DatasetMetadata or None if not found.
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(SELECT_METADATA_SQL, (dataset_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        return None
    return _row_to_metadata(row)


def list_all_metadata() -> list[DatasetMetadata]:
    """
    List all dataset metadata records, ordered by upload time (newest first).
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(SELECT_ALL_SQL)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [_row_to_metadata(row) for row in rows]


def get_metadata_by_name(filename: str) -> Optional[DatasetMetadata]:
    """
    Retrieve metadata for a dataset by its filename.
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM dataset_metadata WHERE dataset_name = %s LIMIT 1;", (filename,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        return None
    return _row_to_metadata(row)


def update_metadata_after_append(
    dataset_id: str,
    row_count: int,
    column_count: int,
    column_names: list[str],
    column_data_types: list[str],
    upload_timestamp: str,
    sample_data: list[dict],
    processing_status: str = "Completed",
) -> None:
    """Update metadata fields after appending new rows to an in-memory dataset."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(UPDATE_AFTER_APPEND_SQL, (
            row_count,
            column_count,
            json.dumps(column_names),
            json.dumps(column_data_types),
            upload_timestamp,
            json.dumps(sample_data, default=str),
            processing_status,
            dataset_id,
        ))
        conn.commit()
        cur.close()
        logger.info(f"Updated metadata after append for dataset {dataset_id}: row_count={row_count}")
    except Exception as e:
        logger.error(f"Failed to update metadata after append for {dataset_id}: {e}")
        raise
    finally:
        conn.close()
