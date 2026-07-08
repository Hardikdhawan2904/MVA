"""
File validation service.
Validates uploaded file type, readability, and DataFrame structure.
"""

from fastapi import HTTPException, UploadFile

import pandas as pd

# Supported file extensions
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def get_file_extension(filename: str) -> str:
    """Extract and return the lowercase file extension."""
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def validate_file_type(filename: str) -> str:
    """
    Validate that the uploaded file has a supported extension.
    """
    ext = get_file_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Accepted formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return ext


def validate_dataframe(df: pd.DataFrame, filename: str) -> None:
    """
    Validate that the loaded DataFrame has valid tabular structure.
    """
    if df is None:
        raise HTTPException(
            status_code=422,
            detail=f"File '{filename}' could not be parsed into a valid table."
        )

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded dataset '{filename}' is empty (0 rows)."
        )

    if len(df.columns) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"File '{filename}' does not contain valid tabular data (no columns detected)."
        )


def validate_upload_file(file: UploadFile) -> str:
    """
    Validate the UploadFile object itself.
    """
    if file is None or file.filename is None or file.filename.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="No file provided or filename is missing."
        )

    return validate_file_type(file.filename)
