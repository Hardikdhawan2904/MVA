"""
Dataset loader service.
Reads CSV and Excel files into Pandas DataFrames.
"""

import io
from fastapi import HTTPException, UploadFile
import pandas as pd

from app.services.validator import get_file_extension


async def load_dataset(file: UploadFile) -> pd.DataFrame:
    """
    Load the uploaded file into a Pandas DataFrame.
    """
    try:
        # Read file content into memory once
        content = await file.read()
        buffer = io.BytesIO(content)

        ext = get_file_extension(file.filename or "")

        if ext == ".csv":
            df = pd.read_csv(buffer)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(buffer, engine="openpyxl")
        else:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file extension: {ext}"
            )

        return df

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read the uploaded file: {str(e)}"
        )
