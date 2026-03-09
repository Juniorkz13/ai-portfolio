from fastapi import APIRouter, File, UploadFile

router = APIRouter()


@router.post("/")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str]:
    # Placeholder: persist PDF file, register metadata, and trigger processing pipeline.
    return {
        "message": "Upload route scaffolded.",
        "filename": file.filename or "unknown.pdf",
    }
