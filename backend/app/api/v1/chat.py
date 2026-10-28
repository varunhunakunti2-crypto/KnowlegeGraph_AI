"""Chat API routes."""
from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def chat():
    return {"status": "pending", "message": "Not implemented"}
