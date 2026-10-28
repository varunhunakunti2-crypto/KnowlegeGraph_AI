"""Graph API routes."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_graph():
    return {"nodes": [], "edges": []}
