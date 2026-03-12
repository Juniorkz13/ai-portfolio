from fastapi import APIRouter

from app.api.categories import router as categories_router
from app.api.messages import router as messages_router
from app.api.summary import router as summary_router
from app.api.transactions import router as transactions_router
from app.api.users import router as users_router

api_router = APIRouter(prefix="/api")
api_router.include_router(users_router)
api_router.include_router(transactions_router)
api_router.include_router(summary_router)
api_router.include_router(categories_router)
api_router.include_router(messages_router)
