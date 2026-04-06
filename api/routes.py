from fastapi import APIRouter
from fastapi.responses import JSONResponse
from api.state import jobs
router = APIRouter()

@router.get("/api/health")
async def health():
    return JSONResponse({"status": "ok", "jobs": len(jobs)})

@router.get("/api/jobs")
async def list_jobs():
    summary = {
        jid: {
            "status":      j["status"],
            "user_prompt": j["user_prompt"],
            "error":       j["error"],
        }
        for jid, j in jobs.items()
    }
    return JSONResponse(summary)