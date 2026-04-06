import os
import uuid
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from config import safe_path, UPLOAD_DIR, OUTPUT_DIR
from api.state import jobs
from api import routes


app = FastAPI(title="Autonomous Data Scientist")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Root — serve frontend
# ------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ------------------------------------------------------------------
# Upload + trigger pipeline
# ------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze(
    file:        UploadFile = File(...),
    user_prompt: str        = Form(...),
):
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Save uploaded file
    job_id   = str(uuid.uuid4())[:8]
    job_dir  = safe_path(OUTPUT_DIR, job_id)
    csv_path = safe_path(UPLOAD_DIR, f"{job_id}_{file.filename}")

    os.makedirs(job_dir, exist_ok=True)

    contents = await file.read()
    with open(csv_path, "wb") as f:
        f.write(contents)

    # Register job
    jobs[job_id] = {
        "status":      "queued",
        "csv_path":    csv_path,
        "user_prompt": user_prompt,
        "job_dir":     job_dir,
        "events":      [],
        "result":      None,
        "error":       None,
    }

    # Run pipeline in background
    asyncio.create_task(_run_pipeline(job_id))

    return JSONResponse({"job_id": job_id})


# ------------------------------------------------------------------
# SSE stream — client listens here for progress events
# ------------------------------------------------------------------

@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def event_generator():
        sent     = 0
        finished = False

        while not finished:
            job    = jobs[job_id]
            events = job["events"]

            # Send any new events
            while sent < len(events):
                event = events[sent]
                data  = json.dumps(event)
                yield f"data: {data}\n\n"
                sent += 1

                # If this event itself is the terminal one, stop after sending it
                if event.get("type") in ("done", "failed"):
                    finished = True
                    break

            if not finished:
                await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

# ------------------------------------------------------------------
# Get job result
# ------------------------------------------------------------------

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    return JSONResponse({
        "status": job["status"],
        "result": job["result"],
        "error":  job["error"],
    })


# ------------------------------------------------------------------
# Serve output files (plots etc)
# ------------------------------------------------------------------

@app.get("/api/file/{job_id}/{filename}")
async def get_file(job_id: str, filename: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    # Sanitize filename — no path traversal
    filename  = Path(filename).name
    file_path = safe_path(OUTPUT_DIR, job_id, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")

    return FileResponse(file_path)


# ------------------------------------------------------------------
# Background pipeline runner
# ------------------------------------------------------------------
async def _run_pipeline(job_id: str):
    job = jobs[job_id]

    def push(event_type: str, message: str, data: dict = None):
        event = {
            "type":    event_type,
            "message": message,
            "data":    data or {},
        }
        job["events"].append(event)
        print(f"[SSE] {job_id}: {event_type} — {message}")

    try:
        job["status"] = "running"
        push("start", "Pipeline started")

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _run_pipeline_sync,
            job_id,
            job["csv_path"],
            job["user_prompt"],
            job["job_dir"],
            push,
        )

        # Sanitize result — remove non-serializable values
        import json
        result_clean = json.loads(json.dumps(result, default=str))

        job["result"] = result_clean
        job["status"] = "done"
        print(f"[SSE] {job_id}: pushing done event, result keys: {list(result_clean.keys())}")
        push("done", "Pipeline complete", result_clean)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[SSE] {job_id}: pipeline error:\n{tb}")
        job["error"]  = str(e)
        job["status"] = "failed"
        push("failed", f"Pipeline failed: {str(e)}")

def _run_pipeline_sync(
    job_id:      str,
    csv_path:    str,
    user_prompt: str,
    job_dir:     str,
    push,
) -> dict:
    """
    Runs the orchestrator synchronously in a thread.
    Uses a progress-aware orchestrator that pushes SSE events.
    """
    from agents.orchestrator import Orchestrator

    # Patch orchestrator to push SSE events at each step
    orch = Orchestrator(use_cache=False)

    # Monkey-patch each agent's run to emit progress events
    _wrap_agent(orch.eda_agent,      "EDA",                 push)
    _wrap_agent(orch.cleaning_agent, "Cleaning",            push)
    _wrap_agent(orch.feature_agent,  "Feature Engineering", push)
    _wrap_agent(orch.model_agent,    "Model Training",      push)
    _wrap_agent(orch.explainer,      "Explanation",         push)

    report = orch.run(csv_path, user_prompt, output_dir=job_dir)

    # Remap file paths to API endpoints
    report["api_files"] = {
        "shap_plot":   f"/api/file/{job_id}/shap_summary.png",
        "report_json": f"/api/file/{job_id}/report.json",
    }

    return report


def _wrap_agent(agent, step_name: str, push):
    """Wrap agent.run() to emit SSE events before and after."""
    original_run = agent.run

    def wrapped(*args, **kwargs):
        push("step_start", f"{step_name} started", {"step": step_name})
        result = original_run(*args, **kwargs)
        push("step_done",  f"{step_name} complete", {"step": step_name})
        return result

    agent.run = wrapped
    




app.include_router(routes.router)