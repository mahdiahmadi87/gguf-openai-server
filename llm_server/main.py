import logging
import os
import time 
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware # Optional: If needed for frontend access
from contextlib import asynccontextmanager
import psutil

from config.settings import settings, get_available_model_ids, get_model_config
from llm_server.core.api_key_manager import load_api_keys
from llm_server.api.v1 import chat_completions, completions, models, admin
from llm_server.core.llm_manager import model_manager, set_cpu_affinity

# --- Logging Setup ---
logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger(settings.APP_NAME)
logger.info("Starting logger...")

def set_main_process_cpu_affinity():
    """Sets the CPU affinity for the main FastAPI server process."""
    try:
        total_cores = psutil.cpu_count(logical=True)
        all_cores = set(range(total_cores))

        # Determine cores used by model workers
        model_cores = set()
        model_ids = get_available_model_ids()
        for model_id in model_ids:
            model_config = get_model_config(model_id)
            if model_config:
                start = model_config.thread_offset
                end = start + model_config.n_threads
                model_cores.update(range(start, end))

        # Determine cores available for the main process
        available_cores = all_cores - model_cores

        if not available_cores:
            logger.warning("No exclusive CPU cores available for the main server process. Affinity will not be set.")
            return

        pid = os.getpid()
        set_cpu_affinity(available_cores)
        logger.info(f"[PID {pid}] Main server process affinity set to cores: {available_cores}")

    except Exception as e:
        logger.error(f"Failed to set main process CPU affinity. Error: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code to run on startup ---
    logger.info("Server is starting up...")

    # Set CPU affinity for the main process BEFORE starting model workers
    set_main_process_cpu_affinity()

    # Start the model worker processes
    model_manager.start_models()

    # Load user API keys from the persistent file
    load_api_keys()

    yield

    # --- Code to run on shutdown ---
    logger.info("Server is shutting down...")
    model_manager.stop_models()


# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan # <-- Use the new lifespan manager
)

# --- Middleware ---
# Optional: Add CORS middleware if your API needs to be called from different domains (e.g., a web UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or specify allowed origins: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Optional: Add logging middleware (example)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    # Log basic request info before processing
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        # Log response info after processing
        logger.info(f"Response: {response.status_code} Process Time: {process_time:.4f}s")
        # Log token usage if available in response headers or content? More complex.
        return response
    except Exception as e:
         process_time = time.time() - start_time
         logger.error(f"Request failed: {request.method} {request.url.path} Error: {e} Process Time: {process_time:.4f}s")
         # Reraise the exception to let FastAPI handle it or return a generic 500
         raise e # Or return JSONResponse(...)

# --- Exception Handlers ---
# Default validation exception handler override (optional, for custom formatting)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the validation error details
    logger.warning(f"Request validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body}, # OpenAI often returns error objects
    )

# --- Routers ---
# Include routers for different API versions/sections
# The prefix ensures all routes in these modules start with /v1
app.include_router(admin.router, prefix=settings.BASE_PATH)
app.include_router(models.router, prefix=settings.BASE_PATH)
app.include_router(completions.router, prefix=settings.BASE_PATH)
app.include_router(chat_completions.router, prefix=settings.BASE_PATH)


# --- Root Endpoint ---
@app.get("/", include_in_schema=False) # Exclude from OpenAPI docs
async def root():
    return {"message": f"{settings.APP_NAME} is running."}

# --- Run Command ---
# This part is for running directly with `python llm_server/main.py`
# In production, you'd use `uvicorn llm_server.main:app --host ... --port ...`
if __name__ == "__main__":
    import uvicorn
    import time # Import time here if using logging middleware

    logger.info(f"Starting Uvicorn server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "llm_server.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False # Set reload=True only for development (watches for file changes)
    )