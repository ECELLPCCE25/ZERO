from fastapi import FastAPI
from routers.video_stream import router as video_stream_router
from routers.device import router as device_router
from routers.analytics import router as analytic_router
from fastapi.middleware.cors import CORSMiddleware
from routers.motion_flow import router as motion_flow_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],  # List of allowed origins
    allow_credentials=True,  # Allow cookies and authentication credentials
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Allowed HTTP methods
    allow_headers=[
        "*"
    ],  # Allow all headers or specify a list like ["Content-Type", "Authorization"]
)

app.include_router(video_stream_router)
app.include_router(device_router)
app.include_router(analytic_router)
app.include_router(motion_flow_router, prefix="/motion-flow")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
