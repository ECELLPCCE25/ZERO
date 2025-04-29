from fastapi import FastAPI
from routers.video_stream import router as video_stream_router
from routers.device import router as device_router

app = FastAPI()

app.include_router(video_stream_router)
app.include_router(device_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
