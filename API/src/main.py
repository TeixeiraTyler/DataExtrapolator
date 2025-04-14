from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as api_router
from src.container import Container as AppContainer

# Define the origins allowed to make requests (your frontend URL)
# IMPORTANT: Change 'http://localhost:3000' if your frontend runs on a different port
origins = [
    "http://localhost:3000",
    # You can add more origins here if needed, e.g., your deployed frontend URL
]

app = FastAPI(
    title="Data Extrapolator API",
    description="API for machine learning predictions using scikit-learn models",
    version="1.0.0"
)

# Add the CORS middleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True, # Allow cookies to be included in requests
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

app.container = AppContainer()

app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Data Extrapolator API is running"}

app.container.wire(modules=[__name__]) 