from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI(title="5G Network Slicing Benchmark API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/api/results")
def get_benchmark_results():
    file_path = os.path.join(BASE_DIR, "outputs_phase2/summary_with_ci95.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join(BASE_DIR, "outputs/benchmark_results.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    return {"error": "Benchmark results not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
