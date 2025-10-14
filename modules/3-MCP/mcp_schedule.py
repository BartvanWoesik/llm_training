import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP

# Path to the movie schedule CSV
PATH_MOVIE_SCHEDULE = "modules/3-MCP/data/cinema_schedule_4days.csv"

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and process movie schedule
def get_movie_schedule(path_movie_schedule: str = PATH_MOVIE_SCHEDULE) -> pd.DataFrame:
    try:
        print("Loading movie schedule...")
        df = pd.read_csv(path_movie_schedule)
        df['date'] = pd.to_datetime(df['Date']).dt.date
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        return df
    except Exception as e:
        print(f"Error loading movie schedule: {e}")
        raise

# Endpoint: Get schedule by film name
@app.post("/get_movie_schedule_by_name")
def get_movie_schedule_by_name(film_name: str):
    try:
        df = get_movie_schedule()
        now = pd.Timestamp.now()
        filtered = df[(df['Film'].str.lower() == film_name.lower()) & (df['datetime'] > now)]
        return filtered.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}

# Endpoint: Get movies by date
@app.post("/get_movie_by_date")
def get_movies_on_date(date: str):
    try:
        df = get_movie_schedule()
        date_obj = pd.to_datetime(date).date()
        filtered = df[df['date'] == date_obj]
        return filtered[['Film', 'Time', 'Room']].to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}

# Endpoint: Get all current films
@app.get("/get_current_films")
def get_all_films():
    try:
        print("Getting all current films...")
        df = get_movie_schedule()
        films = [str(film) for film in df['Film'].unique().tolist()]
        return films
    except Exception as e:
        return {"error": str(e)}

# Register MCP tools
mcp = FastApiMCP(app)
mcp.mount()

# Run the server
if __name__ == "__main__":
    uvicorn.run("mcp_schedule:app", host="0.0.0.0", port=8000, reload=True)