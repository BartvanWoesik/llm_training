import json
import os
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel

RESERVATION_FILE = "modules/3-MCP/data/reservations.json"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Reservation(BaseModel):
    """
    Represents a reservation for a film screening.

    Attributes:
        name (str): Name of the person making the reservation.
        film (str): Title of the film being reserved.
        date (str): Date of the reservation.
        time (str): Time of the reservation.
        seats (int): Number of seats reserved.
    """
    name: str 
    film: str
    date: str
    time: str
    seats: int

def read_reservations() -> List[Reservation]:
    if not os.path.exists(RESERVATION_FILE):
        return []
    with open(RESERVATION_FILE, "r") as f:
        data = json.load(f)
    return [Reservation(**item) for item in data]

def write_reservations(reservations: List[Reservation]):
    with open(RESERVATION_FILE, "w") as f:
        json.dump([r.dict() for r in reservations], f, indent=2)

@app.get("/reservations", response_model=List[Reservation])
def get_reservations():
    return read_reservations()

@app.post("/reservations", response_model=Reservation)
def create_reservation(reservation: Reservation):
    reservations = read_reservations()
    reservations.append(reservation)
    write_reservations(reservations)
    return reservation

mcp = FastApiMCP(app)
mcp.mount()


# Run the server
if __name__ == "__main__":
    uvicorn.run("mcp_reservation:app", host="0.0.0.0", port=8001, reload=True)