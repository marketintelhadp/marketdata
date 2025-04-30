from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional
import httpx
import os

app = FastAPI()

# Setup
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database
DATABASE_URL = "sqlite:///./market_data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ORM Model
class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    market = Column(String)
    fruit = Column(String)
    variety = Column(String)
    grade = Column(String)
    min_price = Column(Float)
    max_price = Column(Float)
    modal_price = Column(Float)
    arrival_qty = Column(Float)
    transaction_volume = Column(Float)
    stock = Column(Float)
    demand = Column(String)
    supply = Column(String)
    event = Column(String, nullable=True)
    weather = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Market to city mapping for weather API
CITY_MAP = {
    "Aglar Shupiyan": "Shupiyan",
    "Pricho Pulwama": "Pulwama",
    "Pachhar Pulwama": "Pulwama",
    "Srinagar Parimpore": "Srinagar",
    "Baramulla Nowpora": "Baramulla",
    "Ganderbal": "Ganderbal",
    "Jammu Narwal": "Narwal",
    "Jammu Kathua": "Kathua",
    "Delhi Azadpur": "Azadpur"
}

# OpenWeatherMap API
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "de155dab03208620bc6b5818e5ceb8e8")

async def get_weather(city: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_desc = data['weather'][0]['description']
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            pressure = data['main']['pressure']
            wind_speed = data['wind']['speed']
            wind_deg = data['wind']['deg']
            cloudiness = data['clouds']['all']
            city_name = data['name']
            return f"{weather_desc.capitalize()} ({temp}°C, Humidity: {humidity}%, Pressure: {pressure}hPa, Wind: {wind_speed} m/s, Clouds: {cloudiness}%)"
        else:
            return "Weather data unavailable"


# Options
MARKET_OPTIONS = list(CITY_MAP.keys())
FRUIT_OPTIONS = ["Apple", "Cherry", "Pear", "Walnut"]
GRADE_OPTIONS = ["A", "B", "C"]

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "market_options": MARKET_OPTIONS,
        "fruit_options": FRUIT_OPTIONS,
        "grade_options": GRADE_OPTIONS
    })

@app.get("/data", response_class=HTMLResponse)
async def fetch_data(request: Request, db: Session = Depends(get_db)):
    data = db.query(MarketData).all()
    return templates.TemplateResponse("data.html", {"request": request, "data": data})

@app.post("/submit-data")
async def submit_form(
    request: Request,
    market: str = Form(...),
    fruit: str = Form(...),
    variety: str = Form(...),
    grade: str = Form(...),
    min_price: float = Form(...),
    max_price: float = Form(...),
    modal_price: float = Form(...),
    arrival_qty: float = Form(...),
    transaction_volume: float = Form(...),
    stock: float = Form(...),
    demand: str = Form(...),
    supply: str = Form(...),
    db: Session = Depends(get_db),
    event: Optional[str] = Form(""),
):
    try:
        city_name = CITY_MAP.get(market, market)  # Get corresponding city from market name
        weather_info = await get_weather(city_name)  # Fetch weather data

        entry = MarketData(
            market=market, fruit=fruit, variety=variety, grade=grade,
            min_price=min_price, max_price=max_price, modal_price=modal_price,
            arrival_qty=arrival_qty, transaction_volume=transaction_volume,
            stock=stock, demand=demand, supply=supply, event=event, weather=weather_info
        )
        db.add(entry)
        db.commit()

        return templates.TemplateResponse("submission_confirmation.html", {
            "request": request,
            "message": "Data submitted successfully!",
            "weather_info": weather_info
        })
    except Exception as e:
        db.rollback()
        return templates.TemplateResponse("error.html", {"request": request, "message": "An error occurred during submission"})


@app.get("/submitted", response_class=HTMLResponse)
async def submitted(request: Request):
    return templates.TemplateResponse("submitted.html", {
        "request": request,
        "message": "Data submitted successfully!",
    })
