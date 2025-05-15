from fastapi import FastAPI, Request, Form, Depends, Security, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional
import httpx
import pytz
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import DateTime
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_302_FOUND
from datetime import timedelta
# Initialize app
app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key="4f1d2b6ccecfbc9a3a7b40d1d9e1b03b51b72f04e066eb9d8a5a4b9474a8b9ab",
    max_age=1800  # Session expires after 1 hour (in seconds)
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


load_dotenv()  # Load environment variables
SECURITY_API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Dependency for API Key Authentication
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == SECURITY_API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )

# Users and Database setup
users = {
    "mickupwara": "mic@kupwara123",
    "michandwara": "mic@handwara123",
    "mickanispora": "mic@kanispora123",
    "micsopore": "mic@sopore123",
    "micganderbal": "mic@ganderbal123",
    "micparimpora": "mic@parimpora123",
    "miczaloosa": "mic@zaloosa123",
    "micbotengoo": "mic@botengoo123",
    "micjablipora": "mic@jablipora123",
    "micprichoo": "mic@prichoo123",
    "micpachhar": "mic@pachhar123",
    "micshopian": "mic@shopian123",
    "mickulgam": "mic@kulgam123",
    "micjammu": "mic@jammu123",
    "micbatote": "mic@batote123",
    "micudhampur": "mic@udhampur123",
    "mickathua": "mic@kathua123",
    "micrajouri": "mic@rajouri123",
    "micpoonch": "mic@poonch123",
    "micdelhi": "mic@delhi123",
    "micbanglore": "mic@banglore123",
    "mickolkatta": "mic@kolkatta123",
    "micmumbai": "mic@mumbai123",
    "micskuast": "mic@skuast123"
}

# Database setup
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


# ORM model for market data
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
    submission_date = Column(DateTime, default=datetime.utcnow)
    User = Column(String)
Base.metadata.create_all(bind=engine)

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Mapping of market to city for weather API
CITY_MAP = {
    "F&V Kupwara": "Kupwara",
    "F&V Handwara": "Handwara",
    "F&V Kanispora Baramulla": "Baramulla",
    "F&V Sopore": "Sopore",
    "F&V Zazna Ganderbal": "Gandarbal",
    "F&V Parimpora Srinagar": "Srinagar",
    "F&V Zaloosa Budgam": "Budgam",
    "F&V Botengoo Anantnag": "Anantnag",
    "F&V Jablipora Anantnag": "Anantnag",
    "F&V Prichoo Pulwama": "Pulwama",
    "F&V Pachhar Pulwama": "Pulwama",
    "F&V Aglar Shopian": "Shupiyan",
    "F&V Kulgam": "Kulgam",
    "F&V Narwal Jammu": "Jammu",
    "F & V Batote Jammu": "Jammu",
    "F&V Udhampur": "Udhampur",
    "F&V Kathua": "Kathua",
    "F&V Rajouri": "Rajouri",
    "F&V Kankote Poonch": "Poonch",
    "Fruit Mandi Azadpur": "Delhi",
    "Fruit Mandi Bangalore": "Bangalore",
    "Fruit Mandi Kolkatta": "Kolkata",
    "Fruit Mandi Mumbai": "Mumbai"
}


# Weather API interaction
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "de155dab03208620bc6b5818e5ceb8e8")

async def get_weather(city: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 200:
            data = data = response.json()
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

# Define route protection
async def check_login(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)
    return None

# Login routes
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login_beautified.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in users and users[username] == password:
        request.session["user"] = username
        return RedirectResponse(url="/", status_code=HTTP_302_FOUND)
    else:
        return templates.TemplateResponse("login_beautified.html", {"request": request, "message": "Invalid username or password"})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    login_redirect = await check_login(request)
    if login_redirect:
        return login_redirect
    
    fruit_options = [
        "Apple", "Cherry", "Plum", "Peach", "Strawberry",
        "Grapes", "Pear", "Walnut"
    ]
    
    return templates.TemplateResponse("form.html", {
        "request": request,
        "market_options": list(CITY_MAP.keys()),
        "fruit_options": fruit_options,
        "grade_options": ["A", "B", "C"]
    })

@app.get("/data", response_class=HTMLResponse)
async def fetch_data(request: Request, db: Session = Depends(get_db)):
    login_redirect = await check_login(request)
    if login_redirect:
        return login_redirect
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
    event: Optional[str] = Form("")
):
    # ✅ Get username from session
    User = request.session.get("user")

    try:
        city_name = CITY_MAP.get(market, market)
        weather_info = await get_weather(city_name)

        entry = MarketData(
            market=market,
            fruit=fruit,
            variety=variety,
            grade=grade,
            min_price=min_price,
            max_price=max_price,
            modal_price=modal_price,
            arrival_qty=arrival_qty,
            transaction_volume=transaction_volume,
            stock=stock,
            demand=demand,
            supply=supply,
            event=event,
            weather=weather_info,
            User=User  # ✅ Automatically assigned
        )
        db.add(entry)
        db.commit()

        # Convert UTC to IST
        local_timezone = pytz.timezone('Asia/Kolkata')
        local_time = pytz.utc.localize(entry.submission_date).astimezone(local_timezone)
        formatted_local_time = local_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # ✅ Store in session
        request.session["confirmation"] = {
            "weather_info": weather_info,
            "city_name": city_name,
            "submission_date": formatted_local_time
        }

        # ✅ Redirect to GET route to avoid form resubmission
        return RedirectResponse(url="/submitted", status_code=HTTP_302_FOUND)

    except Exception as e:
        db.rollback()
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": f"An error occurred during submission: {str(e)}"
        })

@app.get("/submitted", response_class=HTMLResponse)
async def submitted(request: Request):
    confirmation = request.session.pop("confirmation", None)
    
    if confirmation:
        return templates.TemplateResponse("submitted.html", {
            "request": request,
            "message": "Data submitted successfully!",
            "weather_info": confirmation.get("weather_info", "N/A"),
            "submission_date": confirmation.get("submission_date", "Not available")
        })
    else:
        # If no session data found, redirect to form or show fallback message
        return templates.TemplateResponse("submitted.html", {
            "request": request,
            "message": "No recent submission found.",
            "weather_info": "N/A",
            "submission_date": "Not available"
        })

@app.get("/db-test")
async def db_test(db: Session = Depends(get_db)):
    try:
        # Try fetching 1 row from the market_data table
        result = db.execute("SELECT * FROM market_data LIMIT 1").fetchall()
        return {"status": "success", "rows_fetched": len(result)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
