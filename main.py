
from fastapi import FastAPI, Request, Form, Depends, Security, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional
import httpx
from fastapi import Query
import pytz
from dotenv import load_dotenv
import os
from datetime import datetime
from sqlalchemy import DateTime
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_302_FOUND
from fastapi.middleware.wsgi import WSGIMiddleware
from web.main import app as flask_app

# Initialize app
tz = pytz.timezone("Asia/Kolkata")
app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "your_session_secret"),
    max_age=1800  # 30 minutes
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Mount the Flask app under /forecast
app.mount("/forecast", WSGIMiddleware(flask_app))

# Security
SECURITY_API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == SECURITY_API_KEY:
        return api_key
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API Key")

# Route guard
async def check_login(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

# User credentials
authorized_users = {
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
    "micbangalore": "mic@bangalore123",
    "mickolkatta": "mic@kolkatta123",
    "micmumbai": "mic@mumbai123",
    "micskuast": "mic@skuast123",
    "datainspect": "inspect@1234"
}

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
    submission_date = Column(String)
    User = Column(String)

Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Mapping for weather API
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
    "Fruit Mandi Kolkata": "Kolkata",
    "Fruit Mandi Mumbai": "Mumbai"
}

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
async def get_weather(city: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            desc = data['weather'][0]['description'].capitalize()
            m = data['main']
            w = data['wind']
            clouds = data['clouds']['all']
            return f"{desc} ({m['temp']}°C, Humidity: {m['humidity']}%, Pressure: {m['pressure']}hPa, Wind: {w['speed']} m/s, Clouds: {clouds}%)"
    return "Weather data unavailable"

# Login/Logout endpoints
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login_beautified.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if authorized_users.get(username) == password:
        request.session["user"] = username
        return RedirectResponse(url="/", status_code=HTTP_302_FOUND)
    return templates.TemplateResponse("login_beautified.html", {"request": request, "message": "Invalid username or password"})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

# Render form with pending-entry support
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    if redirect := await check_login(request):
        return redirect
    current_user = request.session.get("user")
    if current_user == "datainspect":
        # Redirect nodal officer to data viewing page
        return RedirectResponse(url="/my-data", status_code=HTTP_302_FOUND)
    pending = request.session.pop("pending_entry", None)
    context = {
        "request": request,
        "market_options": list(CITY_MAP.keys()),
        "fruit_options": ["Apple","Cherry","Plum","Peach","Strawberry","Grapes","Pear","Walnut"],
        "grade_options": ["A","B","C"],
        "selected_date": pending.get("sale_date", "") if pending else "",
        "selected_date_manual": pending.get("sale_date", "") if pending else "",
        "selected_market": pending.get("market", "") if pending else "",
        "selected_fruit": pending.get("fruit", "") if pending else "",
        "selected_variety": pending.get("variety", "") if pending else "",
        "selected_grade": pending.get("grade", "") if pending else "",
        "min_price": pending.get("min_price", "") if pending else "",
        "max_price": pending.get("max_price", "") if pending else "",
        "modal_price": pending.get("modal_price", "") if pending else "",
        "arrival_qty": pending.get("arrival_qty", "") if pending else "",
        "transaction_volume": pending.get("transaction_volume", "") if pending else "",
        "stock": pending.get("stock", "") if pending else "",
        "demand": pending.get("demand", "") if pending else "",
        "supply": pending.get("supply", "") if pending else "",
        "event": pending.get("event", "") if pending else ""
    }
    return templates.TemplateResponse("form.html", context)

# View all entries
@app.get("/data", response_class=HTMLResponse)
async def fetch_data(request: Request, db: Session = Depends(get_db)):
    if redirect := await check_login(request):
        return redirect
    current_user = request.session.get("user")
    if current_user != "datainspect":
        # Only nodal officer can access this endpoint
        return RedirectResponse(url="/", status_code=HTTP_302_FOUND)
    entries = db.query(MarketData).order_by(MarketData.submission_date.desc()).all()
    return templates.TemplateResponse("data.html", {"request": request, "data": entries})

# Preview endpoint
@app.post("/preview-data", response_class=HTMLResponse)
async def preview_data(
    request: Request,
    market: str = Form(...), fruit: str = Form(...), variety: str = Form(...), grade: str = Form(...),
    min_price: float = Form(...), max_price: float = Form(...), modal_price: float = Form(...),
    arrival_qty: float = Form(...), transaction_volume: float = Form(...), stock: float = Form(...),
    demand: str = Form(...), supply: str = Form(...), event: Optional[str] = Form(""),
    sale_date: Optional[str] = Form(None), sale_date_manual: Optional[str] = Form(None)
):
    current_user = request.session.get("user")
    if current_user == "datainspect":
        # Nodal officer cannot submit data
        return RedirectResponse(url="/my-data", status_code=HTTP_302_FOUND)
    date_str = (sale_date_manual or sale_date or "").strip()
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    form_data = {
        "market": market, "fruit": fruit, "variety": variety,
        "grade": grade, "min_price": min_price, "max_price": max_price,
        "modal_price": modal_price, "arrival_qty": arrival_qty,
        "transaction_volume": transaction_volume, "stock": stock,
        "demand": demand, "supply": supply, "event": event,
        "sale_date": date_str
    }
    request.session["pending_entry"] = form_data
    return templates.TemplateResponse("preview.html", {"request": request, **form_data})

# Confirm endpoint
@app.post("/confirm-data")
async def confirm_data(request: Request, db: Session = Depends(get_db)):
    current_user = request.session.get("user")
    if current_user == "datainspect":
        # Nodal officer cannot submit data
        return RedirectResponse(url="/my-data", status_code=HTTP_302_FOUND)
    form_data = request.session.pop("pending_entry", None)
    if not form_data:
        raise HTTPException(status_code=400, detail="Nothing to confirm")
    # Pop sale_date for potential use (e.g., saving it separately later)
    form_data.pop('sale_date', None)
    # This captures the current timestamp with time and timezone
    submission_dt = datetime.now(tz)  # timezone-aware
    city = CITY_MAP.get(form_data['market'], form_data['market'])
    weather = await get_weather(city)
    entry = MarketData(
        **form_data,
        weather=weather,
        submission_date=submission_dt,
        User=request.session.get("user")
    )
    db.add(entry)
    db.commit()
    request.session["confirmation"] = {
        "weather_info": weather,
        "city_name": city,
        "submission_date": submission_dt.astimezone(tz).strftime('%Y-%m-%d %H:%M:%S')
    }
    return RedirectResponse(url="/submitted", status_code=HTTP_302_FOUND)


@app.get("/my-data", response_class=HTMLResponse)
async def view_my_data(
    request: Request,
    db: Session = Depends(get_db),
    market: Optional[str] = Query(None),
    fruit: Optional[str] = Query(None)
):
    if redirect := await check_login(request):
        return redirect
    current_user = request.session.get("user")
    fruit_options = ["Apple","Cherry","Plum","Peach","Strawberry","Grapes","Pear","Walnut"]
    market_options = list(CITY_MAP.keys())
    if current_user == "datainspect":
        # Nodal officer can see all data, optionally filtered by market and fruit
        query = db.query(MarketData)
        if market:
            query = query.filter(MarketData.market == market)
        if fruit:
            query = query.filter(MarketData.fruit == fruit)
        user_entries = query.order_by(MarketData.submission_date.desc()).all()
    else:
        user_entries = db.query(MarketData).filter(MarketData.User == current_user).order_by(MarketData.submission_date.desc()).all()
    return templates.TemplateResponse("user_data.html", {
        "request": request,
        "data": user_entries,
        "selected_market": market,
        "selected_fruit": fruit,
        "is_nodal": current_user == "datainspect",
        "market_options": market_options,
        "fruit_options": fruit_options
    })

# Submitted confirmation
@app.get("/submitted", response_class=HTMLResponse)
async def submitted(request: Request):
    conf = request.session.pop("confirmation", None)
    ctx = {
        "request": request,
        "message": "Data submitted successfully!" if conf else "No recent submission found.",
        "weather_info": conf.get("weather_info") if conf else "N/A",
        "submission_date": conf.get("submission_date") if conf else "N/A"
    }
    return templates.TemplateResponse("submitted.html", ctx)

# Health check
@app.get("/db-test")
async def db_test(db: Session = Depends(get_db)):
    try:
        rows = db.execute("SELECT 1").fetchall()
        return {"status": "success", "rows": len(rows)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
