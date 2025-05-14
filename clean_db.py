from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import MarketData, DATABASE_URL  # Reuse your existing model and DB URL

# Set up DB session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
session = Session()

# Delete all rows
session.query(MarketData).delete()
session.commit()

print("✅ market_data table cleaned successfully.")

session.close()
