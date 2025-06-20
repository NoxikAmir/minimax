# create_tables.py
from database import engine, Base
from models import User, GeneratedTTSAudio # استيراد النماذج الجديدة

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("✅ Tables created successfully.")