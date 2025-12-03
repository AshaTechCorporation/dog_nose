"""
Database configuration and models for Dog Recognition System
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np
import pickle

# โหลด environment variables
load_dotenv()

# Database URL จาก environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/dog_recognition_db")

# ตรวจสอบและ validate DATABASE_URL
if not DATABASE_URL or DATABASE_URL.strip() == "":
    raise ValueError("DATABASE_URL ไม่ได้ถูกตั้งค่าในไฟล์ .env")

# ตรวจสอบรูปแบบ DATABASE_URL
if not DATABASE_URL.startswith("postgresql://"):
    raise ValueError(
        f"DATABASE_URL ต้องเริ่มต้นด้วย 'postgresql://'\n"
        f"รูปแบบที่ถูกต้อง: postgresql://username:password@host:port/database\n"
        f"ค่าปัจจุบัน: {DATABASE_URL[:50]}..."
    )

# สร้าง engine พร้อม error handling
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
except Exception as e:
    raise ValueError(
        f"ไม่สามารถสร้าง database connection ได้\n"
        f"กรุณาตรวจสอบ DATABASE_URL ในไฟล์ .env\n"
        f"รูปแบบที่ถูกต้อง: postgresql://username:password@host:port/database\n"
        f"Error: {str(e)}"
    )

# สร้าง SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class สำหรับ models
Base = declarative_base()


class Dog(Base):
    """
    Model สำหรับเก็บข้อมูลหมาที่ลงทะเบียนไว้
    """
    __tablename__ = "dogs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False, comment="ชื่อของหมา")
    average_embedding = Column(LargeBinary, nullable=False, comment="Average embedding (pickled numpy array)")
    tolerance = Column(Float, default=0.6, nullable=False, comment="เกณฑ์ความคลาดเคลื่อน")
    num_images = Column(Integer, default=0, nullable=False, comment="จำนวนรูปภาพที่ใช้ในการลงทะเบียน")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, comment="วันที่สร้าง")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False, comment="วันที่อัปเดตล่าสุด")


def init_db():
    """
    สร้างตารางใน database (ถ้ายังไม่มี)
    """
    try:
        # ทดสอบการเชื่อมต่อ database ก่อน
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # สร้างตาราง
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created/verified successfully")
    except Exception as e:
        error_msg = str(e)
        if "could not translate host name" in error_msg or "nodename nor servname" in error_msg:
            raise ConnectionError(
                f"❌ ไม่สามารถเชื่อมต่อกับ database server ได้\n"
                f"กรุณาตรวจสอบ:\n"
                f"1. DATABASE_URL ในไฟล์ .env ถูกต้องหรือไม่\n"
                f"   ตัวอย่าง: postgresql://username:password@host:port/database\n"
                f"2. Database server กำลังรันอยู่หรือไม่\n"
                f"3. Host, port, username, password ถูกต้องหรือไม่\n"
                f"4. Network/Firewall อนุญาตให้เชื่อมต่อหรือไม่\n"
                f"\nError details: {error_msg}"
            )
        elif "password authentication failed" in error_msg or "authentication failed" in error_msg:
            raise ConnectionError(
                f"❌ การยืนยันตัวตนล้มเหลว\n"
                f"กรุณาตรวจสอบ username และ password ใน DATABASE_URL\n"
                f"\nError details: {error_msg}"
            )
        elif "database" in error_msg.lower() and "does not exist" in error_msg.lower():
            raise ConnectionError(
                f"❌ Database ไม่พบ\n"
                f"กรุณาสร้าง database ก่อน:\n"
                f"  CREATE DATABASE dog_recognition_db;\n"
                f"\nError details: {error_msg}"
            )
        else:
            raise ConnectionError(
                f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ database\n"
                f"กรุณาตรวจสอบ DATABASE_URL และการตั้งค่า database\n"
                f"\nError details: {error_msg}"
            )


def get_db():
    """
    Dependency สำหรับ FastAPI เพื่อให้ได้ database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """
    แปลง numpy array เป็น bytes สำหรับเก็บใน database
    """
    return pickle.dumps(embedding)


def deserialize_embedding(embedding_bytes: bytes) -> np.ndarray:
    """
    แปลง bytes จาก database กลับเป็น numpy array
    """
    return pickle.loads(embedding_bytes)

