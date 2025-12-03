#!/usr/bin/env python3
"""
Script สำหรับตรวจสอบการเชื่อมต่อ database
"""
import os
import sys
from dotenv import load_dotenv

# โหลด environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

print("=" * 60)
print("  Database Connection Checker")
print("=" * 60)

if not DATABASE_URL:
    print("❌ ไม่พบ DATABASE_URL ในไฟล์ .env")
    print("\nกรุณาสร้างไฟล์ .env และตั้งค่า:")
    print("DATABASE_URL=postgresql://username:password@host:port/database")
    sys.exit(1)

print(f"\n📋 DATABASE_URL: {DATABASE_URL[:50]}...")

# ตรวจสอบรูปแบบ
if not DATABASE_URL.startswith("postgresql://"):
    print("❌ DATABASE_URL ต้องเริ่มต้นด้วย 'postgresql://'")
    print("\nรูปแบบที่ถูกต้อง:")
    print("postgresql://username:password@host:port/database")
    sys.exit(1)

# แยกส่วน DATABASE_URL
try:
    # ลบ postgresql://
    url_part = DATABASE_URL.replace("postgresql://", "")
    
    # แยกส่วน
    if "@" in url_part:
        auth_part, host_part = url_part.split("@", 1)
        if ":" in auth_part:
            username, password = auth_part.split(":", 1)
        else:
            username = auth_part
            password = ""
    else:
        print("⚠️  DATABASE_URL ไม่มี @ (อาจไม่มี username/password)")
        username = ""
        password = ""
        host_part = url_part
    
    if "/" in host_part:
        host_port, database = host_part.split("/", 1)
    else:
        host_port = host_part
        database = ""
    
    if ":" in host_port:
        host, port = host_port.split(":", 1)
    else:
        host = host_port
        port = "5432"
    
    print(f"\n📊 แยกส่วน DATABASE_URL:")
    print(f"   Username: {username}")
    print(f"   Password: {'*' * len(password) if password else '(ไม่มี)'}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Database: {database}")
    
except Exception as e:
    print(f"⚠️  ไม่สามารถแยกส่วน DATABASE_URL ได้: {e}")

# ทดสอบการเชื่อมต่อ
print("\n🔌 กำลังทดสอบการเชื่อมต่อ...")
try:
    from sqlalchemy import create_engine, text
    
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]
        print("✅ เชื่อมต่อ database สำเร็จ!")
        print(f"\n📦 PostgreSQL Version: {version}")
        
        # ตรวจสอบว่ามีตาราง dogs หรือไม่
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'dogs'
            );
        """))
        table_exists = result.fetchone()[0]
        
        if table_exists:
            print("✅ พบตาราง 'dogs' ใน database")
            
            # นับจำนวนหมาที่ลงทะเบียน
            result = conn.execute(text("SELECT COUNT(*) FROM dogs;"))
            count = result.fetchone()[0]
            print(f"📊 จำนวนหมาที่ลงทะเบียน: {count}")
        else:
            print("⚠️  ยังไม่มีตาราง 'dogs' (จะถูกสร้างอัตโนมัติเมื่อรันแอปพลิเคชัน)")
        
except ImportError:
    print("❌ ไม่พบ sqlalchemy หรือ psycopg2")
    print("กรุณาติดตั้ง: pip install sqlalchemy psycopg2-binary")
    sys.exit(1)
except Exception as e:
    error_msg = str(e)
    print("❌ ไม่สามารถเชื่อมต่อ database ได้")
    print(f"\nError: {error_msg}")
    
    if "could not translate host name" in error_msg or "nodename nor servname" in error_msg:
        print("\n💡 แนวทางแก้ไข:")
        print("1. ตรวจสอบว่า host name/IP address ถูกต้อง")
        print("2. ตรวจสอบว่า database server กำลังรันอยู่")
        print("3. ตรวจสอบ network/firewall settings")
    elif "password authentication failed" in error_msg:
        print("\n💡 แนวทางแก้ไข:")
        print("1. ตรวจสอบ username และ password")
        print("2. ถ้า password มีอักขระพิเศษ ให้ใช้ URL encoding")
    elif "database" in error_msg.lower() and "does not exist" in error_msg.lower():
        print("\n💡 แนวทางแก้ไข:")
        print("1. สร้าง database ก่อน:")
        print("   CREATE DATABASE dog_recognition_db;")
    else:
        print("\n💡 แนวทางแก้ไข:")
        print("1. ตรวจสอบ DATABASE_URL ในไฟล์ .env")
        print("2. ตรวจสอบว่า PostgreSQL กำลังรันอยู่")
        print("3. ตรวจสอบ username, password, host, port, database name")
    
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ทุกอย่างพร้อมใช้งาน!")
print("=" * 60)

