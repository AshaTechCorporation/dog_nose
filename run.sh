#!/bin/bash

# สีสำหรับ output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Dog Recognition System${NC}"
echo -e "${GREEN}========================================${NC}"

# ตรวจสอบ virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ ไม่พบ virtual environment${NC}"
    echo -e "${YELLOW}กรุณารัน './setup.sh' ก่อน${NC}"
    exit 1
fi

# เปิดใช้งาน virtual environment
echo -e "${YELLOW}เปิดใช้งาน virtual environment...${NC}"
source venv/bin/activate

# ตรวจสอบไฟล์ .env
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ ไม่พบไฟล์ .env${NC}"
    echo -e "${YELLOW}กรุณาสร้างไฟล์ .env และตั้งค่า DATABASE_URL${NC}"
    exit 1
fi

# ตรวจสอบว่า DATABASE_URL ถูกตั้งค่าแล้ว
if ! grep -q "DATABASE_URL=" .env || grep -q "DATABASE_URL=$" .env || grep -q "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/dog_recognition_db" .env; then
    echo -e "${YELLOW}⚠️  ตรวจสอบว่า DATABASE_URL ในไฟล์ .env ถูกตั้งค่าถูกต้อง${NC}"
fi

# ตรวจสอบ dependencies
echo -e "${YELLOW}ตรวจสอบ dependencies...${NC}"
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${RED}❌ ไม่พบ dependencies${NC}"
    echo -e "${YELLOW}กำลังติดตั้ง dependencies...${NC}"
    pip install -r requirements.txt
fi

# ตั้งค่า host และ port (สามารถแก้ไขได้)
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  เริ่มต้นแอปพลิเคชัน${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Host: ${HOST}${NC}"
echo -e "${YELLOW}Port: ${PORT}${NC}"
echo -e "${YELLOW}API Documentation: http://${HOST}:${PORT}/docs${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}กด Ctrl+C เพื่อหยุดแอปพลิเคชัน${NC}"
echo -e "${GREEN}========================================${NC}"

# รันแอปพลิเคชัน
python3 app.py

# หรือใช้ uvicorn โดยตรง (uncomment ถ้าต้องการ)
# uvicorn app:app --host ${HOST} --port ${PORT} --reload

