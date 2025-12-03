#!/bin/bash

# สีสำหรับ output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Dog Recognition System Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# ตรวจสอบ Python version
echo -e "${YELLOW}[1/6] ตรวจสอบ Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ ไม่พบ Python3 กรุณาติดตั้ง Python 3.8 หรือสูงกว่า${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ ต้องการ Python 3.8 หรือสูงกว่า แต่พบ Python $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python version: $(python3 --version)${NC}"

# สร้าง virtual environment
echo -e "${YELLOW}[2/6] สร้าง virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ สร้าง virtual environment สำเร็จ${NC}"
else
    echo -e "${GREEN}✅ พบ virtual environment อยู่แล้ว${NC}"
fi

# เปิดใช้งาน virtual environment
echo -e "${YELLOW}[3/6] เปิดใช้งาน virtual environment...${NC}"
source venv/bin/activate

# อัปเดต pip
echo -e "${YELLOW}[4/6] อัปเดต pip...${NC}"
pip install --upgrade pip --quiet

# ติดตั้ง dependencies
echo -e "${YELLOW}[5/6] ติดตั้ง dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✅ ติดตั้ง dependencies สำเร็จ${NC}"
else
    echo -e "${RED}❌ ไม่พบไฟล์ requirements.txt${NC}"
    exit 1
fi

# ตรวจสอบไฟล์ .env
echo -e "${YELLOW}[6/6] ตรวจสอบไฟล์ .env...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  ไม่พบไฟล์ .env${NC}"
    if [ -f "env_example.txt" ]; then
        echo -e "${YELLOW}กำลังสร้างไฟล์ .env จาก env_example.txt...${NC}"
        cp env_example.txt .env
        echo -e "${GREEN}✅ สร้างไฟล์ .env แล้ว${NC}"
        echo -e "${YELLOW}⚠️  กรุณาแก้ไขไฟล์ .env เพื่อตั้งค่า DATABASE_URL${NC}"
    else
        echo -e "${RED}❌ ไม่พบไฟล์ env_example.txt${NC}"
        echo -e "${YELLOW}กรุณาสร้างไฟล์ .env ด้วยตนเอง${NC}"
    fi
else
    echo -e "${GREEN}✅ พบไฟล์ .env อยู่แล้ว${NC}"
fi

# สร้างโฟลเดอร์ embeddings (สำหรับ backward compatibility)
mkdir -p embeddings

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup เสร็จสมบูรณ์!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}ขั้นตอนต่อไป:${NC}"
echo -e "1. แก้ไขไฟล์ .env เพื่อตั้งค่า DATABASE_URL"
echo -e "2. ตรวจสอบว่า PostgreSQL database ถูกสร้างแล้ว"
echo -e "3. รัน './run.sh' เพื่อเริ่มแอปพลิเคชัน"
echo -e "${GREEN}========================================${NC}"

