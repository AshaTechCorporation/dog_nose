from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from starlette.formparsers import MultiPartParser, FormParser
from starlette.datastructures import UploadFile as StarletteUploadFile
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import cv2
import numpy as np
import pickle
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from database import (
    init_db, get_db, Dog, 
    serialize_embedding, deserialize_embedding
)

# โหลด environment variables
load_dotenv()

# กำหนดเกณฑ์ความคลาดเคลื่อน (Tolerance) ในการเปรียบเทียบจมูกหมา
TOLERANCE = float(os.getenv("TOLERANCE", "0.6"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler สำหรับ startup และ shutdown
    """
    # Startup
    try:
        print("🚀 กำลังเริ่มต้นแอปพลิเคชัน...")
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        print("⚠️  กรุณาตรวจสอบ DATABASE_URL ในไฟล์ .env")
        print("   ตัวอย่าง: postgresql://username:password@host:port/database")
        raise
    
    yield
    
    # Shutdown (ถ้ามี)
    print("🛑 กำลังปิดแอปพลิเคชัน...")


app = FastAPI(
    title="Dog Nose Recognition API",
    description="API สำหรับลงทะเบียนและตรวจสอบจมูกหมา",
    version="2.0.0",
    lifespan=lifespan
)

# Handler สำหรับจัดการ validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    จัดการ validation errors
    """
    return await request_validation_exception_handler(request, exc)

def detect_dog_nose(image_array: np.ndarray):
    """
    ตรวจจับจมูกหมาในรูปภาพโดยใช้ color detection และ contour analysis
    """
    # แปลงเป็น OpenCV format (BGR)
    if len(image_array.shape) == 3:
        cv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        cv_image = image_array
    
    # 1. แปลงเป็น HSV สำหรับ color detection ที่ดีกว่า
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    
    # 2. ตรวจจับจมูกหมา (สีดำหรือชมพู/น้ำตาล)
    # สีดำสำหรับจมูก (HSV: 0-180, 0-255, 0-30)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # สีชมพู/น้ำตาลสำหรับจมูก (HSV range สำหรับสีชมพู)
    lower_pink = np.array([0, 50, 50])
    upper_pink = np.array([20, 255, 255])
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # รวม masks
    mask = cv2.bitwise_or(mask_black, mask_pink)
    
    # 3. Morphological operations เพื่อลบ noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 4. หา contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 5. หา contour ที่เหมาะสม (ขนาดและรูปร่างใกล้เคียงจมูกหมา)
    # จมูกหมามักจะมีขนาดประมาณ 1-5% ของรูปภาพ
    h, w = cv_image.shape[:2]
    min_area = (h * w) * 0.001  # 0.1% ของรูป
    max_area = (h * w) * 0.05   # 5% ของรูป
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # ตรวจสอบ aspect ratio (จมูกมักจะค่อนข้างกลมหรือรีเล็กน้อย)
            x, y, w_cont, h_cont = cv2.boundingRect(contour)
            aspect_ratio = float(w_cont) / h_cont if h_cont > 0 else 0
            if 0.5 <= aspect_ratio <= 2.0:  # อัตราส่วนกว้าง:สูง ระหว่าง 0.5-2.0
                valid_contours.append((contour, area))
    
    if not valid_contours:
        return None
    
    # เลือก contour ที่มีขนาดใหญ่ที่สุด (น่าจะเป็นจมูกหลัก)
    largest_contour = max(valid_contours, key=lambda x: x[1])[0]
    x, y, w_nose, h_nose = cv2.boundingRect(largest_contour)
    
    # เพิ่ม padding เล็กน้อย
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w_nose = min(w - x, w_nose + 2 * padding)
    h_nose = min(h - y, h_nose + 2 * padding)
    
    return (x, y, x + w_nose, y + h_nose)  # return (left, top, right, bottom)

def augment_image(image_array: np.ndarray, num_augmentations: int = 3):
    """
    Data Augmentation - ตัดแต่ง/ขยาย dataset จากรูปเดิม
    
    สร้างรูปเพิ่มจากรูปเดิมโดยการ:
    1. Horizontal flip (พลิกซ้าย-ขวา)
    2. Rotation (หมุนเล็กน้อย)
    3. Brightness adjustment (ปรับความสว่าง)
    4. Contrast adjustment (ปรับความคมชัด)
    
    Args:
        image_array: รูปภาพต้นฉบับ (numpy array RGB)
        num_augmentations: จำนวนรูปที่ต้องการสร้างเพิ่ม (default: 3)
    
    Returns:
        augmented_images: รายการรูปที่ augment แล้ว (รวมรูปต้นฉบับด้วย)
    """
    augmented_images = [image_array.copy()]  # เก็บรูปต้นฉบับไว้ด้วย
    
    h, w = image_array.shape[:2]
    center = (w // 2, h // 2)
    
    # แปลง RGB เป็น BGR สำหรับ OpenCV (ถ้าจำเป็น)
    # แต่เราจะใช้ RGB ตรงๆ ก็ได้ เพราะ cv2.flip และ warpAffine ทำงานกับ RGB ได้
    
    # สร้างรูป augment ตามจำนวนที่ต้องการ
    for i in range(num_augmentations):
        aug_image = image_array.copy().astype(np.float32)
        
        # 1. Horizontal flip (พลิกซ้าย-ขวา) - สลับกัน
        if i % 2 == 0:
            aug_image = np.fliplr(aug_image)  # flip left-right (ใช้ numpy แทน cv2)
        
        # 2. Rotation (หมุนเล็กน้อย ±15 องศา)
        angle = np.random.uniform(-15, 15)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # แปลงเป็น uint8 ก่อน warpAffine
        aug_image_uint8 = aug_image.astype(np.uint8)
        aug_image = cv2.warpAffine(aug_image_uint8, rotation_matrix, (w, h), 
                                   borderMode=cv2.BORDER_REPLICATE)
        aug_image = aug_image.astype(np.float32)
        
        # 3. Brightness adjustment (ปรับความสว่าง ±20%)
        brightness_factor = np.random.uniform(0.8, 1.2)
        aug_image = aug_image * brightness_factor
        aug_image = np.clip(aug_image, 0, 255).astype(np.uint8)
        
        # 4. Contrast adjustment (ปรับความคมชัด ±20%)
        contrast_factor = np.random.uniform(0.8, 1.2)
        aug_image = aug_image.astype(np.float32)
        aug_image = (aug_image - 128) * contrast_factor + 128
        aug_image = np.clip(aug_image, 0, 255).astype(np.uint8)
        
        augmented_images.append(aug_image)
    
    return augmented_images

def extract_image_features(image_array: np.ndarray):
    """
    Extract features จากรูปภาพทั้งรูป - หลักการเหมือนลายนิ้วมือ
    
    ขั้นตอน:
    1. Resize รูปเป็นขนาดมาตรฐาน (128x128) เพื่อให้เทียบได้ทุกรูป
    2. แปลงเป็น grayscale หรือใช้สีทั้งหมด
    3. Extract features โดยใช้ histogram, texture, และ pixel values
    4. Normalize features เพื่อให้เทียบได้
    
    Args:
        image_array: รูปภาพในรูปแบบ numpy array (RGB)
    
    Returns:
        features: feature vector (embedding) ที่สามารถใช้เปรียบเทียบได้
    """
    try:
        # 1. Resize เป็นขนาดมาตรฐาน (128x128) เพื่อให้ทุกรูปมีขนาดเท่ากัน
        # จะช่วยให้สามารถเปรียบเทียบได้แม้รูปต้นฉบับจะมีขนาดต่างกัน
        resized = cv2.resize(image_array, (128, 128))
        
        # 2. แปลงเป็น grayscale เพื่อลดขนาดข้อมูล
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = resized
        
        # 3. Extract features หลายแบบ:
        
        # 3.1 Histogram features - บอกการกระจายของความสว่างในรูป
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist_features = hist.flatten() / (hist.sum() + 1e-6)  # normalize
        
        # 3.2 Texture features - บอกลวดลายและรายละเอียดในรูป
        # ใช้ gradient (ความชัน) เพื่อหา edge และ texture
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        # ใช้ mean, std ของ gradient เป็น features
        gradient_mean = np.mean(gradient_magnitude)
        gradient_std = np.std(gradient_magnitude)
        
        # 3.3 Image pixel features - ใช้ค่าพิกเซลโดยตรง (resize เล็กก่อน)
        # Resize เป็น 32x32 ก่อนเพื่อลดจำนวน features
        small_image = cv2.resize(resized, (32, 32))
        if len(small_image.shape) == 3:
            # ถ้ามีสี ใช้แค่ channel แรก (R) หรือแปลงเป็น grayscale
            image_features = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY).flatten() / 255.0
        else:
            image_features = small_image.flatten() / 255.0
        
        # จำกัดจำนวน features ให้พอดี (ใช้ 992 จาก 1024 เพื่อให้รวมแล้วเป็น 1024)
        image_features = image_features[:992]
        
        # 4. รวม features ทั้งหมด
        features = np.concatenate([
            hist_features,           # 32 features - histogram
            [gradient_mean, gradient_std],  # 2 features - texture stats
            image_features          # 992 features - pixel values
        ])  # รวม = 32 + 2 + 992 = 1026 features (แต่อาจปรับได้)
        
        # 5. Normalize features เพื่อให้เทียบได้
        # (เหมือน normalize ให้มีระยะทางมาตรฐาน)
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    except Exception as e:
        return None

def get_face_embedding_from_bytes(image_bytes: bytes):
    """
    โหลดรูปภาพจาก bytes และแปลงเป็น "ลายนิ้วมือ" (embedding/features)
    
    หลักการ: เอารูปทั้งรูปมาทำเป็นข้อมูล (embedding) เก็บไว้
    พอตอนใช้เส้นหา เราค่อยเอามาเทียบว่ามันใกล้เคียงอันไหน
    
    Args:
        image_bytes: รูปภาพในรูปแบบ bytes
    
    Returns:
        embedding: feature vector (ลายนิ้วมือ) หรือ None ถ้าเกิด error
        error: ข้อความ error ถ้ามี
    """
    try:
        # 1. โหลดรูปภาพจาก bytes
        image = Image.open(BytesIO(image_bytes))
        
        # 2. แปลงเป็น RGB ถ้าเป็น RGBA หรือรูปแบบอื่น
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # ตรวจสอบว่ามีข้อมูลในรูป
        if image_array.size == 0:
            return None, "รูปภาพว่างเปล่า"
        
        # 3. Extract features จากรูปทั้งรูป (ไม่ต้องหาจมูก)
        # เอารูปทั้งรูปมาทำเป็น "ลายนิ้วมือ"
        embedding = extract_image_features(image_array)
        
        if embedding is None:
            return None, "ไม่สามารถสร้าง embedding จากรูปภาพได้"
        
        return embedding, None
    
    except Exception as e:
        return None, f"เกิดข้อผิดพลาดในการประมวลผลรูป: {str(e)}"

def load_person_embedding(db: Session, person_name: str):
    """
    โหลด Face Embedding ที่บันทึกไว้ (เหมือนลายนิ้วมือ) จาก database
    """
    dog = db.query(Dog).filter(Dog.name == person_name).first()
    if not dog:
        return None
    
    try:
        average_embedding = deserialize_embedding(dog.average_embedding)
        return {
            'name': dog.name,
            'average_embedding': average_embedding,
            'tolerance': dog.tolerance,
            'num_images': dog.num_images
        }
    except Exception as e:
        return None

def enroll_person(
    db: Session,
    person_name: str,
    image_bytes_list: List[bytes],
    animal_type: Optional[str] = None,
    breed: Optional[str] = None,
    eye_color: Optional[str] = None,
    fur_color: Optional[str] = None,
    special_characteristics: Optional[str] = None,
    birth_date: Optional[str] = None,
    gender: Optional[str] = None,
    weight: Optional[int] = None,
    description: Optional[str] = None,
):
    """
    ลงทะเบียนสัตว์เลี้ยงตัวใหม่ - หลักการเหมือนบันทึกลายนิ้วมือ
    
    ขั้นตอน:
    1. เอารูปแต่ละรูปมาแปลงเป็น "ข้อมูลลายนิ้วมือ" (embedding/features)
    2. นำข้อมูลทั้งหมดมาหาค่าเฉลี่ย เพื่อให้ได้ "ลายนิ้วมือ" ตัวแทน
    3. เก็บข้อมูลลง database
    
    Args:
        person_name: ชื่อของสัตว์เลี้ยง
        image_bytes_list: รายการรูปภาพ (bytes) ที่ต้องการลงทะเบียน
        animal_type, breed, eye_color, fur_color, special_characteristics,
        birth_date, gender, weight, description: ข้อมูลเพิ่มเติม (optional)
    
    Returns:
        success: ว่าบันทึกสำเร็จหรือไม่
        message: ข้อความบอกผล
        errors: รายการ error ถ้ามี
    """
    all_embeddings = []  # เก็บ "ลายนิ้วมือ" จากทุกรูป
    errors = []
    
    # ขั้นตอนที่ 1: แปลงรูปแต่ละรูปเป็น "ข้อมูลลายนิ้วมือ" (embedding)
    # และใช้ Data Augmentation เพื่อขยาย dataset
    for i, image_bytes in enumerate(image_bytes_list):
        # โหลดรูปภาพ
        try:
            image = Image.open(BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
        except Exception as e:
            errors.append(f"รูปที่ {i+1}: ไม่สามารถโหลดรูปได้ - {str(e)}")
            continue
        
        # Data Augmentation - สร้างรูปเพิ่มจากรูปเดิม (3 รูปเพิ่ม)
        augmented_images = augment_image(image_array, num_augmentations=3)
        
        # Extract features จากรูปต้นฉบับและรูปที่ augment แล้ว
        for aug_idx, aug_image in enumerate(augmented_images):
            embedding = extract_image_features(aug_image)
            if embedding is not None:
                all_embeddings.append(embedding)  # เก็บ "ลายนิ้วมือ" ไว้
            else:
                if aug_idx == 0:  # รูปต้นฉบับ
                    errors.append(f"รูปที่ {i+1}: ไม่สามารถสร้าง embedding ได้")
    
    if not all_embeddings:
        # สร้าง error message ที่บอกรายละเอียดปัญหา
        error_detail = "ไม่สามารถดึง Embedding ได้เลย กรุณาตรวจสอบไฟล์รูปภาพ"
        if errors:
            error_detail += f"\nรายละเอียด: {'; '.join(errors)}"
        return False, error_detail, errors
    
    # ขั้นตอนที่ 2: หาค่าเฉลี่ยของ Embeddings เพื่อใช้เป็น "ลายนิ้วมือ" ตัวแทน
    # (เหมือนนำลายนิ้วมือหลายๆ ครั้งมาหาค่าเฉลี่ย เพื่อได้ลายนิ้วมือที่แม่นยำที่สุด)
    average_embedding = np.mean(all_embeddings, axis=0)
    
    # ขั้นตอนที่ 3: บันทึก "ลายนิ้วมือ" ลง database
    # ตรวจสอบว่ามีหมาตัวนี้อยู่แล้วหรือไม่
    existing_dog = db.query(Dog).filter(Dog.name == person_name).first()
    
    extra_data = {
        "type": animal_type or None,
        "breed": breed or None,
        "eye_color": eye_color or None,
        "fur_color": fur_color or None,
        "special_characteristics": special_characteristics or None,
        "birth_date": birth_date or None,
        "gender": gender or None,
        "weight": weight,
        "description": description or None,
    }

    if existing_dog:
        # อัปเดตข้อมูลที่มีอยู่
        existing_dog.average_embedding = serialize_embedding(average_embedding)
        existing_dog.tolerance = TOLERANCE
        existing_dog.num_images = len(all_embeddings)
        for key, val in extra_data.items():
            setattr(existing_dog, key, val)
        db.commit()
        db.refresh(existing_dog)
        return True, f"อัปเดตข้อมูลสำเร็จ: {person_name}", errors
    else:
        # สร้างข้อมูลใหม่
        new_dog = Dog(
            name=person_name,
            average_embedding=serialize_embedding(average_embedding),
            tolerance=TOLERANCE,
            num_images=len(all_embeddings),
            **extra_data
        )
        db.add(new_dog)
        db.commit()
        db.refresh(new_dog)
        return True, f"บันทึกข้อมูลสำเร็จ: {person_name}", errors

def verify_face(known_average_embedding, unknown_embedding, tolerance):
    """
    เปรียบเทียบ "ลายนิ้วมือ" ใหม่กับ "ลายนิ้วมือ" ที่เก็บไว้
    
    หลักการเหมือนเปรียบเทียบบัตรประชาชน - ถ้าลายนิ้วมือใกล้เคียงกันมาก
    แสดงว่าเป็นตัวเดียวกัน
    
    Args:
        known_average_embedding: "ลายนิ้วมือ" ที่เก็บไว้ในฐานข้อมูล
        unknown_embedding: "ลายนิ้วมือ" ใหม่ที่ต้องการตรวจสอบ
        tolerance: เกณฑ์ความคลาดเคลื่อนที่ยอมรับได้
    
    Returns:
        is_match: ว่าตรงกันหรือไม่
        distance: ระยะทางระหว่าง 2 "ลายนิ้วมือ" (น้อย = ใกล้เคียงมาก)
    """
    if unknown_embedding is None:
        return False, 0.0
    
    # คำนวณระยะทางระหว่าง "ลายนิ้วมือ" 2 อัน
    # ยิ่งใกล้กันมาก = distance น้อย = เป็นตัวเดียวกัน
    distance = np.linalg.norm(known_average_embedding - unknown_embedding)
    
    # ถ้าระยะทางน้อยกว่า tolerance แสดงว่า "ลายนิ้วมือ" ใกล้เคียงกัน
    # = เป็นหมาตัวเดียวกัน
    is_match = distance < tolerance
    # แปลงเป็น Python bool เพื่อให้ serialize เป็น JSON ได้
    return bool(is_match), float(distance)

def find_all_enrolled_persons(db: Session):
    """
    หาสัตว์เลี้ยงทั้งหมดที่ลงทะเบียนไว้จาก database
    """
    dogs = db.query(Dog).all()
    persons = []
    for dog in dogs:
        try:
            average_embedding = deserialize_embedding(dog.average_embedding)
            persons.append({
                'name': dog.name,
                'average_embedding': average_embedding,
                'tolerance': dog.tolerance,
                'num_images': dog.num_images,
                'type': dog.type,
                'breed': dog.breed,
                'eye_color': dog.eye_color,
                'fur_color': dog.fur_color,
                'special_characteristics': dog.special_characteristics,
                'birth_date': dog.birth_date,
                'gender': dog.gender,
                'weight': dog.weight,
                'description': dog.description,
            })
        except Exception as e:
            # ข้ามถ้า deserialize ไม่ได้
            continue
    return persons

@app.get("/")
def root():
    return {
        "message": "Dog Nose Recognition API",
        "version": "2.0.0",
        "database": "PostgreSQL",
        "endpoints": {
            "enroll": "/enroll - POST - ลงทะเบียนด้วยรูปภาพหลายรูป (ไม่จำกัดจำนวน)",
            "recognize": "/recognize - POST - ตรวจสอบหมาจากรูป (เปรียบเทียบรูปทั้งรูป)",
            "persons": "/persons - GET - แสดงรายชื่อหมาทั้งหมดที่ลงทะเบียนไว้"
        }
    }

@app.post("/enroll")
async def enroll(
    request: Request,
    name: str = Form(..., description="ชื่อของสัตว์เลี้ยงที่ต้องการลงทะเบียน"),
    images: Optional[List[UploadFile]] = File(default=None, description="รูปภาพหลายรูป (อย่างน้อย 1 รูป) - ควรมีสัตว์เลี้ยงชัดเจน"),
    animal_type: Optional[str] = Form(None, alias="type", description="ประเภทสัตว์ (เช่น หมา, แมว)", example="แมว"),
    breed: Optional[str] = Form(None, description="สายพันธุ์", example="บ๊อบเทล"),
    eye_color: Optional[str] = Form(None, description="สีตา", example="เหลือง"),
    fur_color: Optional[str] = Form(None, description="สีขน", example="ขาวลายดำ"),
    special_characteristics: Optional[str] = Form(None, description="ลักษณะเด่นพิเศษ", example="มีจุดที่หางสีดำ"),
    birth_date: Optional[str] = Form(None, description="วันเกิด (YYYY-MM-DD)", example="2023-04-28"),
    gender: Optional[str] = Form(None, description="เพศ (male/female)", example="female"),
    weight: Optional[str] = Form(None, description="น้ำหนัก (ตัวเลข)", example="20"),
    description: Optional[str] = Form(None, description="รายละเอียดเพิ่มเติม", example="สัตว์เลี้ยงตัวแรก"),
    db: Session = Depends(get_db)
):
    """
    ลงทะเบียนสัตว์เลี้ยงตัวใหม่ด้วยรูปภาพหลายรูป - ใช้หลักการลายนิ้วมือ
    
    - **name**: ชื่อของสัตว์เลี้ยง (บังคับ)
    - **images**: รูปภาพหลายรูป (บังคับ อย่างน้อย 1 รูป)
    - **type**, **breed**, **eye_color**, **fur_color**, **special_characteristics**,
      **birth_date**, **gender**, **weight**, **description**: ข้อมูลเพิ่มเติม (ไม่บังคับ)
    
    ระบบจะใช้รูปภาพมาสร้าง embedding เพื่อใช้ในการจดจำ
    ยิ่งใช้รูปภาพมาก ยิ่งแม่นยำมากขึ้น
    
    ข้อมูลจะถูกบันทึกลง PostgreSQL database
    """
    try:
        # รวบรวมรูปภาพทั้งหมด
        image_bytes_list = []
        form = None
        form_keys = []
        
        # วิธีที่ 1: ใช้ images parameter (ถ้ามี)
        if images:
            for idx, image in enumerate(images):
                if isinstance(image, (UploadFile, StarletteUploadFile)) or hasattr(image, 'read'):
                    if hasattr(image, 'filename') and image.filename and image.filename.strip():
                        try:
                            await image.seek(0)
                            image_bytes = await image.read()
                            if len(image_bytes) > 0:
                                image_bytes_list.append(image_bytes)
                        except Exception as e:
                            raise HTTPException(
                                status_code=400, 
                                detail=f"ไม่สามารถอ่านรูปภาพที่ {idx+1} ได้: {str(e)}"
                            )
        
        # วิธีที่ 2: Parse form data เอง (สำหรับกรณีที่ Postman/Swagger UI ส่งมาในรูปแบบอื่น)
        if len(image_bytes_list) == 0:
            try:
                form = await request.form()
                form_keys = list(form.keys())
                
                # Debug: แสดงข้อมูล form
                print(f"DEBUG: Form keys: {form_keys}")
                for key in form_keys:
                    value = form[key]
                    print(f"DEBUG: Key '{key}': type={type(value)}, is UploadFile={isinstance(value, UploadFile)}")
                    if isinstance(value, UploadFile):
                        print(f"DEBUG:   - filename: {getattr(value, 'filename', 'N/A')}")
                        print(f"DEBUG:   - content_type: {getattr(value, 'content_type', 'N/A')}")
                
                # วิธีที่ 2.1: ใช้ getlist สำหรับ key "images" (รองรับ Postman ที่ส่ง key เดียวกันหลายตัว)
                if "images" in form:
                    images_list = form.getlist("images")
                    print(f"DEBUG: Found 'images' key, getlist returned {len(images_list)} items")
                    for idx, image_item in enumerate(images_list):
                        print(f"DEBUG: Processing image {idx+1}: type={type(image_item)}, type_name={type(image_item).__name__}")
                        print(f"DEBUG:   - hasattr read: {hasattr(image_item, 'read')}")
                        print(f"DEBUG:   - hasattr filename: {hasattr(image_item, 'filename')}")
                        print(f"DEBUG:   - dir: {[x for x in dir(image_item) if not x.startswith('_')][:10]}")
                        
                        # ลองอ่านไฟล์โดยตรง (ไม่ต้องตรวจสอบอะไรก่อน)
                        try:
                            # Reset file pointer ถ้ามี
                            if hasattr(image_item, 'seek'):
                                try:
                                    await image_item.seek(0)
                                    print(f"DEBUG:   - seek(0) successful (async)")
                                except:
                                    try:
                                        image_item.seek(0)
                                        print(f"DEBUG:   - seek(0) successful (sync)")
                                    except Exception as seek_err:
                                        print(f"DEBUG:   - seek failed: {seek_err}")
                            
                            # อ่านไฟล์ (รองรับทั้ง sync และ async)
                            try:
                                # ลอง async read ก่อน
                                image_bytes = await image_item.read()
                                print(f"DEBUG:   - read() successful (async)")
                            except (TypeError, AttributeError) as e:
                                # ถ้าไม่ใช่ async ให้ใช้ sync
                                try:
                                    image_bytes = image_item.read()
                                    print(f"DEBUG:   - read() successful (sync)")
                                except Exception as read_err:
                                    print(f"DEBUG:   - read() failed: {read_err}")
                                    import traceback
                                    traceback.print_exc()
                                    continue
                            
                            print(f"DEBUG:   - Read {len(image_bytes)} bytes")
                            if len(image_bytes) > 0:
                                image_bytes_list.append(image_bytes)
                                print(f"DEBUG:   - ✅ Added to list (total: {len(image_bytes_list)})")
                            else:
                                print(f"DEBUG:   - ⚠️ File is empty")
                        except Exception as e:
                            print(f"DEBUG:   - ❌ Error reading file: {e}")
                            import traceback
                            traceback.print_exc()
                            # ข้ามไฟล์ที่มีปัญหา
                            continue
                
                # วิธีที่ 2.2: วนลูปผ่านทุก key ใน form เพื่อหา UploadFile ทั้งหมด (สำหรับกรณีอื่น)
                if len(image_bytes_list) == 0:
                    print("DEBUG: Trying to find UploadFiles in all form items...")
                    for key, value in form.items():
                        # ข้าม field "name"
                        if key == "name":
                            continue
                        print(f"DEBUG: Checking key '{key}': type={type(value)}, is UploadFile={isinstance(value, UploadFile)}, is StarletteUploadFile={isinstance(value, StarletteUploadFile)}")
                        # ตรวจสอบทั้ง UploadFile (FastAPI) และ StarletteUploadFile (Starlette)
                        if isinstance(value, (UploadFile, StarletteUploadFile)) or hasattr(value, 'read'):
                            filename = getattr(value, 'filename', None)
                            print(f"DEBUG:   - filename: {filename}")
                            if filename and filename.strip():
                                try:
                                    await value.seek(0)
                                    image_bytes = await value.read()
                                    print(f"DEBUG:   - Read {len(image_bytes)} bytes")
                                    if len(image_bytes) > 0:
                                        image_bytes_list.append(image_bytes)
                                        print(f"DEBUG:   - Added to list (total: {len(image_bytes_list)})")
                                except Exception as e:
                                    print(f"DEBUG:   - Error reading file: {e}")
                                    # ข้ามไฟล์ที่มีปัญหา
                                    continue
            except Exception as e:
                # ถ้า parse ไม่ได้ ให้ข้ามไป
                import traceback
                print(f"ERROR: Error parsing form data: {e}")
                traceback.print_exc()
                pass
        
        # ตรวจสอบว่ามีรูปภาพอย่างน้อย 1 รูป
        if len(image_bytes_list) == 0:
            form_info = f"Form keys found: {form_keys}" if form_keys else "Could not parse form"
            
            raise HTTPException(
                status_code=400, 
                detail=f"กรุณาอัปโหลดรูปภาพอย่างน้อย 1 รูป\n\n"
                       f"Debug info: {form_info}\n\n"
                       f"สำหรับ Postman:\n"
                       f"1. ตรวจสอบว่าเลือก 'File' ใน Type dropdown\n"
                       f"2. คลิก 'Select Files' เพื่อเลือกไฟล์รูปภาพ\n"
                       f"3. ตรวจสอบว่าไฟล์ถูกเลือกแล้ว (ไม่ควรมี alert icon)\n\n"
                       f"สำหรับ Swagger UI:\n"
                       f"1. คลิก 'Add string item' ใน field 'images'\n"
                       f"2. เลือกไฟล์รูปภาพในแต่ละ field\n"
                       f"3. คลิก 'Execute'"
            )
        
        # แปลง weight เป็น int ถ้ามีค่า
        weight_int = None
        if weight and str(weight).strip():
            try:
                weight_int = int(float(str(weight).strip()))
            except (ValueError, TypeError):
                pass

        # ลงทะเบียน
        success, message, errors = enroll_person(
            db, name, image_bytes_list,
            animal_type=animal_type,
            breed=breed,
            eye_color=eye_color,
            fur_color=fur_color,
            special_characteristics=special_characteristics,
            birth_date=birth_date,
            gender=gender,
            weight=weight_int,
            description=description,
        )
        
        if success:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": message,
                    "name": name,
                    "num_images_processed": len(image_bytes_list),
                    "warnings": errors if errors else []
                }
            )
        else:
            # ส่ง error message พร้อมรายละเอียด
            return JSONResponse(
                status_code=400,
                content={
                    "detail": message,
                    "errors": errors if errors else [],
                    "suggestion": "กรุณาตรวจสอบว่า:\n"
                                "- รูปภาพมีหมาชัดเจน\n"
                                "- รูปภาพไม่เบลอหรือเสียหาย\n"
                                "- รูปภาพเป็นไฟล์ภาพที่รองรับ (JPG, PNG, etc.)\n"
                                "- รูปภาพไม่ว่างเปล่า"
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.post("/recognize")
async def recognize(
    image: UploadFile = File(..., description="รูปภาพที่ต้องการตรวจสอบ"),
    db: Session = Depends(get_db)
):
    """
    ตรวจสอบหมาจากรูปภาพที่อัปโหลด (ใช้หลักการลายนิ้วมือ - เปรียบเทียบรูปทั้งรูป)
    
    - **image**: รูปภาพที่ต้องการตรวจสอบ
    
    ระบบจะค้นหาในฐานข้อมูล PostgreSQL ที่ลงทะเบียนไว้ทั้งหมดและคืนค่าข้อมูลของหมาที่ตรงกัน
    """
    try:
        # ขั้นตอนที่ 1: เอารูปใหม่มาแปลงเป็น "ลายนิ้วมือ"
        image_bytes = await image.read()
        unknown_embedding, error = get_face_embedding_from_bytes(image_bytes)
        
        if unknown_embedding is None:
            raise HTTPException(status_code=400, detail=error or "ไม่สามารถประมวลผลรูปภาพได้")
        
        # ขั้นตอนที่ 2: โหลด "ลายนิ้วมือ" ของหมาทั้งหมดที่ลงทะเบียนไว้จากฐานข้อมูล
        enrolled_persons = find_all_enrolled_persons(db)
        
        if not enrolled_persons:
            raise HTTPException(status_code=404, detail="ไม่พบข้อมูลหมาที่ลงทะเบียนไว้")
        
        # ขั้นตอนที่ 3: เทียบ "ลายนิ้วมือ" ใหม่กับ "ลายนิ้วมือ" ทุกตัวในฐานข้อมูล
        # หลักการเหมือนตรวจสอบลายนิ้วมือ - เอาไปเทียบกับฐานข้อมูลทั้งหมด
        matches = []
        for person_data in enrolled_persons:
            known_embedding = person_data['average_embedding']  # "ลายนิ้วมือ" ที่เก็บไว้
            tolerance = person_data['tolerance']
            
            # เปรียบเทียบว่าลายนิ้วมือใกล้เคียงกันไหม
            is_match, distance = verify_face(known_embedding, unknown_embedding, tolerance)
            
            # คำนวณ confidence (ความเหมือน) ในรูปเปอร์เซ็นต์
            # confidence = (1 - distance/tolerance) * 100
            # ถ้า distance = 0 → confidence = 100%
            # ถ้า distance = tolerance → confidence = 0%
            confidence = max(0, (1 - distance / tolerance) * 100) if tolerance > 0 else 0
            
            match_entry = {
                "name": person_data['name'],
                "match": bool(is_match),
                "distance": float(distance),
                "tolerance": float(tolerance),
                "confidence": confidence,
                "num_enrollment_images": person_data['num_images']
            }
            for key in ('type', 'breed', 'eye_color', 'fur_color', 'special_characteristics',
                        'birth_date', 'gender', 'weight', 'description'):
                if key in person_data and person_data[key] is not None:
                    match_entry[key] = person_data[key]
            matches.append(match_entry)
        
        # ขั้นตอนที่ 4: หา "ลายนิ้วมือ" ที่ใกล้เคียงที่สุด (distance น้อยที่สุด)
        # = หาเป็นหมาตัวไหน และตรวจสอบว่า confidence > 90% หรือไม่
        best_match = min(matches, key=lambda x: x['distance'])
        best_confidence = best_match.get('confidence', 0)
        
        # ถ้า confidence มากกว่า 90% → พบหมาที่ตรงกัน
        # ถ้า confidence ไม่มากกว่า 90% → ไม่พบหมาที่ตรงกัน
        if best_confidence > 85:
            person_data = {
                "name": best_match['name'],
                "distance": best_match['distance'],
                "tolerance": best_match['tolerance'],
                "confidence": best_confidence
            }
            # เพิ่มข้อมูลเพิ่มเติมจาก database
            for key in ('type', 'breed', 'eye_color', 'fur_color', 'special_characteristics',
                        'birth_date', 'gender', 'weight', 'description'):
                if key in best_match and best_match[key] is not None:
                    person_data[key] = best_match[key]
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "found": True,
                    "person": person_data,
                    "all_comparisons": matches
                }
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "found": False,
                    "message": f"ไม่พบหมาที่ตรงกันในระบบ (confidence สูงสุด: {best_confidence:.2f}% < 90%)",
                    "closest_match": {
                        "name": best_match['name'],
                        "distance": best_match['distance'],
                        "tolerance": best_match['tolerance'],
                        "confidence": best_confidence
                    },
                    "all_comparisons": matches
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.get("/persons")
async def list_persons(db: Session = Depends(get_db)):
    """
    แสดงรายชื่อหมาทั้งหมดที่ลงทะเบียนไว้ในฐานข้อมูล
    """
    try:
        dogs = db.query(Dog).all()
        
        result = []
        for dog in dogs:
            result.append({
                "id": dog.id,
                "name": dog.name,
                "num_images": dog.num_images,
                "tolerance": float(dog.tolerance),
                "type": dog.type,
                "breed": dog.breed,
                "eye_color": dog.eye_color,
                "fur_color": dog.fur_color,
                "special_characteristics": dog.special_characteristics,
                "birth_date": dog.birth_date,
                "gender": dog.gender,
                "weight": dog.weight,
                "description": dog.description,
                "created_at": dog.created_at.isoformat() if dog.created_at else None,
                "updated_at": dog.updated_at.isoformat() if dog.updated_at else None
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "total": len(result),
                "persons": result
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
