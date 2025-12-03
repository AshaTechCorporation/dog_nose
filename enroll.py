import face_recognition
import numpy as np
import pickle
import os

# กำหนดเกณฑ์ความคลาดเคลื่อน (Tolerance) ในการเปรียบเทียบใบหน้า
TOLERANCE = 0.6 

def get_face_embedding(image_path):
    """
    โหลดรูปภาพและดึง Face Embedding (Feature Vector) ออกมา
    """
    try:
        # 1. โหลดรูปภาพ
        image = face_recognition.load_image_file(image_path)
        
        # 2. ตรวจจับตำแหน่งใบหน้า
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            print(f"**ไม่พบใบหน้าในรูป: {image_path}**")
            return None
        
        # 3. สร้าง Embedding จากใบหน้าแรกที่พบ
        embeddings = face_recognition.face_encodings(image, face_locations)
        return embeddings[0]
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประมวลผลรูป {image_path}: {e}")
        return None

def enroll_person(person_name, image_paths):
    """
    บันทึก Face Embeddings จากรูปภาพหลายรูป และเก็บเป็นไฟล์ (เหมือนลายนิ้วมือ)
    """
    print(f"## กำลังบันทึกข้อมูลสำหรับ: {person_name}")
    
    all_embeddings = []
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"⚠️  ไม่พบไฟล์: {path}")
            continue
            
        embedding = get_face_embedding(path)
        if embedding is not None:
            all_embeddings.append(embedding)
            print(f"✅ ดึง Embedding รูปที่ {i+1} ({path}) ได้สำเร็จ")
        else:
            print(f"❌ ไม่สามารถดึง Embedding จาก {path}")
    
    if not all_embeddings:
        print("❌ ไม่สามารถดึง Embedding ได้เลย กรุณาตรวจสอบไฟล์รูปภาพ")
        return False
    
    # หาค่าเฉลี่ยของ Embeddings เพื่อใช้เป็น Feature Vector หลัก
    average_embedding = np.mean(all_embeddings, axis=0)
    
    # สร้างโฟลเดอร์สำหรับเก็บ embeddings (ถ้ายังไม่มี)
    os.makedirs("embeddings", exist_ok=True)
    
    # บันทึก embeddings เป็นไฟล์ (เหมือนลายนิ้วมือ)
    embedding_file = f"embeddings/{person_name}.pkl"
    data = {
        'name': person_name,
        'average_embedding': average_embedding,
        'all_embeddings': all_embeddings,  # เก็บทั้งหมดไว้เผื่อต้องการใช้
        'tolerance': TOLERANCE,
        'num_images': len(all_embeddings)
    }
    
    with open(embedding_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✅ บันทึกข้อมูลเสร็จสิ้น: {embedding_file}")
    print(f"   - จำนวนรูปที่ใช้: {len(all_embeddings)}")
    print(f"   - ขนาด Embedding: {average_embedding.shape}")
    print(f"   - เกณฑ์ Tolerance: {TOLERANCE}")
    
    return True

# --- ตัวอย่างการใช้งาน ---
if __name__ == "__main__":
    # กำหนดชื่อบุคคลและรูปภาพที่ต้องการบันทึก
    person_name = "face"
    known_images = [
        "face_1.jpg",
        "face_2.jpg",
        "face_3.jpg"
    ]
    
    # บันทึกข้อมูล
    success = enroll_person(person_name, known_images)
    
    if success:
        print("\n🎉 บันทึกข้อมูลเสร็จสิ้น! ตอนนี้สามารถใช้ app.py เพื่อตรวจสอบได้แล้ว")
    else:
        print("\n❌ ไม่สามารถบันทึกข้อมูลได้")





