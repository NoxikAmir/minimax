# app.py (النسخة المستقلة)

import os
import re
import json
import uuid
import math
import logging # 💡 استيراد مكتبة التسجيل
import sys     # 💡 استيراد مكتبة النظام
import time
import threading
import shutil
import requests
from functools import wraps
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import (
    Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from sqlalchemy.exc import IntegrityError
import boto3

# --- إعدادات Flask وأنظمة الدخول ---
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# --- استيراد ملفات المشروع ---
from models import Base, User, GeneratedTTSAudio

# --- 💡 إضافة مهمة: إعداد نظام التسجيل (Logging) ---
# هذا يضمن ظهور كل سجلات التطبيق في fly logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# --- نهاية الإضافة ---

# --- إعداد التطبيق ---
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_very_secret_key_for_standalone_app")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Render سيوفر هذا المتغير تلقائياً عند ربط قاعدة البيانات
DATABASE_URL = os.getenv("DATABASE_URL") 
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1) # SQLAlchemy يتطلب postgresql://

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- 💡 إضافة مهمة: إنشاء جداول قاعدة البيانات عند بدء التشغيل ---
print("Creating database tables if they don't exist...")
from models import Base # تأكد من وجود هذا الاستيراد
Base.metadata.create_all(bind=engine)
print("Tables checked/created successfully.")
# --- نهاية الإضافة ---

ENABLE_R2_UPLOAD = os.getenv('ENABLE_R2_UPLOAD', 'False').lower() == 'true'

CLOUDFLARE_ACCOUNT_ID = os.getenv('CLOUDFLARE_ACCOUNT_ID')
R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')
R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')
R2_PUBLIC_DOMAIN = os.getenv('R2_PUBLIC_DOMAIN')


# --- إعدادات Minimax API ---
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "some_default_value_if_any")
GROUP_ID = "1920065149390032941"

# --- إعداد المجلدات ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- إعداد نظام تسجيل الدخول ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # اسم دالة العرض لصفحة تسجيل الدخول

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    user = db.query(User).get(int(user_id))
    db.close()
    return user

# ===================================================================
# --- دوال الواجهة الخلفية (Backend) ---
# ===================================================================

minimax_tts_tasks_status = {}

# في ملف app.py المستقل

def get_minimax_voices():
    """
    هذه الدالة تقرأ الأصوات من ملف JSON ثابت مع معالجة أفضل للأخطاء.
    """
    # ❗️ تأكد من تطابق هذا المسار مع هيكل المجلدات لديك
    json_path = "static/minimax_voices_with_lang.json"
    try:
        # التأكد من أن الملف موجود قبل محاولة فتحه
        if not os.path.exists(json_path):
            # طباعة خطأ واضح في الطرفية (Terminal) لتسهيل التصحيح
            print(f"❌ ERROR: Voice file not found at '{json_path}'. Please check the file path.")
            return [] # إرجاع قائمة فارغة بدلاً من التسبب في انهيار التطبيق

        with open(json_path, "r", encoding="utf-8") as f:
            voices = json.load(f)
            print(f"✅ Successfully loaded {len(voices)} voices from {json_path}")
            return voices
            
    except json.JSONDecodeError:
        print(f"❌ ERROR: The file at '{json_path}' is not a valid JSON. Please check its content.")
        return []
    except Exception as e:
        print(f"❌ An unexpected error occurred in get_minimax_voices: {e}")
        return []

def generate_minimax_tts(api_key, group_id, voice_id, text, model="speech-02-turbo", speed=1.0, pitch=0, vol=1.0):
    url = f"https://api.minimaxi.chat/v1/t2a_v2?GroupId={group_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model, "text": text, "stream": False,
        "voice_setting": {"voice_id": voice_id, "speed": float(speed), "vol": float(vol), "pitch": int(pitch)},
        "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1},
        "output_format": "hex"
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if result.get("base_resp", {}).get("status_code") != 0:
            return None, result.get("base_resp", {}).get("status_msg", "Unknown API error")
        hex_audio = result.get("data", {}).get("audio")
        return bytes.fromhex(hex_audio) if hex_audio else None, None
    except Exception as e:
        app.logger.error(f"Error in generate_minimax_tts: {e}")
        return None, str(e)

# في ملف app.py، استبدل هذه الدالة بالكامل

def process_minimax_tts_chunk(chunk_text, output_file, voice_id, speed, pitch, vol):
    """
    Processes a single chunk of text using Minimax TTS with corrected keyword arguments.
    """
    # لاحظ أننا لم نعد بحاجة إلى حلقة إعادة المحاولة هنا لأنها معقدة ويمكن إضافتها لاحقًا إذا لزم الأمر
    # التركيز الآن على جعل الطلب الأساسي يعمل
    try:
        # --- 💡 التعديل الرئيسي هنا: استخدام أسماء المتغيرات لضمان وصول كل قيمة لمكانها الصحيح ---
        audio_bytes, error = generate_minimax_tts(
            api_key=MINIMAX_API_KEY,
            group_id=GROUP_ID, # 修正: استخدم المتغير الصحيح
            voice_id=voice_id,
            text=chunk_text,
            model="speech-02-turbo", # 💡 نمرر الموديل بشكل صريح لضمان الدقة
            speed=speed,
            pitch=pitch,
            vol=vol
        )
        # --- نهاية التعديل ---

        if error or not audio_bytes:
            # طباعة الخطأ الفعلي لتسهيل التصحيح
            app.logger.error(f"Minimax API call failed. API returned: '{error}'")
            raise RuntimeError(f"MiniMax chunk failed: {error or 'No audio data'}")

        # حفظ البايتات في ملف الإخراج
        with open(output_file, "wb") as f:
            f.write(audio_bytes)

        app.logger.info(f"✅ Chunk processed and saved to {output_file}")
        return output_file # إرجاع مسار الملف عند النجاح

    except Exception as e:
        # في حال حدوث أي خطأ، سيتم تسجيله وإطلاقه ليتم التقاطه في المهمة الخلفية
        app.logger.error(f"❌ Error processing chunk: {e}")
        raise e

def background_minimax_tts_task(app_context, text_to_speak, voice_id, language, task_id, user_id=None, speed=1.0, pitch=0, vol=1.0, output_format='mp3'):
    with app_context:
        app.logger.info(f"[Task {task_id}] Starting Minimax TTS task for user {user_id}.")
        # --- 💡 تعديل: تحديث الحالة فقط، لا تقم بإنشائها من جديد ---
        minimax_tts_tasks_status[task_id]['status'] = 'processing'
        # --- نهاية التعديل ---
        temp_folder = os.path.join(UPLOAD_FOLDER, "minimax_temp", task_id)
        os.makedirs(temp_folder, exist_ok=True)
        try:
            # Note: Assuming split_text is now correctly imported
            chunks = split_text(text_to_speak, language=language, max_words=400, max_chars=1500)
            if not chunks: raise ValueError("Text could not be split into chunks.")
            
            total_chunks = len(chunks)
            temp_files = [None] * total_chunks
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(process_minimax_tts_chunk, chunk, os.path.join(temp_folder, f"chunk_{i}.mp3"), voice_id, speed, pitch, vol): i for i, chunk in enumerate(chunks)}
                for i, future in enumerate(as_completed(futures)):
                    idx = futures[future]
                    try:
                        temp_files[idx] = future.result()
                    except Exception as e:
                        minimax_tts_tasks_status[task_id]['errors'].append(f"Chunk {idx+1} failed: {e}")
                    # minimax_tts_tasks_status[task_id]['progress'] = 10 + int(((i + 1) / total_chunks) * 80)
                    minimax_tts_tasks_status[task_id]['progress'] = 10 + int(((i + 1) / total_chunks) * 85)

            valid_files = [f for f in temp_files if f]
            if not valid_files: raise RuntimeError("All chunks failed to process.")

            final_filename = f"{task_id}_minimax_audio.{output_format}"
            local_final_dir = os.path.join(UPLOAD_FOLDER, "tts_final")
            os.makedirs(local_final_dir, exist_ok=True)
            local_final_audio_path = os.path.join(local_final_dir, final_filename)
            
            # Note: Assuming merge_audio_files is imported
            merge_audio_files(valid_files, local_final_audio_path)
            
            minimax_tts_tasks_status[task_id]['filename'] = final_filename
            public_url = None
            if ENABLE_R2_UPLOAD:
                public_url = upload_to_r2(local_final_audio_path, final_filename)

            minimax_tts_tasks_status[task_id]['public_download_url'] = public_url or url_for('serve_tts_audio', filename=final_filename, _external=True)
            
            # Save to DB
            db = SessionLocal()
            try:
                snippet = text_to_speak[:150] + ("..." if len(text_to_speak) > 150 else "")
                audio_duration = AudioSegment.from_file(local_final_audio_path).duration_seconds
                new_record = GeneratedTTSAudio(
                    user_id=user_id, text_input_snippet=snippet, voice_name_used=voice_id,
                    language_used=language, audio_filename=final_filename, task_id=task_id,
                    duration_seconds=math.ceil(audio_duration), public_download_url=public_url
                )
                db.add(new_record)
                db.commit()
            finally:
                db.close()

            minimax_tts_tasks_status[task_id]['status'] = 'completed'
            minimax_tts_tasks_status[task_id]['progress'] = 100

        except Exception as e:
            app.logger.error(f"[Minimax Task {task_id}] failed: {e}", exc_info=True)
            minimax_tts_tasks_status[task_id].update({'status': 'error', 'errors': [str(e)]})
        finally:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)

# ===================================================================
# --- مسارات التطبيق (Routes) ---
# ===================================================================

@app.route('/')
def index_route():
    if current_user.is_authenticated:
        return redirect(url_for('minimax_studio_page'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('minimax_studio_page'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        db = SessionLocal()
        user = db.query(User).filter_by(username=username).first()
        db.close()
        if not user or not check_password_hash(user.password_hash, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))
        login_user(user, remember=remember)
        return redirect(url_for('minimax_studio_page'))
    return render_template('login.html') # ❗️ ستحتاج لإنشاء هذا الملف

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('minimax_studio_page'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        db = SessionLocal()
        user = db.query(User).filter_by(username=username).first()
        if user:
            flash('Username already exists.')
            db.close()
            return redirect(url_for('register'))
        
        new_user = User(username=username, password_hash=generate_password_hash(password, method='pbkdf2:sha256'))
        db.add(new_user)
        db.commit()
        db.close()
        flash('Thanks for registering! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html') # ❗️ ستحتاج لإنشاء هذا الملف

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/studio')
@login_required
def minimax_studio_page():
    voices = get_minimax_voices()
    return render_template('minimax_studio.html', minimax_voices_json=voices)


@app.route('/synthesize_minimax', methods=['POST'])
@login_required
def synthesize_minimax_speech_route():
    data = request.get_json()
    task_id = str(uuid.uuid4())
    app.logger.info(f"Received new TTS request. Task ID: {task_id}")
    
    # --- 💡 تعديل: إنشاء حالة أولية للمهمة قبل بدء المهمة الخلفية ---
    minimax_tts_tasks_status[task_id] = {
        'status': 'queued',
        'progress': 5, # يمكن أن تبدأ من 5% لتعطي شعورًا بالتقدم الفوري
        'public_download_url': None,
        'filename': None,
        'errors': []
    }
    # --- نهاية التعديل ---

    thread = threading.Thread(
        target=background_minimax_tts_task,
        args=(
            app.app_context(), data.get('text'), data.get('voice_id'),
            data.get('language'), task_id, current_user.id,
            data.get('speed', 1.0), data.get('pitch', 0), data.get('vol', 1.0)
        )
    )
    thread.start()
    return jsonify({"task_id": task_id, "status_url": url_for('get_minimax_tts_task_status', task_id=task_id)}), 202


@app.route('/minimax_tts_task_status/<task_id>')
def get_minimax_tts_task_status(task_id):
    return jsonify(minimax_tts_tasks_status.get(task_id, {"status": "not_found"}))

@app.route('/tts_minimax_history')
@login_required
def get_tts_minimax_history():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    db = SessionLocal()
    query = db.query(GeneratedTTSAudio).filter_by(user_id=current_user.id, engine='minimax', is_deleted=False)
    total = query.count()
    records = query.order_by(GeneratedTTSAudio.created_at.desc()).limit(per_page).offset((page - 1) * per_page).all()
    db.close()
    
    history_data = [{
        "text_snippet": r.text_input_snippet, "voice_name": r.voice_name_used, "language": r.language_used,
        "duration": r.duration_seconds, "filename": r.audio_filename,
        "created_at": r.created_at.strftime("%Y-%m-%d %H:%M"),
        "download_url": r.public_download_url or url_for('serve_tts_audio', filename=r.audio_filename)
    } for r in records]
    
    return jsonify({
        "success": True, "history": history_data, "current_page": page, 
        "total_pages": math.ceil(total / per_page)
    })

@app.route('/generated_audio/<filename>')
def serve_tts_audio(filename):
    """
    يعرض ملف صوتي تم إنشاؤه مع تعقيم اسم الملف كإجراء أمني.
    """
    # ❗️ إضافة مهمة: تعقيم اسم الملف لمنع هجمات اجتياز المسار
    safe_filename = secure_filename(filename)
    
    # تأكد من أن الملف المطلوب موجود فعلاً بعد التعقيم لتجنب الأخطاء
    directory = os.path.join(UPLOAD_FOLDER, "tts_final")
    if not os.path.isfile(os.path.join(directory, safe_filename)):
        # يمكنك إرجاع خطأ 404 إذا لم يتم العثور على الملف
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(directory, safe_filename)

def split_text(text, language="en", max_words=400, max_chars=3500):
    """
    يقسم النص إلى مقاطع. تم تحسينه للغات CJK مع الحفاظ على المنطق القديم للغات الأخرى.
    """
    CJK_LANGUAGES = {"zh", "ja", "ko", "th", "lo", "my", "km", "si"}
    
    if language in CJK_LANGUAGES:
        print(f"CJK-like language '{language}' detected. Splitting text by sentences to respect API byte limits.")
        chunks = []
        current_chunk = ""
        
        # 1. تعريف فواصل الجمل للغات CJK والإنجليزية
        # تشمل النقطة الصينية، علامة الاستفهام، علامة التعجب، والنظائر الإنجليزية
        sentence_delimiters = r'([.!?。！？])'
        sentences = re.split(sentence_delimiters, text)
        
        # 2. تنظيف وإعادة دمج الفواصل مع الجمل
        processed_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i+1] in ".!?。！？":
                processed_sentences.append(sentences[i] + sentences[i+1])
                i += 2
            else:
                if sentences[i].strip():
                    processed_sentences.append(sentences[i])
                i += 1

        # 3. تجميع الجمل في مقاطع (chunks) لا تتجاوز الحد الأقصى للبايتات
        for sentence in processed_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # التحقق من حجم المقطع الحالي + الجملة الجديدة (بالبايتات)
            if len((current_chunk + " " + sentence).encode('utf-8')) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if not current_chunk:
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
        
        # إضافة آخر مقطع متبقي
        if current_chunk:
            chunks.append(current_chunk)

    else:
        # هذا الجزء يبقى كما هو تمامًا لأنه يعمل بشكل ممتاز مع اللغات الأخرى
        print(f"Language '{language}' detected. Splitting text by {max_words} words per chunk.")
        words = text.split()
        chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    print(f"Total chunks created: {len(chunks)}")
    return chunks

# دمج الملفات الصوتية
def merge_audio_files(audio_files, output_file):
    try:
        combined_audio = AudioSegment.empty()
        for audio_file in audio_files:
            audio_segment = AudioSegment.from_file(audio_file)
            combined_audio += audio_segment

        combined_audio.export(output_file, format="mp3")
        print(f"Final audio file saved in {output_file}")
    except Exception as e:
        raise RuntimeError(f"Error merging audio files: {e}")


def upload_to_r2(local_file_path, object_name=None):
    if not all([CLOUDFLARE_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        print("❌ Cloudflare R2 credentials are not set. Skipping upload.")
        return None

    if object_name is None:
        object_name = os.path.basename(local_file_path)

    content_type_map = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
    }
    file_extension = os.path.splitext(local_file_path)[1].lower()
    content_type = content_type_map.get(file_extension, 'application/octet-stream')

    endpoint_url = f'https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com'

    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto'
    )

    try:
        extra_args = {
            'ContentType': content_type,
            'ContentDisposition': f'attachment; filename="{object_name}"'
        }
        
        s3_client.upload_file(
            local_file_path,
            R2_BUCKET_NAME,
            object_name,
            ExtraArgs=extra_args
        )
        
        public_url = f"{R2_PUBLIC_DOMAIN}/{object_name}"
        print(f"✅ Successfully uploaded {object_name} with force download header to R2: {public_url}")
        return public_url
    except Exception as e:
        print(f"❌ Error uploading to R2: {e}")
        return None
    
from sqlalchemy import text  # ✅ أضف هذا السطر

@app.route("/ping-db")
def ping_db():
    from database import SessionLocal
    session = SessionLocal()
    try:
        session.execute(text("SELECT 1"))  # ✅ استخدم text() هنا
        return jsonify({"status": "Database awake ✅"})
    except Exception as e:
        return jsonify({"status": "Database error ❌", "error": str(e)})
    finally:
        session.close()


@app.route('/health')
def health_check():
    """مسار بسيط وسريع لفحص الصحة."""
    return "OK", 200
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)