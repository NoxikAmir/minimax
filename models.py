# models.py
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base
from flask_login import UserMixin # 🆕 لاستخدامه مع نظام تسجيل الدخول

# 🆕 جدول المستخدمين الجديد
class User(Base, UserMixin):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    # علاقة تربط المستخدم بسجلاته الصوتية
    tts_history = relationship("GeneratedTTSAudio", back_populates="user")

# جدول سجل الأصوات المنشأة (معدّل ليرتبط بالمستخدم الجديد)
class GeneratedTTSAudio(Base):
    __tablename__ = "generated_tts_audio"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # 🔑 ربط بالـ User بدلاً من Channel
    text_input_snippet = Column(Text, nullable=False)
    voice_name_used = Column(String, nullable=False)
    language_used = Column(String, nullable=False)
    engine = Column(String, default="minimax", nullable=False)
    audio_filename = Column(String, nullable=False, unique=True)
    task_id = Column(String, nullable=False, unique=True, index=True)
    duration_seconds = Column(Integer, nullable=True)
    public_download_url = Column(String, nullable=True) # تم التبسيط
    created_at = Column(DateTime, default=datetime.utcnow)
    is_deleted = Column(Boolean, default=False, nullable=False)
    user = relationship("User", back_populates="tts_history")