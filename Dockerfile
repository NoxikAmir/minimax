# استخدم نسخة بايثون 3.11.4
FROM python:3.11.4-slim

# تعيين مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملف المتطلبات أولاً للاستفادة من التخزين المؤقت لـ Docker
COPY requirements.txt .

# تثبيت الحزم المطلوبة
# psycopg2-binary ضروري للاتصال بـ PostgreSQL
RUN pip install --no-cache-dir -r requirements.txt

# نسخ جميع ملفات المشروع إلى مجلد العمل
COPY . .

# المنفذ الذي سيعمل عليه تطبيقك داخل الحاوية (gunicorn)
EXPOSE 8080

# الأمر الذي سيتم تشغيله عند بدء تشغيل الحاوية
# يقوم بتشغيل الخادم gunicorn وتوجيهه لملف app.py
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "app:app"]
