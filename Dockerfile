# Gunakan image Python resmi sebagai base image
FROM python:3.9-slim

# Set working directory dalam container
WORKDIR /app

# Copy dependencies dan aplikasi ke dalam container
COPY requirements.txt /app/
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port yang digunakan oleh FastAPI
EXPOSE 8080

# Perintah untuk menjalankan aplikasi FastAPI menggunakan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
