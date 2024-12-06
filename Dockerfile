# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8080

# Command to run the app using Uvicorn (FastAPI ASGI server)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
