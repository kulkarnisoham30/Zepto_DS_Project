# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask default port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
