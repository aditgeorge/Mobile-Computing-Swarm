# 1. Start with a lightweight version of Python to keep the container small
FROM python:3.12-slim

# 2. Stop Python from creating .pyc files and force it to print logs immediately 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Create a non-root user for security (best practice)
RUN adduser --disabled-password --gecos "" orchestrator_user

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Copy just the requirements file first to take advantage of Docker caching
COPY requirements.txt .

# 6. Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your application code into the container
COPY . .

# 8. Switch to the secure, non-root user we created earlier
USER orchestrator_user

# 9. Define the default command that runs when the container starts
CMD ["python", "orchestrator.py"]