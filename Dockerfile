# 1️⃣ Use a slightly larger Python image to avoid missing libraries
FROM python:3.10

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements first and install dependencies (enables Docker caching)
COPY requirements.txt /app/

# 4️⃣ Upgrade pip and install only your project modules
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy your project source code
COPY src /app/src

# 6️⃣ Default command to run your script
CMD ["python", "src/fx_flow_model.py"]
