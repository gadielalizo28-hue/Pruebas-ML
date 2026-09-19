FROM python:3.10-slim

WORKDIR /app

# 1. Instalar dependencias del sistema y limpiar la caché de apt para reducir espacio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copiar e instalar requerimientos aprovechando la caché
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiar el código del proyecto
COPY . .

# 4. DESCARGA PREVIA DEL MODELO (Crucial para IA)
# Esto ejecuta un script rápido en Python para descargar GPT-2 dentro de la imagen
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('gpt2'); \
    AutoModelForCausalLM.from_pretrained('gpt2')"

EXPOSE 8000

CMD ["uvicorn", "app_inferencia:app", "--host", "0.0.0.0", "--port", "8000"]
