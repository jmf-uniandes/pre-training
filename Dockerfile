FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libssl-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    # copiar archivo de requerimientos solo del API
COPY src/api/requirements_api.txt /app/requirements.txt

# instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# copiar todo el proyecto
COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
