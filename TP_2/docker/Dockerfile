FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY modelo_lluvia.keras .
COPY preprocesamiento.joblib .
COPY inferencia.py .

ENTRYPOINT ["python", "inferencia.py"]
CMD ["/data/input.csv", "/data/predicciones.csv"]