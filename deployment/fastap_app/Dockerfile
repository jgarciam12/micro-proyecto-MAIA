FROM python:3.13-slim

WORKDIR /opt

# 2. Copiar archivos manteniendo la estructura que espera tu código
# Estructura final en el contenedor:
# /opt/src/inference/...
# /opt/artifacts/models/...
# /opt/deployment/fastap_app/main.py

COPY artifacts/ /opt/artifacts/
COPY src/ /opt/src/
COPY deployment/fastap_app/ /opt/deployment/fastap_app/

# 3. Instalación de librerías
COPY deployment/fastap_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Configurar el PYTHONPATH para que Python encuentre el módulo 'src'
ENV PYTHONPATH=/opt

# 5. Permisos y Usuario
RUN adduser --disabled-password api-user && chown -R api-user:api-user /opt
USER api-user

EXPOSE 8001

# 6. Ejecutar desde /opt para que las rutas relativas en main.py coincidan
CMD ["uvicorn", "deployment.fastap_app.main:app", "--host", "0.0.0.0", "--port", "8001"]

# docker build -t api:latest -f deployment/fastap_app/Dockerfile .
# docker run -p 8001:8001 -it -e PORT=8001 api
