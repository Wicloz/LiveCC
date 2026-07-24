FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get upgrade --no-install-recommends --no-install-suggests --assume-yes && rm -rf /var/*/apt/

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests --assume-yes ffmpeg && rm -rf /var/*/apt/
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh

WORKDIR /app/
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY server/requirements.txt ./
RUN pip install --no-cache-dir -Ur requirements.txt
COPY server/requirements.dev.txt ./

COPY server/ ./server/
COPY lua/ ./lua/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--app-dir", "/app/server/", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
