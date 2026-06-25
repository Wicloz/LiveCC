# LiveCC transcoder service.
#
# Debian bookworm ships ffmpeg 5.1, which has the DFPWM encoder we need for CC
# audio AND a normal, dynamically-linked TLS/HLS stack.  (The previous static
# ffmpeg build segfaulted whenever yt-dlp used it to download live HLS — a
# library issue in that self-contained build.)  We don't use the GPU, so there
# is no reason for a CUDA base image.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        unzip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Fail the build early if this ffmpeg can't do DFPWM (CC speaker audio).
RUN ffmpeg -hide_banner -encoders | grep -q dfpwm \
    || (echo "ERROR: ffmpeg is missing the dfpwm encoder" && exit 1)

# yt-dlp needs a JS runtime (deno) to solve YouTube's "n" challenge.
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/
COPY lua/    ./lua/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--app-dir", "/app/server", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
