# forge-calc — build context is the PlasticFEM_v4 repo root
#   docker build -f web/Dockerfile -t forge-calc .
FROM python:3.11-slim

# system libs the gmsh wheel needs (headless; matplotlib uses the Agg backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 \
        libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY web/requirements.txt ./web/requirements.txt
RUN pip install --no-cache-dir -r web/requirements.txt

# solver package + validated cases + web service
COPY plasticfem/ ./plasticfem/
COPY run_case.py ./run_case.py
COPY model_cases/ ./model_cases/
COPY web/ ./web/

ENV MPLBACKEND=Agg PYTHONUNBUFFERED=1
WORKDIR /app/web
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
