ARG DEPENDENCY_BASE=python:3.12-slim
FROM ${DEPENDENCY_BASE} AS builder
USER root
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY constraints-cpu.txt ./
COPY src ./src
COPY app ./app
RUN pip install --no-cache-dir --no-build-isolation -c constraints-cpu.txt .

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 MODEL_BUNDLE=/models
COPY --from=builder /opt/venv /opt/venv
RUN useradd --create-home service
WORKDIR /app
USER service
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
