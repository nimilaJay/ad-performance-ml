# Image for the Streamlit dashboard. Built, tested and deployed to Azure Container Apps by
# .github/workflows/ci-cd.yml; the trained models in models/ are baked into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code-only changes reuse this cached layer.
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY .streamlit/ .streamlit/
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY reports/ reports/
COPY data/ data/

# Git commit the image was built from; shown in the dashboard sidebar.
ARG GIT_SHA=local
ENV APP_VERSION=$GIT_SHA

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
