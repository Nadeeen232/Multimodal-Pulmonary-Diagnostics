FROM python:3.12-slim
WORKDIR /ui
COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ui/app.py ui/theme.py ./
COPY .streamlit ./.streamlit
COPY demo_images/team29 ./demo_images/team29
RUN useradd --create-home service
USER service
ENV API_URL=http://api:8080
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
