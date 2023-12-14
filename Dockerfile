FROM python:3.8

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py"]

