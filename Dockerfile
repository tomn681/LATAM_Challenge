# syntax=docker/dockerfile:1.2
FROM python:3.9

EXPOSE 8080

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r /opt/app/requirements.txt
COPY . /opt/app

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
