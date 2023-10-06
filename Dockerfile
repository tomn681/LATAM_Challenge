# syntax=docker/dockerfile:1.2
FROM python:3.9

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r /opt/app/requirements.txt
COPY . /opt/app

CMD ["python", "challenge/api.py"]
