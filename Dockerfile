FROM python:3.10.13-slim

LABEL maintainer="NicolasFradin <fradin.nicolas@yahoo.com>"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app/medical_bot

COPY ./requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip

# Update and upgrade the existing packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    ninja-build \
    libopenblas-dev \
    build-essential

RUN pip install llama-cpp-python

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./src /app/src

