FROM python:3.8-slim-buster

RUN python3 -m venv /opt/venv


WORKDIR /app

# ENV VIRTUAL_ENV=/opt/venv

# sRUN . /opt/venv/bin/activate && pip3 install -r requirements.txt
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

CMD [ "python3", "facemask.py"]

# CMD . /opt/venv/bin/activate && exec python3 facemask.py
