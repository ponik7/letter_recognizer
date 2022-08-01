FROM python:3.6

RUN python -m pip install flask flask-cors gunicorn numpy torch torchvision opencv-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

ADD image.py image.py
ADD net.py net.py
ADD emnist_recognizer.py emnist_recognizer.py
ADD kek.pth kek.pth

EXPOSE 5050

CMD [ "gunicorn", "--bind", "0.0.0.0:5050", "emnist_recognizer:app" ]