# Inherited from base python image
FROM python:3.8

# Specify the url where the image or container would operate on the server
WORKDIR /auto-grading/roberta

# Move requirements.txt to the workdir
COPY requirements.txt ./

# Install all dependencies
RUN pip install -r requirements.txt

# Add our source code and pre-trained model into the image
COPY . .

# Tell Docker what command we want to run when our image is executed inside a container
CMD ["gunicorn", "server:app", "-c", "./gunicorn.conf.py"]

# For more reference about Dockerfile could view : https://docs.docker.com/language/python/build-images/
