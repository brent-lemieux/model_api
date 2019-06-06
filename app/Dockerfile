FROM python:3-onbuild
MAINTAINER Brent Lemieux <brent@dattablox.com>

# copy our application code
WORKDIR /var/app
COPY . .
COPY requirements.txt .

# fetch app specific deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# expose port
EXPOSE 5000

# start the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
