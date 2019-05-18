FROM continuumio/miniconda3

RUN apt-get update \
 && apt-get install --no-install-recommends -y libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.in requirements.in
RUN conda install -c conda-forge --file requirements.in \
 && conda clean -y -a

COPY app /opt/webapp/
WORKDIR /opt/webapp

# Run the image as a non-root user
RUN adduser --disabled-password myuser
USER myuser

CMD gunicorn --bind 0.0.0.0:$PORT wsgi 