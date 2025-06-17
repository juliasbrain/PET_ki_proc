FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV FSLDIR=/usr/share/fsl/6.0
ENV PATH="${FSLDIR}/bin:${PATH}"
ENV FSLOUTPUTTYPE=NIFTI_GZ

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bc \
    dc \
    bzip2 \
    tcl \
    tk \
    libjpeg62-turbo \
    libtiff5 \
    libgtk2.0-0 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libxrender1 \
    libsm6 \
    libxext6 \
    libice6 \
    libquadmath0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.5-centos7_64.tar.gz -O /tmp/fsl.tar.gz \
    && mkdir -p /usr/share/fsl/6.0 \
    && tar -xzf /tmp/fsl.tar.gz -C /usr/share/fsl/6.0 --strip-components=1 \
    && rm /tmp/fsl.tar.gz

RUN chmod +x ${FSLDIR}/bin/*
RUN ln -s /usr/bin/python3 /usr/bin/fslpython

COPY pet_ki_proc /app/pet_ki_proc
COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install fslpy

WORKDIR /app/pet_ki_proc

ENTRYPOINT ["python"]
