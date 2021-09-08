# Build an image that can serve mlflow models.
ARG BASE_IMAGE=ubuntu:18.04

FROM ${BASE_IMAGE}

ARG MLFLOW_VER="1.19.0"

ENV MLFLOW_DISABLE_ENV_CREATION="true"
ENV DISABLE_NGINX="true"

RUN apt-get -y update \
        && apt-get install -y --no-install-recommends \
                wget \ 
                curl \
                ca-certificates \
                bzip2 \
                build-essential \ 
                cmake \
                git-core \
        && rm -rf /var/lib/apt/lists/*

# Download and setup miniconda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;
ENV PATH="/miniconda/bin:$PATH"

# Set up the program in the image
WORKDIR /opt/mlflow
RUN pip install "mlflow==${MLFLOW_VER}"
ADD model /opt/ml/model
RUN python -c "from mlflow.models.container import _install_pyfunc_deps; _install_pyfunc_deps('/opt/ml/model', install_mlflow=False)"

CMD /bin/bash -c "source /miniconda/bin/activate custom_env && gunicorn --bind=0.0.0.0:8080 mlflow.models.container.scoring_server.wsgi:app"

EXPOSE 8080
