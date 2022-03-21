FROM python:3.7

RUN mkdir -p /app
ENV MODEL_PATH expanded_cat_model.joblib

WORKDIR /app

RUN set -ex \
        # Install General Build Dependencies
        && apt-get update -qq -y --no-install-recommends \
        && apt-get upgrade -y --no-install-recommends \
        && cd /app

# Don't need this until requirements.txt read in, so move down for caching
ADD . /app/

RUN sh dependencies.sh \
        # Delete build dependencies
        && rm -rf /tmp/* \
        && apt-get autoremove -y \
        && apt-get purge -y \
        && apt-get clean -y \
        && rm -rf /var/lib/apt/lists/*

CMD ["/app/launch.sh"]
