FROM python:3.9

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata \
    vim \
    nano \
    sudo \
    curl \
    wget \
    git && \
    rm -rf /var/lib/apt/lists/*


RUN curl https://get.modular.com | sh - 
RUN modular install --install-version 24.3.0 mojo
ARG MODULAR_HOME="/root/.modular"
ENV MODULAR_HOME=$MODULAR_HOME
ENV PATH="$PATH:$MODULAR_HOME/pkg/packages.modular.com_mojo/bin"

# -- mojo install end

# install pypy
RUN apt-get install -y pypy3

# install rustc
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default nightly

# install codon
RUN apt install -y libpython3-dev
RUN /bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)" || echo "Failed to finish installing, ignoring error"
ENV PATH="/root/.codon/bin:${PATH}"
ENV CODON_PYTHON=/usr/local/lib/libpython3.so

# install numpy
RUN python3 -m pip install numpy --upgrade --break-system-packages && pypy3 -m pip install numpy --upgrade --user --break-system-packages
# ENV PYTHONPATH /usr/local/lib/python3.9:/usr/local/lib/python3.9/site-packages

# setup dirs
WORKDIR /app
RUN mkdir bench_times && mkdir tmp

# copy everything
COPY . /app

CMD [ "python3", "_bench.py" ]

