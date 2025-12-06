FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
# Set non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Change default shell to bash
SHELL ["/bin/bash", "-c"]

# Update packages, install build dependencies, and the specified version of CMake
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget make git gfortran libomp-20-dev libboost-all-dev clang-20 clang-tools-20 unzip && \
    apt-get install -y imagemagick vim git-lfs npm openssh-client && \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh && \
    chmod +x cmake-3.28.0-linux-x86_64.sh && \
    ./cmake-3.28.0-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.28.0-linux-x86_64.sh

# setup alternative names for the compiler
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100 && \
    update-alternatives --install /usr/bin/llvm-cxxfilt llvm-cxxfilt /usr/bin/llvm-cxxfilt-20 100

RUN clang++ --version && clang --version

RUN mkdir -p /codex

# Set the working directory
WORKDIR /codex

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ~/anaconda3 && \
    rm ./Miniconda3-latest-Linux-x86_64.sh

RUN source ~/anaconda3/bin/activate && \
    conda init --all && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN source ~/anaconda3/bin/activate && \
    conda create --name "codex-env" python=3.11.11 && \
    conda activate codex-env

RUN source ~/anaconda3/bin/activate && \
    conda activate codex-env && \
    conda install --channel conda-forge pygraphviz

RUN source ~/anaconda3/bin/activate && \
    conda activate codex-env && \
    pwd && ls -lah && \
    pip install --upgrade pip setuptools wheel && \
    pip install jupyterlab notebook ipywidgets matplotlib pandas scikit-learn scipy seaborn langgraph langchain && \
    pip install langchain_core langchain-azure-ai
    #&& pip install -r /codex/requirements.txt

# write out to the bashrc to source conda and activate the environment on container startup
RUN echo 'conda activate codex-env' >> ~/.bashrc

# install the codexcli
RUN npm i -g @openai/codex

# setup ssh key for git operations
RUN ssh-keygen -t ed25519 -C "codex_docker_container" -N '' -f ~/.ssh/id_ed25519 <<<y >/dev/null 2>&1

# print the public key to the docker build logs for easy access
RUN cat ~/.ssh/id_ed25519.pub

# set an environment variable for convenience
ENV CODEX_ROOT=/codex

# expose the Jupyter notebook port
EXPOSE 8888