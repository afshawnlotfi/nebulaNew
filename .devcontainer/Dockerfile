FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

RUN apt-get update -y && \
  apt install -y libgl1-mesa-glx libglu1-mesa libxcursor1 libxinerama1 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
# RUN install -c conda-forge -c cadquery cadquery=master
RUN pip install ezdxf --upgrade

COPY requirements.txt /tmp/pip-tmp/
RUN pip install -r /tmp/pip-tmp/requirements.txt && rm -rf /tmp/pip-tmp
RUN pip install jupyter-cadquery==3.4.0 cadquery-massembly==1.0.0rc0