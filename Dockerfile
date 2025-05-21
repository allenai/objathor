FROM lscr.io/linuxserver/blender:3.3.1

RUN apt update && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa -y && apt update && apt install python3.10 python3.10-venv python3.10-dev -y && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && pip install --upgrade pip
RUN pip3.10 install --ignore-installed --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+f6701d695de3cb10708fc0686e814276db242ecc
RUN pip3.10 install objathor[annotation]
RUN pip3.10 install pillow

RUN mkdir /output
COPY . /objathor
WORKDIR /objathor

RUN pip install -e ".[annotation]"

ENTRYPOINT ["python3.10", "-m", "objathor.asset_conversion.optimization_pipeline", \
     "--glb_paths=/input/model.glb", \
     "--output_dir=/output", \
     "--extension=.pkl.gz", \
     "--live", \
     "--blender_installation_path", "/blender/blender", \
     "--skip_thor_metadata", \
     "--skip_thor_render"]