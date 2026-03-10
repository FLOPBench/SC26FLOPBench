## Docker Setup Instructions


### Container on GPU-Enabled Host
```
# we only really need the Dockerfile from the repo
cd ./{repo-directory}

# this takes about 5-15 minutes
docker build --progress=plain -t 'codex-env' .

# please make sure that the Settings > Resources > Network > 'Enable Host Networking' option is enabled on Docker Desktop
# this is so you can run and view Jupyter Notebooks
docker run -ti --network=host --gpus all --name codex-container --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all codex-env 
```

## Container on Macbook (Apple Silicon M1/2/3/4) -- no NVIDIA GPU
```
git clone git@github.com:gregbolet/gpuFLOPBench-updated.git ./gpuFLOPBench-updated

# we only really need the Dockerfile from the repo
cd ./gpuFLOPBench-updated/gpuFLOPBench-agentic

# this takes about 10 minutes on my Macbook Air M4
docker build --platform=linux/amd64 --progress=plain -t 'codex-env' .

# please make sure that the Settings > Resources > Network > 'Enable Host Networking' option is enabled on Docker Desktop
docker run -ti --network=host --name codex-container --platform=linux/amd64 codex-env 
```


## Later commands
```
docker start codex-container

docker exec -it codex-container /bin/bash
```

## Setup Codex
Once you're in a shell on the container, you'll need to authenticate to be able to use Codex.
Run the following command.
```
codex
```
