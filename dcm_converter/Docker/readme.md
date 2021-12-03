# How to use
sh ./docker_build.sh

# Docker volume mount for multiple docker images
docker run --rm -it -v {host datapath}:{container datapath} {docker container name} # should prepare the input sample dataset(to avoid 'path doesn't exist' error)
