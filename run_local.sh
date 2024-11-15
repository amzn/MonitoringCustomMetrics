#!/bin/sh

set -e

red='\e[1;31m'
black='\e[0m'
yellow='\033[0;33m'
white='\033[0;37m'

echo -e "${white}Please remember to set the environment variables in Dockerfile before running this script. ${black}"

docker build . --load

echo -e "${yellow}Setting IMAGE_ID to the latest Docker image present${black}"
IMAGE_ID="$(docker image list -q | grep -m1 "")"
local_output_path=$(pwd)/local_output/${IMAGE_ID}

echo -e "${yellow}Image ID detected is $IMAGE_ID${black}"
echo -e "If you want to bash in the container, please run:"
echo -e "${yellow}docker run -p8080:8080 --entrypoint bash -it ${IMAGE_ID}${black}"

echo -e "${yellow}Executing entry point:${black}"

echo -e "${white}---------------- BEGINNING OF CONTAINER EXECUTION ----------------------${black}"
docker run -p8080:8080 -v ${local_output_path}:/opt/ml/processing/output -it $IMAGE_ID
echo -e "${white}---------------- END OF CONTAINER EXECUTION ----------------------------${black}"

echo -e "Container executed successfully. You can find the generated files in this directory: ${local_output_path}"
