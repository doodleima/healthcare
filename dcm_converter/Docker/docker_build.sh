#!/bin/bash
cd ..
docker build --rm=true -t dcm_converter:cpu -f ./Docker/Dockerfile .
