docker run  \
	-it \
	--ipc=host \
	--privileged \
	--gpus=all \
	-v /home/nfrumkin/gnns:/gnns \
	-v /home/nfrumkin/gnns:/gnns \
	gat:Dockerfile
	#nvcr.io/nvidia/pytorch:20.11-py3
	#--rm \
	#--security-opt seccomp=default_with_perf.json \
	#-v /opt/nvidia/nsight-compute/2020.2.1/:/opt/nvidia/nsight-compute/2020.2.1 \

