FROM nvidia/cuda:10.1-devel-ubuntu16.04


RUN apt-get -y update && apt install -y wget vim git

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install in batch (silent) mode, does not edit PATH or .bashrc or .bash_profile
# -p path
# -f force
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH=/root/miniconda3/bin:${PATH} 

#RUN source /root/.bashrc
#RUN source /root/.bash_profile

RUN conda update -y conda

RUN conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

RUN pip3 install ofa pyyaml matplotlib scipy pandas sklearn networkx

# create a non-root user named tester, 
# give them the password "tester" put them in the sudo group
RUN useradd -d /home/nfrumkin -m -s /bin/bash nfrumkin && echo "nfrumkin:nfrumkin" | chpasswd && adduser nfrumkin sudo

# start working in the "tester" home directory
WORKDIR /home/nfrumkin
# COPY ./src

# Make the files owned by tester
RUN chown -R nfrumkin:nfrumkin /home/nfrumkin



# Switch to your new user in the docker image
# USER nfrumkin

#RUN apt install -y software-properties-common
#RUN add-apt-repository ppa:deadsnakes/ppa --yes

#RUN apt install -y python3.8
#RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html





