FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    openssh-server \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash ctfuser && \
    echo 'ctfuser:ctfpassword' | chpasswd

RUN mkdir /var/run/sshd
RUN echo 'PermitRootLogin no' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config

USER ctfuser
WORKDIR /home/ctfuser

# Challenge 3: Process Hunter
RUN echo 'A process is running with a flag in its command line. Find it!' > readme.txt

USER root
RUN chown root:root /home/ctfuser/readme.txt && \
    chmod 444 /home/ctfuser/readme.txt && \
    chmod 555 /home/ctfuser

EXPOSE 22

# Start SSH and flag process directly
CMD /usr/sbin/sshd && sleep --flag=dcCTF{ps_aux_r3v34l5_4ll} 99999999 & /usr/sbin/sshd -D