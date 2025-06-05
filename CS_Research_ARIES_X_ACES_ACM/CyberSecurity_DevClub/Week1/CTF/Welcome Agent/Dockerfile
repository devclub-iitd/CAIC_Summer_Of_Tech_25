FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash ctfuser && \
    echo 'ctfuser:ctfpassword' | chpasswd

RUN mkdir /var/run/sshd
RUN echo 'PermitRootLogin no' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config

USER ctfuser
WORKDIR /home/ctfuser

# Challenge 1: Welcome Agent - Immutable flag file
RUN echo 'dcCTF{w3lc0me_t0_7h3_m4tr1x}' > flag.txt
USER root
RUN chown root:root /home/ctfuser/flag.txt && \
    chmod 444 /home/ctfuser/flag.txt && \
    chmod 555 /home/ctfuser

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]