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

# Create 1000 random files
RUN for i in $(seq 1 1000); do \
        fname=$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 5); \
        echo "incorrect{d3c0y_fl4g_${fname}}" > "${fname}"; \
    done

# Create the hidden flag file
RUN echo 'dcCTF{h1dd3n_f1l3s_4r3_n07_s0_h1dd3n}' > .flag
USER root
# Make all files immutable
RUN chown root:root /home/ctfuser/* /home/ctfuser/.flag && \
    chmod 444 /home/ctfuser/* /home/ctfuser/.flag && \
    chmod 555 /home/ctfuser

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]