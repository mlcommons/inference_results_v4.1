FROM debian:latest

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

# Use the host machine's SSH credentials
COPY . /root/.ssh/

RUN sed -i 's/#Port 22/Port 2222/g' /etc/ssh/sshd_config

EXPOSE 2222

CMD ["/usr/sbin/sshd", "-D"]

