from ubuntu:latest

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install build-essential scala openssh-server vim net-tools openjdk-8-jdk -y
RUN wget -q https://archive.apache.org/dist/hadoop/common/hadoop-3.3.4/hadoop-3.3.4.tar.gz
RUN wget -q https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz
RUN tar -xzf hadoop-3.3.4.tar.gz
RUN tar -xzf spark-3.3.2-bin-hadoop3.tgz
RUN mv hadoop-3.3.4 /hadoop
RUN mv spark-3.3.2-bin-hadoop3 /spark

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre 
ENV PATH=$JAVA_HOME/bin:$PATH

ENV HADOOP_HOME=/hadoop 
ENV PATH=$HADOOP_HOME/bin:$PATH 

ENV SPARK_HOME=/spark
ENV PATH=$SPARK_HOME/bin:$PATH 

RUN ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
RUN cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
RUN sed -i 's/{ISO8601}%p/{ISO8601} %p/' /hadoop/etc/hadoop/log4j.properties
RUN sed -i 's/01} %p/01} %p [%t]/' /hadoop/etc/hadoop/log4j.properties

COPY file.txt /spark/file.txt
COPY cluster.txt /spark/cluster.txt
COPY lr_test.txt /spark/lr_test.txt
COPY page.txt /spark/page.txt
COPY sparklord.jar /spark/sparklord.jar

COPY spark-env.sh /spark/conf/spark-env.sh
COPY log4j2.properties /spark/conf/log4j2.properties
COPY hadoop-env.sh /hadoop/etc/hadoop/hadoop-env.sh

RUN mkdir /hadoop/logs
RUN mkdir /spark/logs

WORKDIR /root/
COPY entry_shell.sh entry_shell.sh
ENTRYPOINT bash -C './entry_shell.sh'; 'bash'
