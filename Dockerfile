# Use the official Python 3.10 slim image as the base image
FROM python:3.8-slim-buster

# Set the user to root to perform administrative tasks
USER root

# Create a new directory named /app in the container
RUN mkdir /app

# Copy all files from the current directory on the host to /app in the container
COPY . /app/

# Set the working directory to /app
WORKDIR /app/

#for lightgbm algorithm
RUN apt-get update && apt-get install -y libgomp1

# Install the required Python packages listed in requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install pendulum==2.1.2
RUN pip3 install -r requirements.txt

# Set environment variables for Apache Airflow
# AIRFLOW_HOME specifies the directory where Airflow will look for its configuration files
ENV AIRFLOW_HOME="/app/airflow"

# Configure Airflow settings
# Sets the maximum time (in seconds) for importing DAGs
ENV AIRFLOW_CORE_DAGBAG_IMPORT_TIMEOUT=1000

# Enable pickling for XCom, allowing complex objects to be passed between tasks
ENV AIRFLOW_CORE_ENABLE_XCOM_PICKLING=True

# Initialize the Airflow database, creating necessary tables and configurations
RUN airflow db init

# Create an Airflow user with specified email, first name, last name, password, role, and username
RUN airflow users create -e iamprashant2601@gmail.com -f prashant -l jain -p admin -r Admin -u admin

# Change the permissions of the start.sh script to make it executable
RUN chmod 777 start.sh

# Update the package list in the container
RUN apt update -y

# Set the entry point to the shell, allowing for command execution
ENTRYPOINT [ "/bin/sh" ]

# Specify the default command to run when the container starts
CMD ["start.sh"]