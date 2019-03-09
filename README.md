# class7-hwk
# Docker instructions
To build the Docker image in current directory, type in BASH: docker build -t mdjoh/class7-image:v01 ./

To show Docker image, type in BASH: docker images

To run the Docker image, type in BASH: docker run -t mdjoh/class7-image:v01

The python code to plot the data will be in the Dockerfile and thus will be in the Docker container.
I know this is incomplete. I couldn't get the CMD, ENTRYPOINT, or volume mounting thing to work. But by the end of the course...I will triumph against Docker!
