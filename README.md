# class7-hwk
# Docker instructions
To build the Docker image in current directory, type in BASH: docker build -t mdjoh/class7-image:v01 ./

To show Docker image, type in BASH: docker images

To run the Docker image via volume mounting, type in BASH: docker run -ti -v /localFolderWithDataAndCode:/virtualContainerFolder -w /virtualContainerFolder mdjoh/class7-image:v01

Replace /localFolderWithDataAndCode to your local machine directory that has the data and python code file
