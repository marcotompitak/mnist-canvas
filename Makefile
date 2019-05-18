docker_build:
	docker build -t mnist-canvas .

docker_run:
	docker run -e PORT=5000 -p 5000:5000 mnist-canvas