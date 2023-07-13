DOCKER_IMAGE_NAME:=ink-detection
DOCKER_WORKDIR:=/opt/kaggle-ink-detection
DOCKER_DATA_PATH:=/kaggle/input
LOCAL_DATA_PATH:=/home/yusuke/Datasets/kaggle

.PhONY: build-docker
build-docker:
	echo $(LOCAL_DATA_PATH)
	# @docker build \
	# 	--build-arg WORKDIR=$(DOCKER_WORKDIR) \
	# 	--build-arg HOME=$(HOME) \
	# 	-t ${DOCKER_IMAGE_NAME} \
	# 	-f docker/Dockerfile \
	# 	.

.PHONY: run-docker
run-docker:
	@rocker --nvidia --x11 --user --privileged \
		--name jupyter --network host \
		--oyr-run-arg "--workdir $(DOCKER_WORKDIR) --ipc=host" \
		--volume \
			$(CURDIR):$(DOCKER_WORKDIR) \
			$(LOCAL_DATA_PATH):$(DOCKER_DATA_PATH) \
		-- \
		$(DOCKER_IMAGE_NAME)

.PHONY: run-jupyter
run-jupyter:
	@docker run --network host --ipc host --gpus all \
		--workdir $(DOCKER_WORKDIR) -v $(CURDIR):$(DOCKER_WORKDIR) \
		-v $(LOCAL_DATA_PATH):$(DOCKER_DATA_PATH) \
		$(DOCKER_IMAGE_NAME) jupyter lab --allow-root
