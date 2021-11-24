all: ;nvcc ./*.cu -w --relocatable-device-code=true -I. -o fs
clean: ;rm fs