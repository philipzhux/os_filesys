all: ;nvcc ./*.cu --relocatable-device-code=true -I. -o fs
clean: ;rm fs