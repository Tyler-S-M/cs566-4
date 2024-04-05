all:
	nvcc -std=c++11 -arch=native reduce.cu -o main

clean:
	rm main
