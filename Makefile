CC = g++
CFLAGS = -Wall -w
SRCS = hificode_OpenCV.cpp
PROG = hificode_OpenCV

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)