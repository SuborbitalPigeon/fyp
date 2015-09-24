export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
CXXFLAGS = -std=gnu++11 -Wall -Wextra -g -Og `pkg-config --cflags opencv`
LDLIBS =  `pkg-config --libs opencv`

all: testwebcam

testwebcam: webcamcontroller.o testwebcam.o
	$(CXX) $(LDLIBS) -o testwebcam webcamcontroller.o testwebcam.o

.PHONY: all clean

clean:
	$(RM) testwebcam *.o
