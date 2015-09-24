export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
CXXFLAGS = -std=gnu++11 -Wall -Wextra -g -Og `pkg-config --cflags opencv`
LDLIBS =  `pkg-config --libs opencv`

all: webcam

webcam: webcam.cpp

.PHONY: clean

clean:
	$(RM) webcam
