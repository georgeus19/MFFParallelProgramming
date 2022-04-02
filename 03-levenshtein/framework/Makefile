CPP=g++
CFLAGS=-Wall -O3 -std=c++17 -mavx2 -fopenmp
INCLUDE=. ./internal
LDFLAGS=
LIBS=
LIBDIRS=
SOURCE=levenshtein.cpp
MYSOURCE=test.cpp
HEADERS=$(shell find . -name '*.hpp')
EXECUTABLE=./levenshtein
MYEXECUTABLE=./test


.PHONY: all clear clean purge

all: $(EXECUTABLE) $(MYEXECUTABLE)



# Building Targets

$(EXECUTABLE): $(SOURCE) $(HEADERS)
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $< -o $@



$(MYEXECUTABLE): $(MYSOURCE) $(HEADERS)
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $< -o $@


# Cleaning Stuff

clear:
	@echo Removing all generated files...
	-@rm -f $(EXECUTABLE)

clean: clear

purge: clear
