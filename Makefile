CC = mpicc
CFLAGS = -Wall -O3
INCLUDES = -I./include
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = .

SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TARGET = summa

all: directories $(TARGET)

directories:
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(INC_DIR)/*.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

run:

#Medium Matrices -Square  16384 and 32768 4096  16384 32768
	mpirun -np 4 ./summa -m 100 -n 100 -k 100 -b 4 -s a -v -p
	mpirun -np 4 ./summa -m 100 -n 100 -k 100 -b 4 -s b -v -p
	mpirun -np 4 ./summa -m 100 -n 100 -k 100 -b 4 -s c -v -p



.PHONY: all clean run directories


