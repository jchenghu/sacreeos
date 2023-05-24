CC = c++
OPT = -O3
CFLAGS = -fPIC
LDFLAGS = -shared
LIB = -std=c++11 -fopenmp
INC_DIR = includes
INC += -I$(INC_DIR)

SRC_DIR = src
OBJ_DIR	= obj

SRC = $(wildcard $(SRC_DIR)/*.cc)
OBJ = $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(SRC))

c_core_lib.so: $(OBJ)
	$(CC) $(CFLAGS) $(LIB) -o $@ $(OBJ_DIR)/*.o $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CC) $(CFLAGS) $(OPT) $(LIB) $(INC) -c $< -o $@


clean:
	$(RM) $(OBJ_DIR)/* *.out
	$(RM) c_core_lib.so
