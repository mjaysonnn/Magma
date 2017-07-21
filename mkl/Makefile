target_name = dgesv.c
target_name1 = dgesv1.c
OBJ = dgesv
OBJ1 = dgesv1
all : test test1

test :
	gcc -DMKL_ILP64 -fopenmp -m64 -I$MKLROOT/include $(target_name) -L$MKLROOT/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread -lm -o $(OBJ)
test1 :
	gcc -DMKL_ILP64 -fopenmp -m64 -I$MKLROOT/include $(target_name1) -L$MKLROOT/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread -lm -o $(OBJ1)
clean : clean1 clean2

clean1 :
	rm -rf $(OBJ)
clean2 :
	rm -rf $(OBJ1)
