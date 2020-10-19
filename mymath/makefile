math: ./src/20209324add.c ./src/20209324sub.c ./src/20209324mul.c ./src/20209324div.c ./include/head.h
	gcc ./src/*.c -I./include -o ./bin/math 
	gcc -shared -o ./lib/libmath.so ./src/*.c
	ar rcvs ./lib/libmath.a ./src/*.c

	



