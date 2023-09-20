CC = gcc
CFLAGS = -Wall -O3 -I.

all: univariate_linear_regression

univariate_linear_regression: univariate_linear_regression.c
	$(CC) $(CFLAGS) -o univariate_linear_regression univariate_linear_regression.c

run: univariate_linear_regression
	./univariate_linear_regression

clean:
	rm -f univariate_linear_regression
