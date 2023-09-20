#ifndef MATRIX_H
#define MATRIX_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef double mfloat;

typedef struct {
  mfloat *buf;
  int rows;
  int cols;
} matrix;

static inline matrix matrix_from_buf(double *buf, int rows,
                                     int cols) {
  return (matrix){
      .buf = buf,
      .rows = rows,
      .cols = cols,
  };
}

static inline mfloat matrix_get(matrix m, int row, int col) {
  if (!(row >= 0 && col >= 0 && row < m.rows && col < m.cols)) {
    fprintf(stderr,
            "matrix_get: Index out of bounds (%d, %d) for matrix "
            "size (%d, %d)\n",
            row, col, m.rows, m.cols);
    exit(1);
  }

  return m.buf[row * m.cols + col];
}

static inline void matrix_set(matrix m, int row, int col,
                              mfloat val) {
  if (!(row >= 0 && col >= 0 && row < m.rows && col < m.cols)) {
    fprintf(stderr,
            "matrix_set: Index out of bounds (%d, %d) for matrix "
            "size (%d, %d)\n",
            row, col, m.rows, m.cols);
    exit(1);
  }

  m.buf[row * m.cols + col] = val;
}

static inline void mat_bounds_check_elementwise(const matrix out,
                                                const matrix m1,
                                                const matrix m2) {
  if (m1.rows != m2.rows || m1.cols != m2.cols ||
      out.rows != m1.rows || out.cols != m1.cols) {
    fprintf(stderr,
            "Incompatible dimensions for elementwise operation "
            "(%d, %d) & (%d, %d) => (%d, %d) \n",
            m1.rows, m1.cols, m2.rows, m2.cols, out.rows,
            out.cols);
    exit(1);
  }
}

static inline void mat_bounds_check_elementwise_ip(
    matrix m1, const matrix m2) {
  if (m1.rows != m2.rows || m1.cols != m2.cols) {
    fprintf(stderr,
            "Incompatible dimensions for elementwise in-place "
            "operation (%d, %d) & (%d, %d) \n",
            m1.rows, m1.cols, m2.rows, m2.cols);
    exit(1);
  }
}

static inline void mat_bounds_check_elementwise_T(
    const matrix out, const matrix m1, const matrix m2) {
  if (m1.rows != m2.cols || m1.cols != m2.rows ||
      out.rows != m1.rows || out.cols != m1.cols) {
    fprintf(stderr,
            "Incompatible dimensions for elementwise transposed "
            "operation (%d, %d) & (%d, %d) \n",
            m1.rows, m1.cols, m2.rows, m2.cols);
    exit(1);
  }
}

static inline void mat_bounds_check_elementwise_ip_T(
    matrix m1, const matrix m2) {
  if (m1.rows != m2.cols || m1.cols != m2.rows) {
    fprintf(stderr,
            "Incompatible dimensions for elementwise transposed "
            "in-place operation (%d, %d) & "
            "(%d, %d) \n",
            m1.rows, m1.cols, m2.rows, m2.cols);
    exit(1);
  }
}

static inline void matrix_dot(matrix out, const matrix m1,
                              const matrix m2) {
  if (m1.cols != m2.rows) {
    printf(
        "matrix dot: dimension error %d, %d not compat w/ "
        "%d,%d\n",
        m1.rows, m1.cols, m2.rows, m2.cols);
    exit(1);
  }
  for (int row = 0; row < m1.rows; row++) {
    for (int col = 0; col < m2.cols; col++) {
      double sum = 0.0;
      for (int k = 0; k < m1.cols; k++) {
        double x1 = matrix_get(m1, row, k);
        double x2 = matrix_get(m2, k, col);
        sum += x1 * x2;
      }
      matrix_set(out, row, col, sum);
    }
  }
}

static inline void matrix_print(matrix m) {
  for (int i = 0; i < m.rows; i++) {
    printf("[ ");
    for (int j = 0; j < m.cols; j++) {
      printf("%.4e", matrix_get(m, i, j));
      printf(" ");
    }
    printf("]\n");
  }
  printf("\n");
}

matrix matrix_new(int rows, int cols) {
  double *buf = calloc(rows * cols, sizeof(double));
  if (buf == NULL) {
    printf("Fail with calloc\n");
    exit(1);
  }

  return (matrix){
      .buf = buf,
      .rows = rows,
      .cols = cols,
  };
}

void matrix_del(matrix m) { free(m.buf); }

matrix matrix_get_row(matrix m, int row) {
  return (matrix){
      .buf = m.buf + row * m.cols, .rows = 1, .cols = m.cols};
}

#define INIT_ZEROS(m, rows, cols)        \
  mfloat _buf_##m[rows][cols] = {{0.0}}; \
  m = matrix_from_buf((mfloat *)_buf_##m, rows, cols)

#define INIT_ID(m, size)                               \
  mfloat _buf_##m[size][size] = {{0.0}};               \
  for (int i = 0; i < size; i++) _buf_##m[i][i] = 1.0; \
  m = matrix_from_buf((mfloat *)_buf_##m, size, size)

#define MAT_ELEMENTWISE_LOOP        \
  for (int i = 0; i < m1.rows; i++) \
    for (int j = 0; j < m1.cols; j++)

#define DEF_MAT_ELEMENTWISE_BUF(opname, op)           \
  static inline void matrix_##opname(                 \
      matrix out, const matrix m1, const matrix m2) { \
    mat_bounds_check_elementwise(out, m1, m2);        \
    MAT_ELEMENTWISE_LOOP {                            \
      mfloat x = matrix_get(m1, i, j);                \
      mfloat y = matrix_get(m2, i, j);                \
      matrix_set(out, i, j, op);                      \
    }                                                 \
  }

#define DEF_MAT_ELEMENTWISE_BUF_T(opname, op)         \
  static inline void matrix_T_##opname(               \
      matrix out, const matrix m1, const matrix m2) { \
    mat_bounds_check_elementwise_T(out, m1, m2);      \
    MAT_ELEMENTWISE_LOOP {                            \
      mfloat x = matrix_get(m1, i, j);                \
      mfloat y = matrix_get(m2, j, i);                \
      matrix_set(out, i, j, op);                      \
    }                                                 \
  }

#define DEF_MAT_ELEMENTWISE_IP(opname, op)                 \
  static inline void matrix_ip_##opname(matrix m1,         \
                                        const matrix m2) { \
    mat_bounds_check_elementwise_ip(m1, m2);               \
    MAT_ELEMENTWISE_LOOP {                                 \
      mfloat x = matrix_get(m1, i, j);                     \
      mfloat y = matrix_get(m2, i, j);                     \
      matrix_set(m1, i, j, op);                            \
    }                                                      \
  }

#define DEF_MAT_ELEMENTWISE_IP_T(opname, op)                 \
  static inline void matrix_ip_T_##opname(matrix m1,         \
                                          const matrix m2) { \
    mat_bounds_check_elementwise_ip_T(m1, m2);               \
    MAT_ELEMENTWISE_LOOP {                                   \
      mfloat x = matrix_get(m1, i, j);                       \
      mfloat y = matrix_get(m2, j, i);                       \
      matrix_set(m1, i, j, op);                              \
    }                                                        \
  }

#define DEF_MAT_SCALAR_BUF(opname, op)         \
  static inline void matrix_scalar_##opname(   \
      matrix out, matrix m1, const mfloat y) { \
    MAT_ELEMENTWISE_LOOP {                     \
      mfloat x = matrix_get(m1, i, j);         \
      matrix_set(out, i, j, op);               \
    }                                          \
  }

#define DEF_MAT_SCALAR_IP(opname, op)                            \
  static inline void matrix_scalar_ip_##opname(matrix m1,        \
                                               const mfloat y) { \
    MAT_ELEMENTWISE_LOOP {                                       \
      mfloat x = matrix_get(m1, i, j);                           \
      matrix_set(m1, i, j, op);                                  \
    }                                                            \
  }

#define DEF_MAT_SINGLE_ARG_IP(opname, op)            \
  static inline void matrix_ip_##opname(matrix m1) { \
    MAT_ELEMENTWISE_LOOP {                           \
      mfloat x = matrix_get(m1, i, j);               \
      matrix_set(m1, i, j, op);                      \
    }                                                \
  }

#define DEF_ALL_OPS(OP_MACRO) \
  OP_MACRO(sub, (x - y));     \
  OP_MACRO(add, (x + y));     \
  OP_MACRO(div, (x / y));     \
  OP_MACRO(mul, (x * y));

DEF_ALL_OPS(DEF_MAT_ELEMENTWISE_BUF)
DEF_ALL_OPS(DEF_MAT_ELEMENTWISE_BUF_T)
DEF_ALL_OPS(DEF_MAT_ELEMENTWISE_IP)
DEF_ALL_OPS(DEF_MAT_ELEMENTWISE_IP_T)
DEF_ALL_OPS(DEF_MAT_SCALAR_IP)
DEF_ALL_OPS(DEF_MAT_SCALAR_BUF)

DEF_MAT_SINGLE_ARG_IP(sqrt, (sqrt(x)))
DEF_MAT_SINGLE_ARG_IP(square, (x * x))

#endif
