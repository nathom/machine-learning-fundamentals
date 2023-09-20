#include "data.h"
#include "matrix.h"

struct grad {
  // Column vector of derivatives
  matrix dJ_dW;
  double dJ_db;
};

struct mlinear_model {
  matrix W;
  double b;
};

// x is a row vector, or (1 x n) matrix
double predict(struct mlinear_model *model, const matrix x) {
  // (1 x n) . (n x 1) => (1 x 1) matrix, which is a number
  mfloat result[1][1] = {0.0};
  matrix tmp = {.buf = (mfloat *)result, .rows = 1, .cols = 1};
  // Set tmp to the result
  matrix_dot(tmp, x, model->W);
  return tmp.buf[0] + model->b;
}

double compute_cost(struct mlinear_model *model, matrix X,
                    matrix Y) {
  double cost = 0.0;
  for (int i = 0; i < X.rows; i++) {
    matrix x_i = matrix_get_row(X, i);
    double f_wb = predict(model, x_i);
    double diff = matrix_get(Y, 0, i) - f_wb;
    cost += diff * diff;
  }
  return cost / (2.0 * X.rows);
}

/* Compute gradient and write result to out. */
void compute_gradient(struct grad *out,
                      struct mlinear_model *model, const matrix X,
                      const matrix Y) {
  int m = X.rows;  // number of samples
  int n = X.cols;  // number of features

  // Using tmp to store each row of X
  matrix tmp = matrix_new(1, n);
  for (int i = 0; i < m; i++) {
    // tmp = X^(i)
    matrix curr_row = matrix_get_row(X, i);
    // y_hat = (X^(i) dot W) + b
    double y_hat = predict(model, curr_row);
    // yi = y^(i)
    double yi = matrix_get(Y, 0, i);
    // The term in parentheses
    double err = y_hat - yi;

    /*
     * For dJ_dW, we need to multiply the error
     * by the current row, and add it to the running sum
     */

    // tmp = X^(i) * (y_hat^(i) - y^(i))
    matrix_scalar_mul(tmp, curr_row, err);
    // dJ_dW += tmp
    matrix_ip_T_add(out->dJ_dW, tmp);

    // dJ_db += (y_hat^(i) - y^(i))
    out->dJ_db += err;
  }

  /*
   * I'm going to replace 2/m with 1/m here since the 2
   * can be moved into alpha in the next step.
   */

  // dJ/db = (dJ/db) / m
  out->dJ_db /= m;
  // dJ/dW = (dJ/dW) / m
  matrix_scalar_ip_mul(out->dJ_dW, 1.0 / m);
  matrix_del(tmp);
}

void gradient_descent(struct mlinear_model *model, const matrix X,
                      const matrix Y, const int num_iterations,
                      const double alpha) {
  // reusable buffer for gradient
  int n = X.cols, m = X.rows;
  matrix dJ_dW = matrix_new(n, 1);
  struct grad tmp_grad = {.dJ_dW = dJ_dW, .dJ_db = 0.0};

  for (int i = 0; i < num_iterations; i++) {
    // Log progress
    if (i % (num_iterations >> 4) == 0) {
      printf("\tCost at iteration %d: %f\n", i,
             compute_cost(model, X, Y));
    }
    // tmp_grad = current gradient for the model
    compute_gradient(&tmp_grad, model, X, Y);
    // dJ/dW *= -alpha
    matrix_scalar_ip_mul(tmp_grad.dJ_dW, -alpha);
    // W += dJ/dW
    matrix_ip_add(model->W, tmp_grad.dJ_dW);
    // b += -alpha * dJ/db
    model->b += -alpha * tmp_grad.dJ_db;
  }
  matrix_del(dJ_dW);
}

// normalize the input data using z-score
void z_score_normalize(matrix X) {
  // calculate mean and standard deviation
  int n = X.cols, m = X.rows;
  matrix mean = matrix_new(1, n);
  matrix stdev = matrix_new(1, n);

  for (int i = 0; i < NUM_SAMPLES; i++) {
    matrix_ip_add(mean, matrix_get_row(X, i));
  }
  matrix_scalar_ip_mul(mean, 1.0 / NUM_SAMPLES);

  matrix buf = matrix_new(1, n);
  for (int i = 0; i < NUM_SAMPLES; i++) {
    matrix row = matrix_get_row(X, i);
    matrix_sub(buf, mean, row);
    matrix_ip_square(buf);
    matrix_ip_add(stdev, buf);
  }
  matrix_ip_sqrt(stdev);

  // normalize
  for (int i = 0; i < NUM_SAMPLES; i++) {
    matrix row = matrix_get_row(X, i);
    matrix_ip_sub(row, mean);
    matrix_ip_div(row, stdev);
  }
}

int main() {
  int n = X.cols, m = X.rows;
  z_score_normalize(X);
  matrix W = matrix_new(n, 1);
  struct mlinear_model model = {.W = W, .b = 0.0};
  printf("Initial cost: %f\n", compute_cost(&model, X, Y));
  const int num_iterations = 3e4;
  const double alpha = 1;
  gradient_descent(&model, X, Y, num_iterations, alpha);
  printf("Final cost: %f\n", compute_cost(&model, X, Y));

  printf("Model parameters:\nW=");
  matrix_print(model.W);
  printf(" b=%f\n", model.b);

  return 0;
}

int main2() {
  int n = X.cols, m = X.rows;
  matrix W = matrix_new(n, 1);
  struct mlinear_model model = {.W = W, .b = 0.0};
  printf("Cost of zero model: %f\n", compute_cost(&model, X, Y));
  matrix_set(W, 0, 0, 0.34);
  matrix_set(W, 1, 0, 3.3);
  matrix_set(W, 2, 0, 5.84);
  matrix_set(W, 3, 0, -8.07);
  model.b = 1634.5;
  printf("Cost of guesstimate model: %f\n",
         compute_cost(&model, X, Y));

  return 0;
}
