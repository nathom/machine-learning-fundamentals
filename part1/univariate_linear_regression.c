#include <assert.h>
#include <stdio.h>

struct linear_model {
    double w;
    double b;
};

double
predict(struct linear_model m, double x)
{
    return m.w * x + m.b;  // y hat
}

double
loss(double y_hat, double y)
{
    double diff = y_hat - y;
    return diff * diff;
}

double
error(struct linear_model model, double *xs, double *ys, int m)
{
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        double y_hat = predict(model, xs[i]);
        double _loss = loss(y_hat, ys[i]);
        sum += _loss;
    }
    return sum / ((double)m);
}

struct linear_model
optimize_brute_force(double *xs, double *ys, int m)
{
    const double min = -1000.0, max = 1000.0;
    const int num_samples = 1e4;
    const double increment = (max - min) / ((double)num_samples);

    double min_error = 1e10;
    struct linear_model curr;
    struct linear_model optimal;
    for (int i = 0; i < num_samples; i++) {
        curr.w = min + increment * i;
        for (int j = 0; j < num_samples; j++) {
            curr.b = min + increment * j;

            double err = error(curr, xs, ys, m);
            if (err < min_error) {
                min_error = err;
                optimal = curr;
            }
        }
    }
    return optimal;
}

struct gradient {
    double dJ_dw;
    double dJ_db;
};

struct gradient
calculate_gradient(struct linear_model model, double *xs, double *ys, int m)
{
    double dJ_dw = 0.0, dJ_db = 0.0;
    for (int i = 0; i < m; i++) {
        double y_hat = predict(model, xs[i]);
        double diff = y_hat - ys[i];
        dJ_db += diff;
        dJ_dw += diff * xs[i];
    }
    // We're going push that factor of 2 into alpha to save a multiplication
    dJ_dw /= ((double)m);
    dJ_db /= ((double)m);
    return (struct gradient){.dJ_dw = dJ_dw, .dJ_db = dJ_db};
}

struct linear_model
optimize_gradient_descent(double *xs, double *ys, int m, int num_iterations, double alpha)
{
    struct linear_model model = {0.0, 0.0};
    for (int i = 0; i < num_iterations; i++) {
        struct gradient g = calculate_gradient(model, xs, ys, m);
        model.w -= alpha * g.dJ_dw;
        model.b -= alpha * g.dJ_db;
    }
    return model;
}

int
main()
{
    // House sizes, in square feet. Our x variable
    static double xs[] = {2360, 5272, 4592, 1966, 5926, 4944, 4671, 4419, 1630, 3185};
    // Corresponding house prices, in 1000s of dollars. Our y variable
    static double ys[] = {1359, 2576, 2418, 655, 2531, 2454, 2657, 1541, 1057, 1657};
    assert(sizeof(xs) == sizeof(ys));

    int m = sizeof(xs) / sizeof(double);
    struct linear_model optimal_bf = optimize_brute_force(xs, ys, m);
    printf("Brute force optimization model: {w: %f, b: %f}\nError: %f\n", optimal_bf.w,
           optimal_bf.b, error(optimal_bf, xs, ys, m));

    const int num_iterations = 1e8;
    const double alpha = 1e-7;
    struct linear_model optimal_gd = optimize_gradient_descent(xs, ys, m, num_iterations, alpha);
    printf("Gradient descent optimization model: {w: %f, b: %f}\nError: %f\n", optimal_gd.w,
           optimal_gd.b, error(optimal_gd, xs, ys, m));
}
