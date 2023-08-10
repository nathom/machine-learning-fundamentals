import json
import os

import matplotlib.pyplot as plt
import numpy as np


def generate_approx_linear_data(n) -> tuple[list, list]:
    np.random.seed(42)
    # Generate random x data
    x = np.random.randint(1500, 6000, size=n)

    # Generate y data with a linear relationship and some noise
    y = x // 2 + 100 + np.random.normal(0, 1000, size=n)
    return x, list(map(int, y))


def plot(xs, ys):
    plt.scatter(xs, ys, color="blue", label="Data Points")
    plt.xlabel("Square Footage of House")
    plt.ylabel("House Price ($1000s)")
    plt.title("House Price vs. Square Footage", color="white")
    plt.legend()

    # Save the plot as an SVG file
    plt.savefig("housing_data.svg", format="svg", transparent=True)


def plot_predicted_line(xs, ys, w, b):
    yhats = xs * w + b
    plt.scatter(xs, ys, color="blue", label="Data Points")
    plt.plot(xs, yhats, color="red", label=f"Predicted with {w=} {b=}")
    plt.xlabel("Square Footage of House")
    plt.ylabel("House Price ($1000s)")
    plt.title("House Price vs. Square Footage", color="white")
    plt.legend()

    # Save the plot as an SVG file
    plt.savefig("housing_data_with_prediction.svg", format="svg", transparent=True)


if __name__ == "__main__":
    ax = plt.gca()
    ax.set_facecolor("none")

    # Change the color of the axis lines to blue
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")

    # Change the color of the tick parameters to green
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # Change the color of the axis labels to red
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    xs, ys = generate_approx_linear_data(10)
    # plot(xs, ys)
    # plot_predicted_line(xs, ys, 0.4, 331.8)
    plot_predicted_line(xs, ys, 0.451677, 124.026882)
