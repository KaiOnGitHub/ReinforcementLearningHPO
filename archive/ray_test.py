import ray
from ray import tune


def objective(x, a, b):  # Define an objective function.
    return a * (x ** 0.5) + b


def trainable(config):  # Pass a "config" dictionary into your trainable.

    for x in range(20):  # "Train" for 20 iterations and compute intermediate scores.
        score = objective(x, config["a"], config["b"])

        tune.report(score=score)  # Send the score to Tune.

search_space = {
    "a": tune.uniform(0,1),
    "b": tune.uniform(0,1)
}

tune.run(trainable, config=search_space, num_samples=5)