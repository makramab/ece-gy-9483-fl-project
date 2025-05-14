import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple

# Define a weighted‐average aggregation for accuracy
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute the weighted average of client accuracies."""
    # metrics is a list of tuples (num_examples, { "accuracy": accuracy_value })
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def main():
    # Configure FedAvg to use our accuracy aggregation
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        # Tell Flower how to aggregate the {"accuracy": …} dicts from clients:
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()

