
import flwr as fl
from flwr.server import ServerConfig
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        logger.info(f"Aggregating results for round {server_round}")
        return super().aggregate_fit(server_round, results, failures)

def start_server():
    strategy = SimpleStrategy(min_fit_clients=3, min_available_clients=3)
    server_config = ServerConfig(num_rounds=3)
    fl.server.start_server(
        server_address="0.0.0.0:8099",
        config=server_config,
        strategy=strategy
    )

if __name__ == "__main__":
    logger.info("Starting Flower server...")
    start_server()
