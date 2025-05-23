import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import warnings
import os
import multiprocessing as mp
import copy
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('threat_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Lightweight modality-specific encoders
class NetworkEncoder(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3)
        self.fc = nn.Linear(32 * (input_dim - 2), hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return x

class LogEncoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
    
    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n.squeeze(0)

# Simple attention-based fusion
class FusionLayer(nn.Module):
    def __init__(self, hidden_dim=64, num_modalities=2):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * num_modalities, num_modalities)
        self.fc = nn.Linear(hidden_dim * num_modalities, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, modalities):
        concat = torch.cat(modalities, dim=1)
        weights = self.softmax(self.attention(concat))
        weighted = sum(weights[:, i].unsqueeze(1) * m for i, m in enumerate(modalities))
        return self.fc(concat)

# Threat detection model
class ThreatDetector(nn.Module):
    def __init__(self, input_dim_net=18, input_dim_log=10, hidden_dim=64):
        super().__init__()
        self.net_encoder = NetworkEncoder(input_dim_net, hidden_dim)
        self.log_encoder = LogEncoder(input_dim_log, hidden_dim)
        self.fusion = FusionLayer(hidden_dim, num_modalities=2)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, net_data, log_data):
        net_emb = self.net_encoder(net_data)
        log_emb = self.log_encoder(log_data)
        fused = self.fusion([net_emb, log_emb])
        return self.classifier(fused)

# Load and preprocess CICIDS2017 data
def load_cicids2017_data(data_dir, num_samples=1000):
    logger.info(f"Loading CICIDS2017 data from {data_dir}")
    try:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            logger.error("No CSV files found in the directory")
            raise FileNotFoundError("No CSV files found")
        
        dfs = []
        for file in csv_files:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(data)} samples from {len(csv_files)} files")
        
        data.columns = data.columns.str.strip()
        feature_cols = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow Bytes/s', 'Flow Packets/s', 'Fwd IAT Total', 'Bwd IAT Total',
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'Packet Length Mean', 'Packet Length Std', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count'
        ]
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}. Using available features")
            feature_cols = [col for col in feature_cols if col in data.columns]
        
        X = data[feature_cols].fillna(0).values
        y = (data['Label'] != 'BENIGN').astype(int).values
        
        X, _, y, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        log_train = np.random.randn(len(X_train), 5, 10)
        log_val = np.random.randn(len(X_val), 5, 10)
        
        logger.info(f"Prepared {len(X_train)} training and {len(X_val)} validation samples")
        return (torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(log_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(log_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long))
    except Exception as e:
        logger.error(f"Failed to load CICIDS2017 data: {e}")
        raise

# Federated learning client function
def client_fn(client_id, train_data, val_data, global_params, result_queue):
    logger.info(f"Client {client_id} starting")
    device = torch.device("cpu")
    
    # Initialize model
    model = ThreatDetector(input_dim_net=18, input_dim_log=10, hidden_dim=64).to(device)
    if global_params:
        state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), global_params)}
        model.load_state_dict(state_dict)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    
    # Train
    model.train()
    total_loss = 0
    try:
        for _ in range(1):  # 1 local epoch
            for net_data, log_data, labels in train_loader:
                net_data, log_data, labels = net_data.to(device), log_data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(net_data, log_data)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                
                # Manual differential privacy
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad)
                        if grad_norm > 1.0:
                            param.grad.mul_(1.0 / (grad_norm + 1e-6))
                        noise = torch.normal(mean=0.0, std=1.0, size=param.grad.shape, device=param.grad.device)
                        param.grad.add_(noise)
                
                optimizer.step()
                total_loss += loss.item()
        logger.info(f"Client {client_id} training completed, loss: {total_loss:.4f}")
    except Exception as e:
        logger.error(f"Client {client_id} training failed: {e}")
        raise
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for net_data, log_data, labels in val_loader:
                net_data, log_data, labels = net_data.to(device), log_data.to(device), labels.to(device)
                outputs = model(net_data, log_data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        logger.info(f"Client {client_id} evaluation completed, accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Client {client_id} evaluation failed: {e}")
        raise
    
    # Return model parameters and metrics
    params = [val.cpu().numpy() for val in model.state_dict().values()]
    result_queue.put((client_id, params, total_loss, accuracy))

# Aggregate client parameters (FedAvg)
def aggregate_parameters(client_params):
    logger.info("Aggregating client parameters")
    num_clients = len(client_params)
    aggregated_params = []
    
    # Initialize with first client's parameters
    for param in client_params[0][1]:
        aggregated_params.append(np.zeros_like(param))
    
    # Sum parameters across clients
    for _, params, _, _ in client_params:
        for i, param in enumerate(params):
            aggregated_params[i] += param / num_clients
    
    logger.info("Aggregation completed")
    return aggregated_params

def main():
    logger.info("Starting federated learning simulation")
    device = torch.device("cpu")
    torch.set_num_threads(4)
    
    # Load data
    data_dir = "C:\\Users\\Samsung\\OneDrive\\Desktop\\Random Projects Cause I Was Bored\\People\\Adi Senior Research\\MachineLearningCVE"
    try:
        X_train, log_train, y_train, X_val, log_val, y_val = load_cicids2017_data(data_dir, num_samples=1000)
    except Exception as e:
        logger.error("Failed to start due to data loading error")
        return
    
    # Split data for 3 clients (simulating different organizations)
    num_clients = 3
    train_size = len(X_train) // num_clients
    val_size = len(X_val) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * train_size
        end_idx = (i + 1) * train_size if i < num_clients - 1 else len(X_train)
        val_start_idx = i * val_size
        val_end_idx = (i + 1) * val_size if i < num_clients - 1 else len(X_val)
        
        train_dataset = TensorDataset(
            X_train[start_idx:end_idx],
            log_train[start_idx:end_idx],
            y_train[start_idx:end_idx]
        )
        val_dataset = TensorDataset(
            X_val[val_start_idx:val_end_idx],
            log_val[val_start_idx:val_end_idx],
            y_val[val_start_idx:val_end_idx]
        )
        client_datasets.append((train_dataset, val_dataset))
    
    # Run federated learning for 3 rounds
    global_params = None
    num_rounds = 3
    
    for round_idx in range(num_rounds):
        logger.info(f"Starting round {round_idx + 1}")
        result_queue = mp.Queue()
        processes = []
        
        # Start client processes
        for i in range(num_clients):
            train_data, val_data = client_datasets[i]
            p = mp.Process(
                target=client_fn,
                args=(i + 1, train_data, val_data, global_params, result_queue)
            )
            processes.append(p)
            p.start()
        
        # Collect results
        client_results = []
        for _ in range(num_clients):
            client_results.append(result_queue.get())
        
        # Wait for processes to finish
        for p in processes:
            p.join()
        
        # Aggregate parameters
        global_params = aggregate_parameters(client_results)
        
        # Evaluate global model
        global_model = ThreatDetector(input_dim_net=18, input_dim_log=10, hidden_dim=64).to(device)
        state_dict = {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), global_params)}
        global_model.load_state_dict(state_dict)
        global_model.eval()
        
        # Evaluate on combined validation set
        val_dataset = TensorDataset(X_val, log_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=16)
        correct = 0
        total = 0
        with torch.no_grad():
            for net_data, log_data, labels in val_loader:
                net_data, log_data, labels = net_data.to(device), log_data.to(device), labels.to(device)
                outputs = global_model(net_data, log_data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        global_accuracy = correct / total
        logger.info(f"Round {round_idx + 1} global model accuracy: {global_accuracy:.4f}")

    logger.info("Federated learning simulation completed")

if __name__ == "__main__":
    main()