import torch
import dataclasses
from .. import graphs

@dataclasses.dataclass
class StandardNeuralDAGCreateInfo:
    input_node_count: int = 2
    hidden_node_count: int = 5
    output_node_count: int = 1

@dataclasses.dataclass
class StandardNeuralDAGTrainInfo:
    input_data: list[list[float]]
    output_data: list[list[float]]

    epochs: int = 5000
    epoch_print_cadence: int = 500

    learning_rate: float = 0.05

    lambda_edges: float = 1e-4
    lambda_nodes: float = 1e-3
    node_strength_sharpness: float = 10.0

class StandardNeuralDAG():
    ## *=================================================
    ## *
    ## * __init__
    ## *
    ## *=================================================

    def __init__(self, create_info: StandardNeuralDAGCreateInfo) -> None:
        super().__init__()

        # ? Static Sparse DAG Model Parameters

        self.input_node_count: int = int(max(0, create_info.input_node_count))
        self.hidden_node_count: int = int(max(0, create_info.hidden_node_count))
        self.output_node_count: int = int(max(1, create_info.output_node_count))
        self.total_node_count: int = self.input_node_count + self.hidden_node_count + self.output_node_count

        # ? Parameters

        self.weights: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(self.total_node_count, self.total_node_count)
        )

        self.biases: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(self.total_node_count),
        )

        # ? Upper Triangular Mask (Enforces DAG Structure)

        self.mask: torch.Tensor = torch.triu(
            torch.ones(self.total_node_count, self.total_node_count),
            diagonal=1,
        )

    ## *=================================================
    ## *
    ## * __internalEvaluateTensor__
    ## *
    ## *=================================================

    def __internalEvaluateTensor__(self, values: torch.Tensor) -> torch.Tensor:
        dtype: torch.dtype = self.weights.dtype
        device: torch.device = self.weights.device

        states: list[torch.Tensor] = []

        input_size: int = min(self.input_node_count, values.numel())
        for idx in range(input_size):
            states.append(values[idx])

        for idx in range(input_size, self.input_node_count):
            states.append(torch.zeros((), dtype=dtype, device=device))

        for idx in range(self.input_node_count, self.total_node_count):
            prev_states: torch.Tensor = torch.stack(states)

            weights: torch.Tensor = self.weights[:idx, idx] * self.mask[:idx, idx]
            weighted_sum: torch.Tensor = torch.dot(weights, prev_states)

            node_value: torch.Tensor = torch.sigmoid(weighted_sum + self.biases[idx])
            states.append(node_value)

        return torch.stack(states[-self.output_node_count:])
    
    ## *=================================================
    ## *
    ## * evaluateTensor
    ## *
    ## *=================================================

    def evaluateTensor(self, values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.__internalEvaluateTensor__(values)

    ## *=================================================
    ## *
    ## * evaluate
    ## *
    ## *=================================================

    def evaluate(self, values: list[float]) -> list[float]:
        dtype: torch.dtype = self.weights.dtype
        device: torch.device = self.weights.device

        if len(values) > 0:
            values_tensor: torch.Tensor = torch.tensor(values, dtype=dtype, device=device)
            return self.evaluateTensor(values_tensor).tolist()
        
        return self.evaluateTensor(torch.empty(0, dtype=dtype, device=device)).tolist()

    ## *=================================================
    ## *
    ## * train
    ## *
    ## *=================================================

    def train(self, train_info: StandardNeuralDAGTrainInfo) -> None:
        assert (
            len(train_info.input_data) == len(train_info.output_data)
        ), f"Input and output sets do not match in length! (Input = {len(train_info.input_data)}; Output = {len(train_info.output_data)})"

        dtype: torch.dtype = self.weights.dtype
        device: torch.device = self.weights.device

        data_set: torch.Tensor = torch.tensor(train_info.input_data, dtype=dtype, device=device)
        targets: torch.Tensor = torch.tensor(train_info.output_data, dtype=dtype, device=device)

        computeLoss: torch.nn.BCELoss = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(
            [self.weights, self.biases],
            lr=train_info.learning_rate,
        )

        for epoch in range(train_info.epochs):
            total_loss: float = 0.0

            annealing_value: float = float(epoch) / float(train_info.epochs)
            lambda_edges: float = train_info.lambda_edges * annealing_value
            lambda_nodes: float = train_info.lambda_nodes * annealing_value

            for x, y in zip(data_set, targets):
                optimizer.zero_grad()

                output: torch.Tensor = self.__internalEvaluateTensor__(x)
                loss: torch.Tensor = computeLoss(output, y)

                outgoing_edges: torch.Tensor = torch.abs(self.weights * self.mask)

                edge_penalty: torch.Tensor = torch.abs(outgoing_edges).sum()
                loss += lambda_edges * edge_penalty

                outgoing_sum: torch.Tensor = torch.sum(outgoing_edges, dim=1)
                node_penalty: torch.Tensor = (1.0 - torch.exp(-train_info.node_strength_sharpness * outgoing_sum)).sum()
                loss += lambda_nodes * node_penalty

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            if epoch % train_info.epoch_print_cadence == 0:
                print(f"StaticSparseDAG Training (Epoch = {epoch}; Loss = {total_loss:.4f})")

    ## *=================================================
    ## *
    ## * extractDAG
    ## *
    ## *=================================================

    def extractDAG(self, edge_prune_threshold: float = 0.1) -> graphs.NeuralGraph:
        return graphs.NeuralGraph(graphs.NeuralGraphCreateInfo(
            input_node_count=self.input_node_count,
            output_node_count=self.output_node_count,
            edge_weights=self.weights * self.mask,
            biases=self.biases,
            edge_prune_threshold=edge_prune_threshold
        ))