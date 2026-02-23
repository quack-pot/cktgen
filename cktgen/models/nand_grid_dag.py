import torch
import dataclasses
from .. import graphs

@dataclasses.dataclass
class NandGridDAGCreateInfo:
    input_node_count: int = 2
    hidden_node_count: int = 5
    output_node_count: int = 1

@dataclasses.dataclass
class NandGridDAGTrainInfo:
    input_data: list[list[float]]
    output_data: list[list[float]]

    epochs: int = 5000
    epoch_print_cadence: int = 500

    learning_rate: float = 0.05

    lambda_edges: float = 1e-4
    lambda_discrete: float = 1e-4

class NandGridDAG():
    ## *=================================================
    ## *
    ## * __init__
    ## *
    ## *=================================================

    def __init__(self, create_info: NandGridDAGCreateInfo) -> None:
        super().__init__()

        # ? Static Sparse DAG Model Parameters

        self.input_node_count: int = int(max(0, create_info.input_node_count))
        self.hidden_node_count: int = int(max(0, create_info.hidden_node_count))
        self.output_node_count: int = int(max(1, create_info.output_node_count))
        self.total_node_count: int = self.input_node_count + self.hidden_node_count + self.output_node_count

        # ? Parameters

        self.edge_gates: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(self.total_node_count, self.total_node_count)
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
        dtype: torch.dtype = self.edge_gates.dtype
        device: torch.device = self.edge_gates.device

        states: list[torch.Tensor] = []

        input_size: int = min(self.input_node_count, values.numel())
        for idx in range(input_size):
            states.append(values[idx])

        for idx in range(input_size, self.input_node_count):
            states.append(torch.zeros((), dtype=dtype, device=device))

        for idx in range(self.input_node_count, self.total_node_count):
            prev_states: torch.Tensor = torch.stack(states)

            edge_gate: torch.Tensor = torch.sigmoid(self.edge_gates[:idx, idx])

            weights: torch.Tensor = self.mask[:idx, idx] * edge_gate

            node_value: torch.Tensor = torch.sigmoid(-5 * (torch.sum(weights * prev_states) - 0.5))
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
        dtype: torch.dtype = self.edge_gates.dtype
        device: torch.device = self.edge_gates.device

        if len(values) > 0:
            values_tensor: torch.Tensor = torch.tensor(values, dtype=dtype, device=device)
            return self.evaluateTensor(values_tensor).tolist()
        
        return self.evaluateTensor(torch.empty(0, dtype=dtype, device=device)).tolist()

    ## *=================================================
    ## *
    ## * train
    ## *
    ## *=================================================

    def train(self, train_info: NandGridDAGTrainInfo) -> None:
        assert (
            len(train_info.input_data) == len(train_info.output_data)
        ), f"Input and output sets do not match in length! (Input = {len(train_info.input_data)}; Output = {len(train_info.output_data)})"

        dtype: torch.dtype = self.edge_gates.dtype
        device: torch.device = self.edge_gates.device

        data_set: torch.Tensor = torch.tensor(train_info.input_data, dtype=dtype, device=device)
        targets: torch.Tensor = torch.tensor(train_info.output_data, dtype=dtype, device=device)

        computeLoss: torch.nn.BCELoss = torch.nn.BCELoss()
        optimizer: torch.optim.Adam = torch.optim.Adam(
            [self.edge_gates],
            lr=train_info.learning_rate,
        )

        for epoch in range(train_info.epochs):
            total_loss: float = 0.0

            annealing_value: float = float(epoch) / float(train_info.epochs)
            lambda_edges: float = train_info.lambda_edges * annealing_value
            lambda_discrete: float = train_info.lambda_discrete * annealing_value

            for x, y in zip(data_set, targets):
                optimizer.zero_grad()

                output: torch.Tensor = self.__internalEvaluateTensor__(x)
                loss: torch.Tensor = computeLoss(output, y)

                edge_activation: torch.Tensor = torch.sigmoid(self.edge_gates) * self.mask

                edge_penalty: torch.Tensor = edge_activation.sum()
                loss += lambda_edges * edge_penalty

                edge_discrete_penalty: torch.Tensor = (edge_activation * (1.0 - edge_activation)).sum()
                loss += lambda_discrete * edge_discrete_penalty

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            if epoch % train_info.epoch_print_cadence == 0:
                print(f"NandGridDAG Training (Epoch = {epoch}; Loss = {total_loss:.4f})")

    ## *=================================================
    ## *
    ## * extractDAG
    ## *
    ## *=================================================

    def extractDAG(self, edge_prune_threshold: float = 0.1) -> graphs.NeuralGraph:
        return graphs.NeuralGraph(graphs.NeuralGraphCreateInfo(
            input_node_count=self.input_node_count,
            output_node_count=self.output_node_count,
            edge_weights=self.mask * torch.sigmoid(self.edge_gates),
            biases=torch.zeros(self.total_node_count),
            edge_prune_threshold=edge_prune_threshold
        ))