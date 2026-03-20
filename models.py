import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

class CAGAT_MinSum_Layer_Lite(nn.Module):
    def __init__(self):
        super(CAGAT_MinSum_Layer_Lite, self).__init__()
        self.min_sum_scaler = nn.Parameter(torch.tensor([0.8])) 
        self.attention_net = nn.Linear(3, 1) 
        self.cycle_penalty = nn.Parameter(torch.tensor([-1.0]))

    def forward(self, node_features, edge_index, cycle_mask):
        src, dst = edge_index
        x_src = node_features[src]
        x_dst = node_features[dst]
        att_input = torch.cat([x_src.unsqueeze(1), x_dst.unsqueeze(1), cycle_mask.unsqueeze(1)], dim=1)
        raw_attention = F.leaky_relu(self.attention_net(att_input).squeeze(-1))
        raw_attention = raw_attention + (cycle_mask * self.cycle_penalty)
        attention_weights = torch.sigmoid(raw_attention) 
        messages = x_src * attention_weights * self.min_sum_scaler
        out = torch.zeros_like(node_features)
        out.scatter_add_(0, dst, messages)
        return out

class CAGAT_MinSum_Layer_True(nn.Module):
    def __init__(self, hidden_dim=16, num_heads=4):
        super(CAGAT_MinSum_Layer_True, self).__init__()
        self.feature_proj = nn.Linear(1, hidden_dim)
        self.attention_net = nn.Linear(2 * hidden_dim + 1, num_heads)
        self.cycle_penalty = nn.Parameter(torch.full((num_heads,), -1.0))
        self.min_sum_scaler = nn.Parameter(torch.tensor([0.8])) 

    def forward(self, node_features, edge_index, cycle_mask):
        src, dst = edge_index
        x_src_hidden = self.feature_proj(node_features[src].unsqueeze(-1))
        x_dst_hidden = self.feature_proj(node_features[dst].unsqueeze(-1))
        att_input = torch.cat([x_src_hidden, x_dst_hidden, cycle_mask.unsqueeze(-1)], dim=-1)
        raw_attention = self.attention_net(att_input)
        raw_attention = F.leaky_relu(raw_attention, negative_slope=0.2)
        raw_attention = raw_attention + (cycle_mask.unsqueeze(-1) * self.cycle_penalty)
        attention_weights = softmax(raw_attention, index=dst, dim=0)
        mean_attention = attention_weights.mean(dim=-1)
        messages = node_features[src] * mean_attention * self.min_sum_scaler
        out = torch.zeros_like(node_features)
        out.scatter_add_(0, dst, messages)
        return out

class NeuralDecoder(nn.Module):
    def __init__(self, num_nodes, pcm, num_iterations=10, shared_weights=False, use_true_gat=False):
        super(NeuralDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.num_iterations = num_iterations
        self.shared_weights = shared_weights
        
        LayerClass = CAGAT_MinSum_Layer_True if use_true_gat else CAGAT_MinSum_Layer_Lite
        
        if self.shared_weights:
            self.shared_layer = LayerClass()
        else:
            self.layers = nn.ModuleList([LayerClass() for _ in range(num_iterations)])
        self.register_buffer('pcm_t', torch.tensor(pcm, dtype=torch.float32).t())

    def forward(self, initial_llrs, edge_index, cycle_mask):
        batch_size = initial_llrs.shape[0]
        check_node_zeros = torch.zeros(batch_size, self.num_nodes - initial_llrs.shape[1], device=initial_llrs.device)
        x = torch.cat([initial_llrs, check_node_zeros], dim=1)
        
        all_iteration_outputs = []
        for i in range(self.num_iterations):
            current_layer = self.shared_layer if self.shared_weights else self.layers[i]
            x_updated = current_layer(x.view(-1), edge_index, cycle_mask)
            x = x_updated.view(batch_size, -1) + torch.cat([initial_llrs, check_node_zeros], dim=1)
            current_vars = x[:, :initial_llrs.shape[1]]
            all_iteration_outputs.append(current_vars)
            
            if not self.training:
                hard_decisions = (current_vars < 0).float()
                syndrome = torch.matmul(hard_decisions, self.pcm_t) % 2
                if torch.sum(syndrome) == 0:
                    break 
        return all_iteration_outputs

class Standard_NeuralBP_Layer(nn.Module):
    def __init__(self):
        super(Standard_NeuralBP_Layer, self).__init__()
        self.learned_weight = nn.Parameter(torch.tensor([0.8])) 

    def forward(self, node_features, edge_index):
        src, dst = edge_index
        x_src = node_features[src]
        messages = x_src * self.learned_weight
        out = torch.zeros_like(node_features)
        out.scatter_add_(0, dst, messages)
        return out

class Standard_NeuralDecoder(nn.Module):
    def __init__(self, num_nodes, pcm, num_iterations=10, shared_weights=False):
        super(Standard_NeuralDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.num_iterations = num_iterations
        self.shared_weights = shared_weights
        
        if self.shared_weights:
            self.shared_layer = Standard_NeuralBP_Layer()
        else:
            self.layers = nn.ModuleList([Standard_NeuralBP_Layer() for _ in range(num_iterations)])
        self.register_buffer('pcm_t', torch.tensor(pcm, dtype=torch.float32).t())

    def forward(self, initial_llrs, edge_index):
        batch_size = initial_llrs.shape[0]
        check_node_zeros = torch.zeros(batch_size, self.num_nodes - initial_llrs.shape[1], device=initial_llrs.device)
        x = torch.cat([initial_llrs, check_node_zeros], dim=1)
        
        all_iteration_outputs = []
        for i in range(self.num_iterations):
            current_layer = self.shared_layer if self.shared_weights else self.layers[i]
            x_updated = current_layer(x.view(-1), edge_index) 
            x = x_updated.view(batch_size, -1) + torch.cat([initial_llrs, check_node_zeros], dim=1)
            current_vars = x[:, :initial_llrs.shape[1]]
            all_iteration_outputs.append(current_vars)
            
            if not self.training:
                hard_decisions = (current_vars < 0).float()
                syndrome = torch.matmul(hard_decisions, self.pcm_t) % 2
                if torch.sum(syndrome) == 0:
                    break 
        return all_iteration_outputs
