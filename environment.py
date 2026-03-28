import os
import torch
import numpy as np
import sionna
from sionna.phy.fec.ldpc import LDPC5GEncoder
from sionna.phy.utils import ebnodb2no
from sionna.phy.mapping import BinarySource

class LDPCEnvironment:
    def __init__(self, k=100, n=200):
        self.k = k
        self.n = n
        self.encoder = LDPC5GEncoder(k, n)
        self.pcm = self.encoder.pcm.to_dense().numpy()
        self.num_check_nodes, self.num_var_nodes = self.pcm.shape
        self.edge_index = self._build_tanner_graph()

    def _build_tanner_graph(self):
        edges_source, edges_target = [], []
        for c in range(self.num_check_nodes):
            for v in range(self.num_var_nodes):
                if self.pcm[c, v] == 1:
                    check_id = c + self.num_var_nodes
                    edges_source.extend([v, check_id])
                    edges_target.extend([check_id, v])
        return torch.tensor([edges_source, edges_target], dtype=torch.long)

    def generate_batch(self, batch_size, ebno_db):
        no = ebnodb2no(ebno_db, num_bits_per_symbol=1, coderate=self.k/self.n)
        no_val = float(no) # Convert from TF to float
        sigma = np.sqrt(no_val / 2.0)
        
        # 1. Generate bits and encode using Sionna
        source = BinarySource()
        bits = source([batch_size, self.k])
        codewords_tf = self.encoder(bits)
        
        # 2. Convert directly to PyTorch
        codewords = torch.tensor(codewords_tf.numpy(), dtype=torch.float32)
        
        # 3. Pure PyTorch BPSK AWGN Channel
        symbols = 1.0 - 2.0 * codewords
        noise = sigma * torch.randn_like(symbols)
        y = symbols + noise
        
        # 4. Calculate LLRs
        llrs = (2.0 / (sigma**2)) * y
        
        return llrs, codewords, y, no_val

class ExternalLDPCEnvironment:
    def __init__(self, filepath):
        self.filepath = filepath
        self.pcm = self._load_matrix(filepath)
        self.num_check_nodes, self.num_var_nodes = self.pcm.shape
        self.n = self.num_var_nodes
        self.k = self.n - self.num_check_nodes 
        self.edge_index = self._build_tanner_graph()

    def _load_matrix(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.npy':
            return np.load(filepath)
        elif ext == '.alist':
            return self._parse_alist(filepath)
        else:
            raise ValueError("Unsupported file format. Please use .npy or .alist")

    def _parse_alist(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                data.append([int(x) for x in parts])
        n, m = data[0][0], data[0][1]
        pcm = np.zeros((m, n), dtype=np.float32)
        for i in range(n):
            check_nodes = [c - 1 for c in data[4 + i] if c > 0]
            for c in check_nodes:
                pcm[c, i] = 1.0
        return pcm

    def _build_tanner_graph(self):
        edges_source, edges_target = [], []
        for c in range(self.num_check_nodes):
            for v in range(self.num_var_nodes):
                if self.pcm[c, v] == 1.0:
                    check_id = c + self.num_var_nodes
                    edges_source.extend([v, check_id])
                    edges_target.extend([check_id, v])
        return torch.tensor([edges_source, edges_target], dtype=torch.long)

    def generate_batch(self, batch_size, ebno_db):
        no = ebnodb2no(ebno_db, num_bits_per_symbol=1, coderate=self.k/self.n)
        no_val = float(no)
        sigma = np.sqrt(no_val / 2.0)
        
        # Pure PyTorch BPSK Channel on all-zero codeword
        codewords = torch.zeros((batch_size, self.n), dtype=torch.float32)
        symbols = 1.0 - 2.0 * codewords # Maps 0 to +1.0
        
        noise = sigma * torch.randn_like(symbols)
        y = symbols + noise
        
        llrs = (2.0 / (sigma**2)) * y
        
        return llrs, codewords, y, no_val


