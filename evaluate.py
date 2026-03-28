import torch
import torch.nn as nn
import numpy as np
from sionna.phy.fec.ldpc.decoding import LDPCBPDecoder

class Evaluator:
    def __init__(self, model, gat_no_cycle_model, baseline_neural_model, environment, cycle_mask):
        self.model = model
        self.gat_no_cycle_model = gat_no_cycle_model
        self.baseline_neural_model = baseline_neural_model
        
        self.env = environment
        self.cycle_mask = cycle_mask
        self.zero_mask = torch.zeros_like(cycle_mask)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.optimizer_no_cycle = torch.optim.Adam(self.gat_no_cycle_model.parameters(), lr=0.005)
        self.baseline_optimizer = torch.optim.Adam(self.baseline_neural_model.parameters(), lr=0.005)
        self.criterion = nn.BCEWithLogitsLoss()
        
        if hasattr(self.env, 'encoder'):
            self.baseline_decoder = LDPCBPDecoder(encoder=self.env.encoder, num_iter=10)
        else:
            self.baseline_decoder = LDPCBPDecoder(pcm=self.env.pcm, num_iter=10)

    def train_step(self, batch_size, ebno_db):
        self.model.train()
        self.gat_no_cycle_model.train()
        self.baseline_neural_model.train()
        
        self.optimizer.zero_grad()
        self.optimizer_no_cycle.zero_grad()
        self.baseline_optimizer.zero_grad()
        
        llrs, true_codewords, _, _ = self.env.generate_batch(batch_size, ebno_db)
        true_float = true_codewords.float()
        
        iteration_outputs = self.model(llrs, self.env.edge_index, self.cycle_mask)
        loss_cagat = sum((t + 1) / len(iteration_outputs) * self.criterion(out, true_float) for t, out in enumerate(iteration_outputs))
        loss_cagat.backward()
        self.optimizer.step()
        
        no_cycle_outputs = self.gat_no_cycle_model(llrs, self.env.edge_index, self.zero_mask)
        loss_no_cycle = sum((t + 1) / len(no_cycle_outputs) * self.criterion(out, true_float) for t, out in enumerate(no_cycle_outputs))
        loss_no_cycle.backward()
        self.optimizer_no_cycle.step()
        
        base_outputs = self.baseline_neural_model(llrs, self.env.edge_index)
        loss_base = sum((t + 1) / len(base_outputs) * self.criterion(out, true_float) for t, out in enumerate(base_outputs))
        loss_base.backward()
        self.baseline_optimizer.step()
        
        return loss_cagat.item()

    def evaluate_baselines(self, test_batches, batch_size, snr_range):
        self.model.eval()
        self.gat_no_cycle_model.eval()
        self.baseline_neural_model.eval()
        
        results = {"SNR": [], "BP_BER": [], "Neural_BP_BER": [], "GAT_No_Cycle_BER": [], "CAGAT_BER": [],
                   "BP_FER": [], "Neural_BP_FER": [], "GAT_No_Cycle_FER": [], "CAGAT_FER": []}
        
        with torch.no_grad():
            for snr in snr_range:
                total_bits = 0
                total_frames = test_batches * batch_size
                errs = {"bp_bit": 0, "bp_frame": 0, "nbp_bit": 0, "nbp_frame": 0, 
                        "gat_nc_bit": 0, "gat_nc_frame": 0, "cagat_bit": 0, "cagat_frame": 0}
                
                for _ in range(test_batches):
                    llrs, true_codewords, _, _ = self.env.generate_batch(batch_size, snr)
                    true_info_bits = true_codewords[:, :self.env.k]
                    
                    # ---> THE FIX: Convert PyTorch LLRs to numpy for Sionna BP Decoder <---
                    bp_dec_tf = self.baseline_decoder(llrs.numpy())
                    bp_dec = torch.tensor(np.array(bp_dec_tf))[:, :self.env.k]
                    
                    nbp_dec = (torch.sigmoid(self.baseline_neural_model(llrs, self.env.edge_index)[-1]) > 0.5).float()[:, :self.env.k]
                    gat_nc_dec = (torch.sigmoid(self.gat_no_cycle_model(llrs, self.env.edge_index, self.zero_mask)[-1]) > 0.5).float()[:, :self.env.k]
                    cagat_dec = (torch.sigmoid(self.model(llrs, self.env.edge_index, self.cycle_mask)[-1]) > 0.5).float()[:, :self.env.k]
                    
                    errs["bp_bit"] += torch.sum(bp_dec != true_info_bits).item()
                    errs["nbp_bit"] += torch.sum(nbp_dec != true_info_bits).item()
                    errs["gat_nc_bit"] += torch.sum(gat_nc_dec != true_info_bits).item()
                    errs["cagat_bit"] += torch.sum(cagat_dec != true_info_bits).item()
                    
                    errs["bp_frame"] += torch.sum(torch.any(bp_dec != true_info_bits, dim=1)).item()
                    errs["nbp_frame"] += torch.sum(torch.any(nbp_dec != true_info_bits, dim=1)).item()
                    errs["gat_nc_frame"] += torch.sum(torch.any(gat_nc_dec != true_info_bits, dim=1)).item()
                    errs["cagat_frame"] += torch.sum(torch.any(cagat_dec != true_info_bits, dim=1)).item()
                    
                    total_bits += batch_size * self.env.k
                
                results["SNR"].append(snr)
                results["BP_BER"].append(errs["bp_bit"] / total_bits)
                results["Neural_BP_BER"].append(errs["nbp_bit"] / total_bits)
                results["GAT_No_Cycle_BER"].append(errs["gat_nc_bit"] / total_bits)
                results["CAGAT_BER"].append(errs["cagat_bit"] / total_bits)
                
                results["BP_FER"].append(errs["bp_frame"] / total_frames)
                results["Neural_BP_FER"].append(errs["nbp_frame"] / total_frames)
                results["GAT_No_Cycle_FER"].append(errs["gat_nc_frame"] / total_frames)
                results["CAGAT_FER"].append(errs["cagat_frame"] / total_frames)
                
                print(f"SNR: {snr} dB | BER -> BP: {results['BP_BER'][-1]:.4f} | GAT(No Mask): {results['GAT_No_Cycle_BER'][-1]:.4f} | CAGAT: {results['CAGAT_BER'][-1]:.4f}")
                print(f"          | FER -> BP: {results['BP_FER'][-1]:.4f} | GAT(No Mask): {results['GAT_No_Cycle_FER'][-1]:.4f} | CAGAT: {results['CAGAT_FER'][-1]:.4f}")
                
        return results
