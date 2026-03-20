import matplotlib.pyplot as plt
import os

class ResultPlotter:
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    def plot_metrics(self, results, title_prefix="Decoding Performance", filename_prefix="results"):
        snrs = results["SNR"]
        styles = {
            "BP": {"marker": 'o', "linestyle": '--', "color": 'black', "label": 'Standard Min-Sum BP'},
            "Neural_BP": {"marker": 's', "linestyle": '-.', "color": 'blue', "label": 'Standard Neural BP (2018)'},
            "GAT_No_Cycle": {"marker": '^', "linestyle": ':', "color": 'orange', "label": 'GAT (No Cycle Mask)'},
            "CAGAT": {"marker": '*', "linestyle": '-', "color": 'red', "markersize": 10, "label": 'CA-GAT-NMS (Ours)'}
        }
        metrics_to_plot = [("BER", "Bit Error Rate (BER)"), ("FER", "Frame Error Rate (FER)")]
        
        for metric_key, y_label in metrics_to_plot:
            plt.figure(figsize=(8, 6))
            if f"BP_{metric_key}" in results: plt.semilogy(snrs, results[f"BP_{metric_key}"], **styles["BP"])
            if f"Neural_BP_{metric_key}" in results: plt.semilogy(snrs, results[f"Neural_BP_{metric_key}"], **styles["Neural_BP"])
            if f"GAT_No_Cycle_{metric_key}" in results: plt.semilogy(snrs, results[f"GAT_No_Cycle_{metric_key}"], **styles["GAT_No_Cycle"])
            if f"CAGAT_{metric_key}" in results: plt.semilogy(snrs, results[f"CAGAT_{metric_key}"], **styles["CAGAT"])
            
            plt.title(f"{title_prefix} - {metric_key}", fontsize=14, fontweight='bold')
            plt.xlabel("Eb/N0 (dB)", fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(fontsize=11)
            
            base_save_path = os.path.join(self.save_dir, f"{filename_prefix}_{metric_key.lower()}")
            plt.savefig(f"{base_save_path}.pdf", format='pdf', bbox_inches='tight')
            plt.savefig(f"{base_save_path}.png", format='png', dpi=300, bbox_inches='tight')
            
            print(f"{metric_key} Graphs saved successfully (.pdf and .png) to: {self.save_dir}")
            plt.show()
