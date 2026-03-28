from environment import LDPCEnvironment, ExternalLDPCEnvironment
from cycle_detector import CycleDetector
from models import NeuralDecoder, Standard_NeuralDecoder
from evaluate import Evaluator
from plotter import ResultPlotter

if __name__ == "__main__":
    print("1. Initializing Environment (5G LDPC N=200, K=100)...")
    #env = LDPCEnvironment(k=100, n=200)
    # env = ExternalLDPCEnvironment("mackay_96_48.alist") # For RPTU database files
    env =  ExternalLDPCEnvironment("BCH_63_27.alist")
    total_nodes = env.num_var_nodes + env.num_check_nodes
    
    print("2. Running Offline Algorithmic Pre-processing (Cycle Detection)...")
    detector = CycleDetector(env.edge_index, total_nodes)
    cycle_mask = detector.extract_cycle_mask()
    print(f"   Found {int(cycle_mask.sum().item())} edges participating in short cycles.")
    
    print("3. Initializing Models & Evaluator...")
    decoder_model = NeuralDecoder(num_nodes=total_nodes, pcm=env.pcm, num_iterations=10)
    gat_no_cycle_model = NeuralDecoder(num_nodes=total_nodes, pcm=env.pcm, num_iterations=10)
    baseline_neural = Standard_NeuralDecoder(num_nodes=total_nodes, pcm=env.pcm, num_iterations=10)
    
    evaluator = Evaluator(decoder_model, gat_no_cycle_model, baseline_neural, env, cycle_mask)
    
    print("4. Training Models (10 Epochs)...")
    for epoch in range(10):
        loss = evaluator.train_step(batch_size=64, ebno_db=2.0)
        print(f"   Epoch {epoch+1:02d} | Loss: {loss:.4f}")
        
    print("5. Evaluating Against Baselines...")
    snrs_to_test = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = evaluator.evaluate_baselines(test_batches=5, batch_size=100, snr_range=snrs_to_test)
    
    print("6. Generating Publication Graphs...")
    plotter = ResultPlotter()
    plotter.plot_metrics(results, title_prefix=f"Performance: N={env.n}, K={env.k}", filename_prefix="results")
