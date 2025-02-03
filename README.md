# ML4G
# Graph-Based Recommendation Models

This repository contains two PyTorch implementations for graph-based recommendation:

- **GraphSAGE Recommendation Model**  
  Uses the GraphSAGE approach to learn embeddings for users and items from a user–item bipartite graph.

- **Vectorized PinSAGE Recommendation Model**  
  Uses a vectorized version of PinSAGE with offline neighbor precomputation (via random walks with restart) to generate node embeddings.

## Requirements

- Python 3.6+
- PyTorch
- NumPy, Pandas, scikit-learn

Install dependencies with:
```bash
pip install torch numpy pandas scikit-learn

