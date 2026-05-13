import gensim
import torch
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from munkres import Munkres


def verify_installations():
    print("🚀 开始验证库安装情况...\n")

    try:
        # 1. NumPy & Pandas
        print(f"[OK] NumPy 版本: {np.__version__}")
        print(f"[OK] Pandas 版本: {pd.__version__}")

        # 2. PyTorch (检查 CPU/GPU 可用性)
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        x = torch.rand(2, 3)
        print(f"[OK] PyTorch 版本: {torch.__version__} | 运行设备: {device}")

        # 3. Gensim (NLP)
        print(f"[OK] Gensim 版本: {gensim.__version__}")

        # 4. Scikit-learn (ML)
        model = LinearRegression()
        print(f"[OK] Scikit-learn 版本: {sklearn.__version__}")

        # 5. NetworkX (图论)
        G = nx.Graph()
        G.add_edge(1, 2)
        print(f"[OK] NetworkX 版本: {nx.__version__} | 测试边: {G.edges}")

        # 6. Munkres (匈牙利算法)
        m = Munkres()
        print(f"[OK] Munkres 已就绪 (用于指派问题优化)")

        print("\n✅ 所有库均已安装成功并能正常运行！")

    except ImportError as e:
        print(f"\n❌ 发现缺失的库: {e}")
    except Exception as e:
        print(f"\n⚠️ 运行测试时出错: {e}")

import gensim
from gensim.models.word2vec import FAST_VERSION
print(FAST_VERSION)
if __name__ == "__main__":
    verify_installations()
print("All packages installed successfully!")