from typing import List, Optional
import faiss
import numpy as np

chunks: Optional[List[str]] = None
faiss_index: Optional[faiss.Index] = None