import torch
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
from clip import clip
import pandas as pd

from src.utils.utils import create_vocab, get_clip_embeddings

class SimilarityMatrix:
    """
    A class to compute and store a similarity matrix for sentence embeddings.
    """
    
    def __init__(self, conf: dict, sentences_df: pd.DataFrame, split: str, clip_model: object) -> None:
        """
        Intantiate the SimilarityMatrix, loading or computing the similarity matrix.
            
        Parameters
        ----------
        conf : dict
            Configuration dictionary containing paths, batch sizes and other settings.
        sentences_df : pandas.DataFrame
            DataFrame containing sentence data with index 'sent_id'.
        split : str
            Dataset split ('train', 'val', 'test') to differentiate saved matrices.
        clip_model : Any
            The CLIP model instance for computing sentence embeddings.
        """

        if split == "test" and conf["dt_full_test_set"] == True:
            self.sm_path = f"{conf['dt_ess_sm_path']}/{split}_sim_matrix.npz"
        else:
            self.sm_path = f"{conf['dt_ess_sm_path']}/{split}_sim_matrix_limited_s{conf['dt_samples_limit']}.npz" if conf["dt_samples_limit"] is not None else f"{conf['dt_ess_sm_path']}/{split}_sim_matrix.npz"
        self.sm_path = os.path.abspath(self.sm_path)
        os.makedirs(conf["dt_ess_sm_path"], exist_ok = True)
        self.sentences = sentences_df
        self.k = conf['dt_ess_sm_k']
        self.device = conf["device"]
        self.sim_matrix = None
        self.sent2topk = {}  # Dictionary to store top-k most similar sentences
        self.batch_size = conf['dt_ess_sm_batch_size']
        self.clip_model = clip_model
        
        self.vocab, self.reverse_vocab = create_vocab(self.sentences.index.tolist())  # sent_id as index
        
        print(f"{self.sm_path}, exists -> {os.path.exists(self.sm_path)}")
        if os.path.exists(self.sm_path):
            print("\nLoading precomputed similarity matrix.")
            with np.load(self.sm_path) as data:
                self.sim_matrix = data['arr_0']
        else:
            print("\nComputing similarity matrix.")
            self.compute_similarity_matrix()

        
        # Calculate top-k for computed similarity matrix
        print("Computing top-k most similar sentences for each sentence.\n")
        for sent_idx in range(len(self.sim_matrix)):
            self.sent2topk[sent_idx] = np.argpartition(-self.sim_matrix[sent_idx], self.k + 1)[1: self.k + 1].tolist()
            
    def compute_similarity_matrix(self) -> None:
        """
        Compute the pairwise cosine similarity between all sentences in the dataset.

        Workflow:
        - Sentences are processed in batches.
        - Similarities are computed using batched matrix multiplications on GPU.
        - The resulting similarity matrix is saved to a file for future uses.
        """
        num_sentences = len(self.sentences)
        all_sentences = self.sentences['sent'].tolist()

        # Encode all sentences in batches
        all_embeddings = []
        for i in tqdm(range(0, num_sentences, self.batch_size), desc="Batch encoding sentences"):
            batch_sentences = all_sentences[i:i + self.batch_size]
            batch_embeddings = get_clip_embeddings(clip.tokenize(batch_sentences).to(self.device), self.clip_model).to(self.device)
            all_embeddings.append(batch_embeddings)

        # Concatenate all sentence embeddings into a single tensor
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Normalize embeddings
        all_embeddings = F.normalize(all_embeddings, p=2, dim=-1)

        # Initialize an empty similarity matrix
        self.sim_matrix = np.zeros((num_sentences, num_sentences))

        # Compute cosine similarity in batches
        for i in tqdm(range(0, num_sentences, self.batch_size), desc="Batched similarity computation"):
            # Take a batch of embeddings for comparison
            batch_embeddings = all_embeddings[i:i + self.batch_size]

            # Compute the cosine similarity of this batch with the entire set
            with torch.no_grad():
                sim_matrix_gpu = torch.matmul(batch_embeddings, all_embeddings.T)

            # Transfer the result to CPU and save it in the appropriate part of the similarity matrix
            self.sim_matrix[i:i + self.batch_size, :] = sim_matrix_gpu.cpu().numpy()

        # Save the similarity matrix to a file
        print("Similarity matrix computed. Compressing ...")
        np.savez_compressed(self.sm_path, self.sim_matrix)
        print("Similarity matrix compressed and saved.")

    def get_top_k_similar(self, sent_id: str) -> list[str]:
        """
        Retrieve the top-k most similar sentences for a given sentence ID.

        Parameters
        ----------
        sent_id : int
            The unique identifier of the sentence.

        Returns
        -------
        list[int]
            A list of top-k most similar sentence IDs.
        """
        # Convert sent_id to matrix index
        sent_idx = self.vocab[sent_id]
        top_k_indices = self.sent2topk[sent_idx]

        # Map indices back to original sentence IDs
        return [self.reverse_vocab[idx] for idx in top_k_indices]