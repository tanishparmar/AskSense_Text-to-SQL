"""
Sentence Transformer Model Module

Handles the all-MiniLM-L6-v2 model for semantic similarity and text embeddings.
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SentenceModelPredictor:
    """Handles the all-MiniLM-L6-v2 sentence transformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            # Patch the SentenceTransformer initialization to avoid meta tensor error
            original_to = torch.nn.Module.to

            def patched_to(self, *args, **kwargs):
                try:
                    return original_to(self, *args, **kwargs)
                except NotImplementedError as e:
                    if "Cannot copy out of meta tensor" in str(e):
                        logger.warning("Meta tensor error detected, using to_empty() instead")
                        # Extract device argument (positional or keyword) or default to 'cpu'
                        if args and isinstance(args[0], (str, torch.device)):
                            device = args[0]
                        elif 'device' in kwargs:
                            device = kwargs['device']
                        else:
                            device = 'cpu'
                        # Call to_empty with required keyword-only argument
                        return self.to_empty(device=device)
                    raise

            # Apply the patch
            torch.nn.Module.to = patched_to

            # Now initialize the model on CPU
            self.model = SentenceTransformer(self.model_name, device='cpu')

            # Restore original method
            torch.nn.Module.to = original_to

            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            numpy array of embeddings
        """
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        try:
            embeddings = self.model.encode([text1, text2])
            similarity = torch.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0)
            )
            return similarity.item()
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (text, similarity_score) tuples
        """
        try:
            all_texts = [query] + candidates
            embeddings = self.model.encode(all_texts)

            query_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]

            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                sim = torch.cosine_similarity(
                    torch.tensor(query_embedding).unsqueeze(0),
                    torch.tensor(candidate_embedding).unsqueeze(0)
                ).item()
                similarities.append((candidates[i], sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Error finding similar texts: {str(e)}")
            return []

    def enhance_sql_generation(self, question: str, schema_info: str, db_id: str) -> str:
        """
        Use sentence similarity to enhance SQL generation.

        Args:
            question: Natural language question
            schema_info: Database schema information
            db_id: Database identifier

        Returns:
            Enhanced prompt for SQL generation
        """
        try:
            prompt_variations = [
                f"Question: {question}\nSchema: {schema_info}",
                f"Given the schema: {schema_info}\nAnswer this question: {question}",
                f"Database: {db_id}\nSchema: {schema_info}\nQuery: {question}",
                f"Using the database schema: {schema_info}\nGenerate SQL for: {question}"
            ]

            embeddings = self.model.encode(prompt_variations)
            question_emb = self.model.encode([question])[0]

            sims = []
            for idx, var_emb in enumerate(embeddings):
                sim = torch.cosine_similarity(
                    torch.tensor(question_emb).unsqueeze(0),
                    torch.tensor(var_emb).unsqueeze(0)
                ).item()
                sims.append((prompt_variations[idx], sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            return sims[0][0]
        except Exception as e:
            logger.error(f"Error enhancing SQL generation: {str(e)}")
            return f"Question: {question}\nSchema: {schema_info}"

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'model_type': 'SentenceTransformer',
            'loaded': self.model is not None,
            'max_seq_length': self.model.max_seq_length if self.model else None
        }
