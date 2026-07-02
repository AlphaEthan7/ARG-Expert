"""
ARG-Expert inference module.

Predicts ARG vs. non-ARG and resistance category from protein sequences.
"""

import os
import argparse
import logging
from typing import List, Dict, Union
import json

import torch
import torch.nn.functional as F
import numpy as np
from Bio import SeqIO

from .config import Config, AMINO_ACIDS, AA_TO_IDX, set_seed
from .architecture import ARGTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ARGPredictor:
    """ARG prediction and classification from protein sequences."""

    def __init__(self, model_path: str, config: Config = None, device: str = None):
        """
        Args:
            model_path: path to model checkpoint (.pt)
            config: model configuration (auto-detected from checkpoint if None)
            device: compute device (auto-detected if None)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"Using device: {self.device}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if config is None:
            config = checkpoint.get('config', Config())
        self.config = config

        self.model = ARGTransformer(config)
        self.model.to(self.device)

        state_dict = checkpoint['model_state_dict']
        # Strip torch.compile _orig_mod. prefix if present
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        logger.info(f"Model loaded from {model_path}")

        self.category_names = self._load_category_names()

    def _load_category_names(self) -> Dict[int, str]:
        """14-class resistance category mapping."""
        category_names = {
            0: "beta-lactam",
            1: "multidrug",
            2: "MLS",
            3: "aminoglycoside",
            4: "peptide",
            5: "tetracycline",
            6: "phosphonic",
            7: "glycopeptide",
            8: "quinolone",
            9: "diaminopyrimidine",
            10: "other",
            11: "phenicol",
            12: "sulfonamide",
            13: "aminocoumarin"
        }
        return category_names

    def _preprocess_sequence(self, sequence: str) -> torch.Tensor:
        """Clean, tokenize, and pad a single protein sequence."""
        sequence = sequence.upper()
        sequence = ''.join([aa for aa in sequence if aa in AMINO_ACIDS])

        if len(sequence) > self.config.max_seq_length:
            sequence = sequence[:self.config.max_seq_length]

        tokens = [AA_TO_IDX.get(aa, 0) for aa in sequence]

        if len(tokens) < self.config.max_seq_length:
            tokens.extend([0] * (self.config.max_seq_length - len(tokens)))

        return torch.tensor([tokens], dtype=torch.long)

    @torch.inference_mode()
    def predict(
        self,
        sequence: Union[str, List[str]],
        return_attention: bool = False,
        batch_size: int = 32
    ) -> Union[Dict, List[Dict]]:
        """
        Predict ARG probability and resistance category.

        Args:
            sequence: single amino acid sequence or list of sequences
            return_attention: whether to include attention weights in output
            batch_size: inference batch size

        Returns:
            dict (single input) or list of dicts (multiple inputs)
        """
        single_input = isinstance(sequence, str)
        if single_input:
            sequences = [sequence]
        else:
            sequences = sequence

        results = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]

            batch_tokens = []
            for seq in batch_seqs:
                tokens = self._preprocess_sequence(seq)
                batch_tokens.append(tokens)

            batch_tokens = torch.cat(batch_tokens, dim=0).to(self.device)

            with torch.autocast(device_type='cuda', enabled=True):
                outputs = self.model(batch_tokens, return_attention=return_attention)

            binary_logits = outputs['binary_pred'].squeeze(-1)
            binary_probs = torch.sigmoid(binary_logits).cpu().numpy()
            multiclass_probs = F.softmax(outputs['multiclass_pred'], dim=1).cpu().numpy()

            if return_attention:
                attention_weights = outputs['attention_weights'].cpu().numpy()

            binary_threshold = getattr(self.config, 'binary_threshold', 0.5)

            for j, seq in enumerate(batch_seqs):
                result = {
                    'sequence': seq,
                    'is_arg': bool(binary_probs[j] > binary_threshold),
                    'arg_probability': float(binary_probs[j]),
                }

                if result['is_arg']:
                    # Exclude class 10 ("other") from argmax; use it only as
                    # a fallback when confidence is below threshold.
                    named_class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
                    named_probs = multiclass_probs[j][named_class_ids]
                    max_named_prob = float(named_probs.max())
                    predicted_class = int(named_class_ids[named_probs.argmax()])

                    fallback_threshold = getattr(self.config, 'multiclass_confidence_threshold', 0.3)

                    if max_named_prob < fallback_threshold:
                        result['predicted_category'] = "other/unknown"
                        result['predicted_category_id'] = -1
                        result['confidence'] = 1.0 - max_named_prob
                    else:
                        result['predicted_category'] = self.category_names.get(predicted_class, 'unknown')
                        result['predicted_category_id'] = predicted_class
                        result['confidence'] = max_named_prob

                        top3_indices = named_probs.argsort()[-3:][::-1]
                        top3_predictions = [
                            {
                                'category': self.category_names.get(int(named_class_ids[idx]), 'unknown'),
                                'category_id': int(named_class_ids[idx]),
                                'probability': float(named_probs[idx])
                            }
                            for idx in top3_indices
                        ]
                        result['top3_predictions'] = top3_predictions

                if return_attention:
                    result['attention_weights'] = attention_weights[j].tolist()

                results.append(result)

        if single_input:
            return results[0]
        return results

    def predict_from_fasta(
        self,
        fasta_path: str,
        output_path: str = None,
        return_attention: bool = False,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict from a FASTA file.

        Args:
            fasta_path: path to input FASTA file
            output_path: optional JSON output path for results
            return_attention: whether to include attention weights
            batch_size: inference batch size

        Returns:
            list of prediction dicts (one per sequence)
        """
        logger.info(f"Loading sequences from {fasta_path}")

        sequences = []
        headers = []

        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq))
            headers.append(record.id)

        logger.info(f"Loaded {len(sequences)} sequences")

        results = self.predict(sequences, return_attention=return_attention, batch_size=batch_size)

        for i, result in enumerate(results):
            result['header'] = headers[i]

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results

    def predict_from_csv(
        self,
        csv_path: str,
        sequence_column: str = 'sequence',
        output_path: str = None,
        return_attention: bool = False,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict from a CSV file.

        Args:
            csv_path: path to input CSV file
            sequence_column: name of the column containing protein sequences
            output_path: optional JSON output path for results
            return_attention: whether to include attention weights
            batch_size: inference batch size

        Returns:
            list of prediction dicts (one per row)
        """
        import pandas as pd

        logger.info(f"Loading sequences from {csv_path}")

        df = pd.read_csv(csv_path)
        sequences = df[sequence_column].tolist()

        logger.info(f"Loaded {len(sequences)} sequences")

        results = self.predict(sequences, return_attention=return_attention, batch_size=batch_size)

        for i, result in enumerate(results):
            for col in df.columns:
                if col != sequence_column:
                    result[col] = df.iloc[i][col]

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results


def main():
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(description='ARG-Expert Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input file (FASTA or CSV)')
    parser.add_argument('--input_type', type=str, choices=['fasta', 'csv'], default='fasta',
                        help='Input file type')
    parser.add_argument('--output', type=str, default='predictions.json', help='Output file path')
    parser.add_argument('--sequence_column', type=str, default='sequence',
                        help='Sequence column name for CSV input')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--return_attention', action='store_true',
                        help='Return attention weights')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    set_seed(42)

    predictor = ARGPredictor(args.model, device=args.device)

    if args.input_type == 'fasta':
        results = predictor.predict_from_fasta(
            args.input,
            output_path=args.output,
            return_attention=args.return_attention,
            batch_size=args.batch_size
        )
    else:
        results = predictor.predict_from_csv(
            args.input,
            sequence_column=args.sequence_column,
            output_path=args.output,
            return_attention=args.return_attention,
            batch_size=args.batch_size
        )

    total = len(results)
    arg_count = sum(1 for r in results if r['is_arg'])

    logger.info(f"\nPrediction Summary:")
    logger.info(f"  Total sequences: {total}")
    logger.info(f"  Predicted ARGs: {arg_count} ({arg_count/total*100:.2f}%)")
    logger.info(f"  Predicted non-ARGs: {total - arg_count} ({(total-arg_count)/total*100:.2f}%)")

    if arg_count > 0:
        from collections import Counter
        categories = [r['predicted_category'] for r in results if r['is_arg']]
        category_counts = Counter(categories)

        logger.info(f"\nPredicted ARG Categories:")
        for cat, count in category_counts.most_common():
            logger.info(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
