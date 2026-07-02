"""
ARG-Expert usage examples.

Demonstrates model creation, prediction, training, evaluation, and export.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from model.config import Config, set_seed
from model.data import ARGDataset, collate_fn
from model.architecture import ARGTransformer
from model.trainer import ARGTrainer
from model.evaluator import ARGEvaluator
from torch.utils.data import DataLoader

set_seed(42)


def example_1_basic_model_creation():
    """Example 1: Create a basic model."""
    print("="*60)
    print("Example 1: Basic Model Creation")
    print("="*60)

    config = Config()
    print(f"Configuration: {config}")

    model = ARGTransformer(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model created successfully!")

    return model, config


def example_2_single_prediction():
    """Example 2: Single-sequence prediction."""
    print("\n" + "="*60)
    print("Example 2: Single-Sequence Prediction")
    print("="*60)

    # Example sequence (TEM-1 beta-lactamase fragment)
    test_sequence = (
        "MKTLLILTLVVVTIVCLDLGYTLMSNDNQIETISSEDLLEYGSNGSLSFSGRNFTTRDQV"
        "SGEIYQPETFIIKYPVDYKSSGVENPYNTPIMFKSNEYNQYNPDYPSENPFSVSIQMNQ"
        "GNVKSLQNGSYSFQISDLGYKISGVDSQNSIKDPTLQKKVTWEVNQGYTYNGAYNTES"
        "YTFSVNGTDNAYFETKPNQVAVIGTGTNGKTSVSHYIAQALNAGYKTVGVIGTLGYGK"
        "NSEVKSNTTPESTTLAQKLAYTAGLNENVTIHQDNLYGSKSGLKTAKKIVDLQNAQNI"
        "KATKIGYTLDSLVNQKKIAVLGTSNLSYQHNVYTQAADYLGGLQLPQNAELDSAIYQN"
        "QNQ"
    )

    print(f"Test sequence length: {len(test_sequence)} aa")
    print(f"Sequence: {test_sequence[:50]}...")

    # Prediction requires a trained model:
    # from inference import ARGPredictor
    # predictor = ARGPredictor('./output/best_model_warmup.pt')
    # result = predictor.predict(test_sequence)
    # print(f"\nPrediction result:")
    # print(f"  Is ARG: {result['is_arg']}")
    # print(f"  ARG probability: {result['arg_probability']:.4f}")
    # if result['is_arg']:
    #     print(f"  Predicted category: {result['predicted_category']}")
    #     print(f"  Confidence: {result['confidence']:.4f}")

    print("\nNote: requires a trained model checkpoint for actual prediction")


def example_3_batch_prediction():
    """Example 3: Batch prediction."""
    print("\n" + "="*60)
    print("Example 3: Batch Prediction")
    print("="*60)

    sequences = [
        "MKTLLILTLVVVTIVCLDLGYTLMSNDNQIETISSEDLLEYGSNGSLSFSGRNFTTRDQV",
        "MKRILLIVLLFLATLVALAADQPVTQDNQLISNENITLQEFKKGASVDSLQAQQEWLNIL",
        "MKKSLVLAVSLVLLVCSAAVAETVSFNATQKITRVEQKIELISNQIKEISKQMEQFSQY",
    ]

    print(f"Batch prediction for {len(sequences)} sequences")

    # Batch prediction requires a trained model:
    # results = predictor.predict(sequences, batch_size=32)

    print("\nNote: requires a trained model checkpoint for actual prediction")


def example_4_custom_training():
    """Example 4: Custom training pipeline."""
    print("\n" + "="*60)
    print("Example 4: Custom Training Pipeline")
    print("="*60)

    config = Config()
    config.warmup_epochs = 3

    print("Custom configuration:")
    print(f"  Warmup epochs: {config.warmup_epochs}")
    print(f"  Warmup batch size: {config.warmup_batch_size}")

    model = ARGTransformer(config)
    trainer = ARGTrainer(model, config)

    print("\nTrainer created successfully!")
    print("Note: requires prepared data to begin training")

    # Example training flow (requires actual data):
    # train_dataset = ARGDataset(train_sequences, train_labels, train_categories)
    # val_dataset = ARGDataset(val_sequences, val_labels, val_categories)
    # train_loader = DataLoader(train_dataset, batch_size=config.warmup_batch_size,
    #                           shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=config.finetune_batch_size,
    #                         shuffle=False, collate_fn=collate_fn)
    # history = trainer.train(train_loader, val_loader, stage_name="warmup")


def example_5_attention_visualization():
    """Example 5: Attention visualization."""
    print("\n" + "="*60)
    print("Example 5: Attention Visualization")
    print("="*60)

    sequence = (
        "MKTLLILTLVVVTIVCLDLGYTLMSNDNQIETISSEDLLEYGSNGSLSFSGRNFTTRDQV"
        "SGEIYQPETFIIKYPVDYKSSGVENPYNTPIMFKSNEYNQYNPDYPSENPFSVSIQMNQ"
    )

    print(f"Sequence length: {len(sequence)} aa")

    # Visualization requires a trained model:
    # from inference import ARGPredictor
    # predictor = ARGPredictor('./output/best_model_warmup.pt')
    # attention = predictor.visualize_attention(
    #     sequence=sequence,
    #     save_path='attention_heatmap.png'
    # )

    print("\nNote: requires a trained model checkpoint for visualization")


def example_6_model_evaluation():
    """Example 6: Model evaluation."""
    print("\n" + "="*60)
    print("Example 6: Model Evaluation")
    print("="*60)

    config = Config()
    model = ARGTransformer(config)

    evaluator = ARGEvaluator(model, config)

    print("Evaluator created successfully!")

    # Evaluation requires actual data:
    # results = evaluator.evaluate_comprehensive(test_loader, save_results=True)
    # print("\nEvaluation results:")
    # print(f"  Binary accuracy: {results['binary']['accuracy']:.4f}")
    # print(f"  Binary F1: {results['binary']['f1']:.4f}")
    # print(f"  Binary AUROC: {results['binary']['auroc']:.4f}")
    # print(f"  Multiclass accuracy: {results['multiclass']['accuracy']:.4f}")
    # print(f"  Multiclass macro F1: {results['multiclass']['macro_f1']:.4f}")

    print("\nNote: requires prepared test data for evaluation")


def example_7_export_model():
    """Example 7: Model export."""
    print("\n" + "="*60)
    print("Example 7: Model Export")
    print("="*60)

    config = Config()
    model = ARGTransformer(config)
    model.eval()

    example_input = torch.randint(0, 21, (1, config.max_seq_length))

    # Export as TorchScript
    try:
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save("arg_transformer_traced.pt")
        print("Model exported as TorchScript: arg_transformer_traced.pt")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    # Export as ONNX
    try:
        torch.onnx.export(
            model,
            example_input,
            "arg_transformer.onnx",
            input_names=['input'],
            output_names=['binary_pred', 'multiclass_pred'],
            dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}}
        )
        print("Model exported as ONNX: arg_transformer.onnx")
    except Exception as e:
        print(f"ONNX export failed: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" "*25 + "ARG-Expert Usage Examples")
    print("="*80 + "\n")

    example_1_basic_model_creation()
    example_2_single_prediction()
    example_3_batch_prediction()
    example_4_custom_training()
    example_5_attention_visualization()
    example_6_model_evaluation()
    example_7_export_model()

    print("\n" + "="*80)
    print(" "*30 + "All examples complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
