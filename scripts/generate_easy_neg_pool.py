"""Generate easy negative pool by sampling from non-hit negatives."""
import os
import random
import argparse


def reservoir_sample_easy(merged_fasta: str, hard_seqs: set, k: int, seed: int = 42):
    """Reservoir sample k easy negatives from merged FASTA, excluding hard pool."""
    rng = random.Random(seed)
    reservoir = []
    count = 0
    seq_buffer = []

    with open(merged_fasta, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # Flush previous sequence
                if seq_buffer:
                    seq = ''.join(seq_buffer)
                    if seq not in hard_seqs:
                        count += 1
                        if len(reservoir) < k:
                            reservoir.append(seq)
                        else:
                            j = rng.randint(0, count - 1)
                            if j < k:
                                reservoir[j] = seq
                    seq_buffer = []
            else:
                seq_buffer.append(line)

        # Flush last sequence
        if seq_buffer:
            seq = ''.join(seq_buffer)
            if seq not in hard_seqs:
                count += 1
                if len(reservoir) < k:
                    reservoir.append(seq)
                else:
                    j = rng.randint(0, count - 1)
                    if j < k:
                        reservoir[j] = seq

    return reservoir, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hard', type=str,
                        default='data/processed/train_neg_pool.csv')
    parser.add_argument('--merged', type=str,
                        default='data/processed/temp/neg_merged.fasta')
    parser.add_argument('--output', type=str,
                        default='data/processed/train_neg_pool_easy.csv')
    parser.add_argument('--target', type=int, default=142101,
                        help='Target number of easy negatives (default: match hard pool size)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Loading hard pool: {args.hard}")
    hard_seqs = set()
    with open(args.hard, 'r') as f:
        header = f.readline()
        for line in f:
            seq = line.strip()
            if seq:
                hard_seqs.add(seq)
    print(f"  Hard pool size: {len(hard_seqs):,}")

    print(f"Reservoir sampling from: {args.merged}")
    print(f"  Target easy negatives: {args.target:,}")
    reservoir, total_easy = reservoir_sample_easy(
        args.merged, hard_seqs, args.target, args.seed
    )
    print(f"  Total non-hit negatives scanned: {total_easy:,}")
    print(f"  Sampled easy negatives: {len(reservoir):,}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('sequence\n')
        for seq in reservoir:
            f.write(f'{seq}\n')
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
