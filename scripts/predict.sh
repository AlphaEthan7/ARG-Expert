#!/bin/bash
# ARG-Expert Inference Script

# Default parameters
MODEL_PATH="./output/best_model_finetune.pt"
INPUT_FILE=""
INPUT_TYPE="fasta"
OUTPUT_FILE="predictions.json"
BATCH_SIZE=32
DEVICE=""

# Show help
show_help() {
    echo "Usage: ./predict.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE        Input file (FASTA or CSV) [required]"
    echo "  -t, --type TYPE         Input type: fasta or csv (default: fasta)"
    echo "  -m, --model PATH        Model checkpoint path (default: ./output/best_model_finetune.pt)"
    echo "  -o, --output FILE       Output file (default: predictions.json)"
    echo "  -b, --batch-size SIZE   Batch size (default: 32)"
    echo "  -d, --device DEVICE     Device: cuda or cpu (default: auto)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./predict.sh -i test.fasta -o results.json"
    echo "  ./predict.sh -i test.csv -t csv -m ./custom_model.pt -b 64"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -t|--type)
            INPUT_TYPE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file is required."
    show_help
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Build command
CMD="python inference.py \
    --model ${MODEL_PATH} \
    --input ${INPUT_FILE} \
    --input_type ${INPUT_TYPE} \
    --output ${OUTPUT_FILE} \
    --batch_size ${BATCH_SIZE}"

if [ -n "$DEVICE" ]; then
    CMD="${CMD} --device ${DEVICE}"
fi

echo "===================================================="
echo "ARG_Expert Prediction"
echo "===================================================="
echo "Model: ${MODEL_PATH}"
echo "Input: ${INPUT_FILE}"
echo "Type: ${INPUT_TYPE}"
echo "Output: ${OUTPUT_FILE}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Run prediction
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================="
    echo "Prediction completed successfully!"
    echo "===================================================="
    echo "Results saved to: ${OUTPUT_FILE}"
    
    # Show summary
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "Result summary:"
        python -c "
import json
with open('${OUTPUT_FILE}') as f:
    data = json.load(f)
    if isinstance(data, list):
        total = len(data)
        arg_count = sum(1 for r in data if r.get('is_arg', False))
        print(f'  Total sequences: {total}')
        print(f'  Predicted ARGs: {arg_count} ({arg_count/total*100:.1f}%)')
        print(f'  Predicted non-ARGs: {total-arg_count} ({(total-arg_count)/total*100:.1f}%)')
    else:
        print(f'  Is ARG: {data.get(\"is_arg\", False)}')
        print(f'  Probability: {data.get(\"arg_probability\", 0):.4f}')
        if data.get('is_arg'):
            print(f'  Category: {data.get(\"predicted_category\", \"N/A\")}')
"
    fi
else
    echo ""
    echo "===================================================="
    echo "Prediction failed!"
    echo "===================================================="
    exit 1
fi
