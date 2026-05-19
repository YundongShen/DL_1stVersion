#!/bin/bash
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --job-name=ebd_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

set -euo pipefail

PROJECT=/proj/test_extender/ICLR/EditBoundaryDetection
cd "$PROJECT"
source gpu_env/bin/activate

VLLM_NODE=${VLLM_NODE:-$(cat logs/vllm_node.txt 2>/dev/null || echo "localhost")}
VLLM_URL="http://${VLLM_NODE}:8000/v1"
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "vLLM: $VLLM_URL"

# Wait for vLLM to be ready (retries for up to 15 minutes)
echo "Waiting for vLLM at $VLLM_URL ..."
for i in $(seq 1 30); do
    python -c "
from openai import OpenAI, APIConnectionError
import os, sys
try:
    client = OpenAI(base_url='$VLLM_URL', api_key=os.environ.get('OPENAI_API_KEY', 'EMPTY'))
    resp = client.chat.completions.create(
        model='Qwen2.5-Coder-32B-Instruct',
        messages=[{'role': 'user', 'content': 'Say hello in one word.'}],
        max_tokens=10,
    )
    print('LLM test OK:', resp.choices[0].message.content)
    sys.exit(0)
except Exception as e:
    print(f'Not ready ({e})', file=sys.stderr)
    sys.exit(1)
" && break
    echo "Attempt $i/30 failed, retrying in 30s..."
    sleep 30
done

# Run 5 sympy instances to test end-to-end
python scripts/collect_pairs.py \
    --mode training \
    --input  data/raw/swebench_instances.jsonl \
    --output data/cache/test_pairs.jsonl \
    --repo-filter sympy/sympy \
    --max 20 \
    --workers 4 \
    --timeout 120 \
    --filter-prefix sympy \
    --repos-dir "${TMPDIR}/repos" \
    --llm-provider openai \
    --llm-api-base "$VLLM_URL" \
    --llm-model Qwen2.5-Coder-32B-Instruct \
    --llm-max-tokens 4096

echo "Done: $(date)"
echo "Test pairs written: $(wc -l < data/cache/test_pairs.jsonl 2>/dev/null || echo 0)"
