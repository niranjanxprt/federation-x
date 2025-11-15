#!/bin/bash
# quick_submit.sh - Quick job submission to cluster

set -e

SSH_HOST="team02@129.212.178.168"
SSH_PORT="32605"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  QUICK JOB SUBMISSION${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Parse arguments
JOB_TYPE=${1:-""}
JOB_NAME=${2:-"fl_job_$(date +%Y%m%d_%H%M%S)"}

if [ -z "$JOB_TYPE" ]; then
    echo "Usage: ./quick_submit.sh [1|2|3] [optional_job_name]"
    echo ""
    echo "Job types:"
    echo "  1 - Job 1: Aggressive (LR=0.01, Rounds 1-9)"
    echo "  2 - Job 2: Medium (LR=0.005, Rounds 10-18)"
    echo "  3 - Job 3: Fine-tune (LR=0.001, Rounds 19-27)"
    echo ""
    echo "Example: ./quick_submit.sh 1 my_experiment"
    exit 1
fi

case $JOB_TYPE in
    1)
        LR="0.01"
        DESCRIPTION="Aggressive Learning"
        ;;
    2)
        LR="0.005"
        DESCRIPTION="Refinement"
        ;;
    3)
        LR="0.001"
        DESCRIPTION="Fine-tuning"
        ;;
    *)
        echo "Error: Invalid job type. Use 1, 2, or 3"
        exit 1
        ;;
esac

ROUNDS="9"
EPOCHS="3"

echo -e "${GREEN}Job Type:${NC} $DESCRIPTION"
echo -e "${GREEN}Configuration:${NC}"
echo "   - Learning Rate:      $LR"
echo "   - Rounds:             $ROUNDS"
echo "   - Local Epochs:       $EPOCHS"
echo -e "${GREEN}Job Name:${NC} $JOB_NAME"
echo ""
echo -e "${BLUE}Note: Config will be passed via --run-config (pyproject.toml stays unchanged)${NC}"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

echo ""
echo "Submitting job to cluster..."

ssh -p $SSH_PORT $SSH_HOST << ENDSSH
cd ~/coldstart
source ~/hackathon-venv/bin/activate

echo "Current directory: \$(pwd)"
echo "Git branch: \$(git branch --show-current)"
echo ""

# Check if submit-job.sh exists
if [ ! -f "submit-job.sh" ]; then
    echo "Error: submit-job.sh not found"
    exit 1
fi

# Submit the job with runtime config overrides
echo "Submitting job: $JOB_NAME"
echo "Runtime config: num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=$LR"
echo ""

./submit-job.sh "flwr run . cluster --stream --run-config \"num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=$LR\"" --gpu --name "$JOB_NAME"

echo ""
echo "✓ Job submitted successfully!"
echo ""
echo "Monitor with:"
echo "  squeue -u team02"
echo "  tail -f ~/logs/${JOB_NAME}*.out"

ENDSSH

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Job submission complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Check status: ssh -p $SSH_PORT $SSH_HOST 'squeue -u team02'"
    echo "  2. Monitor: ssh -p $SSH_PORT -t $SSH_HOST 'cd ~/coldstart && ./monitor_20min.sh'"
    echo "  3. View logs: ssh -p $SSH_PORT $SSH_HOST 'tail -f ~/logs/${JOB_NAME}*.out'"
else
    echo -e "${RED}✗ Job submission failed${NC}"
    exit 1
fi
