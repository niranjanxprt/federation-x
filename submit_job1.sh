#!/bin/bash
# Non-interactive job submission script for Job 1
# Run this script: bash submit_job1.sh

SSH_HOST="team02@129.212.178.168"
SSH_PORT="32605"
JOB_NAME="job1_aggressive_20min"

echo "═══════════════════════════════════════════════════════════"
echo "  SUBMITTING JOB 1: AGGRESSIVE LEARNING"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Learning Rate: 0.01"
echo "  - Rounds:        9"
echo "  - Local Epochs:  3"
echo "  - Job Name:      $JOB_NAME"
echo ""
echo "Connecting to cluster and submitting job..."
echo ""

# SSH and submit job (will prompt for password if keys not set up)
ssh -p $SSH_PORT $SSH_HOST << 'ENDSSH'
cd ~/coldstart
source ~/hackathon-venv/bin/activate

echo "Current directory: $(pwd)"
echo "Git branch: $(git branch --show-current)"
echo ""

# Check if submit-job.sh exists
if [ ! -f "submit-job.sh" ]; then
    echo "Error: submit-job.sh not found"
    exit 1
fi

# Submit Job 1: Aggressive Learning
JOB_NAME="job1_aggressive_20min"
LR="0.01"
ROUNDS="9"
EPOCHS="3"

echo "Submitting job: $JOB_NAME"
echo "Runtime config: num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=$LR"
echo ""

./submit-job.sh "flwr run . cluster-gpu --stream --run-config \"num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=$LR\"" --gpu --name "$JOB_NAME"

echo ""
echo "✓ Job submitted successfully!"
echo ""
echo "Monitor with:"
echo "  squeue -u team02"
echo "  tail -f ~/logs/${JOB_NAME}*.out"
echo "  ./monitor_20min.sh"
ENDSSH

if [ $? -eq 0 ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  ✓ JOB SUBMITTED SUCCESSFULLY!"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "  1. Check status: ssh -p $SSH_PORT $SSH_HOST 'squeue -u team02'"
    echo "  2. Monitor: ssh -p $SSH_PORT -t $SSH_HOST 'cd ~/coldstart && ./monitor_20min.sh'"
    echo "  3. View logs: ssh -p $SSH_PORT $SSH_HOST 'tail -f ~/logs/${JOB_NAME}*.out'"
    echo ""
    echo "Expected completion: ~20 minutes"
    echo "Expected AUROC: 0.7389 → 0.7720 (+0.033)"
    echo ""
else
    echo ""
    echo "❌ Job submission failed"
    echo ""
    echo "Possible issues:"
    echo "  1. SSH authentication failed - set up SSH keys:"
    echo "     ./setup_ssh_keys.sh"
    echo "  2. Connection timeout - check network connection"
    echo "  3. Cluster unavailable - check with: ssh -p $SSH_PORT $SSH_HOST 'squeue -u team02'"
    exit 1
fi

