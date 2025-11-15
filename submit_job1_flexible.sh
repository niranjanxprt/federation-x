#!/bin/bash
# Flexible job submission script - tries multiple methods
# Usage: ./submit_job1_flexible.sh

SSH_HOST="team02@129.212.178.168"
SSH_PORT="32605"
JOB_NAME="job1_aggressive_20min"
SSH_PASSWORD="_)yY@jg4<wnht*fJMKlKxEx9CilopM1X"

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

sshpass -p "$SSH_PASSWORD" ssh -p $SSH_PORT -o StrictHostKeyChecking=no $SSH_HOST << 'ENDSSH'
set -e
cd ~/coldstart
source ~/hackathon-venv/bin/activate

echo "Current directory: $(pwd)"
echo "Git branch: $(git branch --show-current)"
echo ""

JOB_NAME="job1_aggressive_20min"
LR="0.01"
ROUNDS="9"
EPOCHS="3"

# Try method 1: submit-job.sh (if it exists)
if [ -f "submit-job.sh" ]; then
    echo "✓ Found submit-job.sh - using it to submit job"
    echo ""
    ./submit-job.sh "flwr run . cluster-gpu --stream --run-config \"num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=$LR\"" --gpu --name "$JOB_NAME"
    
# Try method 2: sbatch with a temporary script
elif command -v sbatch &> /dev/null; then
    echo "⚠ submit-job.sh not found, using sbatch directly"
    echo ""
    
    # Create a temporary SLURM script
    SLURM_SCRIPT="/tmp/job_${JOB_NAME}_$$.sh"
    cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=~/logs/${JOB_NAME}_%j.out
#SBATCH --error=~/logs/${JOB_NAME}_%j.err

cd ~/coldstart
source ~/hackathon-venv/bin/activate

export JOB_NAME="${JOB_NAME}"

flwr run . cluster-gpu --stream --run-config "num-server-rounds=${ROUNDS} local-epochs=${EPOCHS} lr=${LR}"
EOF
    
    mkdir -p ~/logs
    chmod +x "$SLURM_SCRIPT"
    echo "Submitting job via sbatch..."
    sbatch "$SLURM_SCRIPT"
    echo "✓ Job submitted via sbatch"
    rm -f "$SLURM_SCRIPT"
    
# Try method 3: Run directly (if not on cluster)
elif command -v flwr &> /dev/null; then
    echo "⚠ No SLURM detected, attempting direct execution (may timeout)"
    echo "⚠ This is not recommended - job should be submitted via SLURM"
    echo ""
    export JOB_NAME="$JOB_NAME"
    timeout 1200 flwr run . cluster-gpu --stream --run-config "num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=$LR" || echo "Direct execution completed or timed out"
    
else
    echo "❌ Error: Could not find submit-job.sh, sbatch, or flwr command"
    echo "Available commands:"
    which sbatch || echo "  sbatch: not found"
    which flwr || echo "  flwr: not found"
    echo ""
    echo "Please check:"
    echo "  1. Are you on the cluster node?"
    echo "  2. Is submit-job.sh in the current directory?"
    echo "  3. Is SLURM configured?"
    exit 1
fi

echo ""
echo "✓ Job submission completed!"
echo ""
echo "Monitor with:"
echo "  squeue -u team02"
echo "  tail -f ~/logs/${JOB_NAME}*.out"
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
    echo "Please check:"
    echo "  1. Network connection"
    echo "  2. SSH credentials"
    echo "  3. Cluster availability"
    exit 1
fi

