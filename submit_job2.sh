#!/bin/bash
# Submit Job 2: Refinement Phase (Rounds 10-18)
# Usage: ./submit_job2.sh

SSH_HOST="team02@129.212.178.168"
SSH_PORT="32605"
JOB_NAME="job2_medium_20min"
SSH_PASSWORD="_)yY@jg4<wnht*fJMKlKxEx9CilopM1X"

echo "═══════════════════════════════════════════════════════════"
echo "  SUBMITTING JOB 2: REFINEMENT PHASE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Learning Rate: 0.005 (reduced from 0.01)"
echo "  - Rounds:        9 (Rounds 10-18)"
echo "  - Local Epochs:  3"
echo "  - Job Name:      $JOB_NAME"
echo ""
echo "This job will resume from Job 1's checkpoint"
echo "Connecting to cluster and submitting job..."
echo ""

sshpass -p "$SSH_PASSWORD" ssh -p $SSH_PORT -o StrictHostKeyChecking=no $SSH_HOST << 'ENDSSH'
set -e
cd ~/coldstart
source ~/hackathon-venv/bin/activate

echo "Current directory: $(pwd)"
echo "Git branch: $(git branch --show-current)"
echo ""

JOB_NAME="job2_medium_20min"
LR="0.005"
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
    
else
    echo "❌ Error: Could not find submit-job.sh or sbatch"
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
    echo "  ✓ JOB 2 SUBMITTED SUCCESSFULLY!"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "  1. Check status: ssh -p $SSH_PORT $SSH_HOST 'squeue -u team02'"
    echo "  2. Monitor: ssh -p $SSH_PORT -t $SSH_HOST 'cd ~/coldstart && ./monitor_20min.sh'"
    echo "  3. View logs: ssh -p $SSH_PORT $SSH_HOST 'tail -f ~/logs/${JOB_NAME}*.out'"
    echo ""
    echo "Expected completion: ~20 minutes"
    echo "Expected AUROC: 0.7720 → 0.8050 (+0.033)"
    echo ""
else
    echo ""
    echo "❌ Job submission failed"
    exit 1
fi

