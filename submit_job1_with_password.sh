#!/bin/bash
# Submit Job 1 with password authentication
# Usage: ./submit_job1_with_password.sh

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

# Check if sshpass is available
if command -v sshpass &> /dev/null; then
    echo "Using sshpass for password authentication..."
    sshpass -p "$SSH_PASSWORD" ssh -p $SSH_PORT -o StrictHostKeyChecking=no $SSH_HOST << 'ENDSSH'
cd ~/coldstart
source ~/hackathon-venv/bin/activate

echo "Current directory: $(pwd)"
echo "Git branch: $(git branch --show-current)"
echo ""

if [ ! -f "submit-job.sh" ]; then
    echo "Error: submit-job.sh not found"
    exit 1
fi

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
ENDSSH
    EXIT_CODE=$?
elif command -v expect &> /dev/null; then
    echo "Using expect for password authentication..."
    expect << EOF
spawn ssh -p $SSH_PORT -o StrictHostKeyChecking=no $SSH_HOST
expect {
    "password:" {
        send "$SSH_PASSWORD\r"
        expect "$ "
        send "cd ~/coldstart && source ~/hackathon-venv/bin/activate\r"
        expect "$ "
        send "echo 'Submitting job...'\r"
        expect "$ "
        send "./submit-job.sh 'flwr run . cluster-gpu --stream --run-config \"num-server-rounds=9 local-epochs=3 lr=0.01\"' --gpu --name job1_aggressive_20min\r"
        expect "$ "
        send "exit\r"
        expect eof
    }
    timeout {
        exit 1
    }
}
EOF
    EXIT_CODE=$?
else
    echo "Error: Neither sshpass nor expect is installed."
    echo "Please install one of them:"
    echo "  macOS: brew install hudochenkov/sshpass/sshpass"
    echo "  Linux: sudo apt-get install sshpass"
    exit 1
fi

if [ $EXIT_CODE -eq 0 ]; then
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
    echo "⚠️  Note: Consider setting up SSH keys for passwordless access:"
    echo "   ./setup_ssh_keys.sh"
    echo ""
else
    echo ""
    echo "❌ Job submission failed"
    exit 1
fi

