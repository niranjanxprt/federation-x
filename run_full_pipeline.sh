#!/bin/bash
# Complete pipeline to deploy code, run training, and evaluate
# Usage: ./run_full_pipeline.sh [job_type] [job_name]
# job_type: 1=aggressive, 2=refinement, 3=fine-tuning (default: 1)
# job_name: custom name for the job (default: auto-generated)

set -e  # Exit on error

SSH_CMD="ssh -p 32605 team02@129.212.178.168"
BRANCH="claude/test-01TPDEivdvegb7uMnXnhx9U7"
JOB_TYPE=${1:-1}
JOB_NAME=${2:-"job_$(date +%Y%m%d_%H%M%S)"}

echo "═══════════════════════════════════════════════════════════"
echo "  FEDERATED LEARNING PIPELINE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Job Type: $JOB_TYPE"
echo "Job Name: $JOB_NAME"
echo "Branch: $BRANCH"
echo ""

# Step 1: Deploy latest code
echo "─────────────────────────────────────────────────────────"
echo "Step 1/5: Deploying latest code to cluster..."
echo "─────────────────────────────────────────────────────────"
$SSH_CMD "cd ~/coldstart && git checkout $BRANCH && git pull origin $BRANCH"
echo "✓ Code deployed"
echo ""

# Step 2: Run preflight checks
echo "─────────────────────────────────────────────────────────"
echo "Step 2/5: Running preflight checks..."
echo "─────────────────────────────────────────────────────────"
$SSH_CMD "cd ~/coldstart && ./preflight_20min.sh" || {
    echo "⚠️  Preflight checks failed. Continuing anyway..."
}
echo ""

# Step 3: Submit training job
echo "─────────────────────────────────────────────────────────"
echo "Step 3/5: Submitting training job..."
echo "─────────────────────────────────────────────────────────"
$SSH_CMD "cd ~/coldstart && ./quick_submit.sh $JOB_TYPE $JOB_NAME"
echo ""

# Step 4: Wait for job to complete
echo "─────────────────────────────────────────────────────────"
echo "Step 4/5: Monitoring job (checking every 30 seconds)..."
echo "─────────────────────────────────────────────────────────"
while true; do
    QUEUE_STATUS=$($SSH_CMD "squeue -u team02 --format='%.10i %.30j %.8T' | grep -v 'JOBID' || true")
    
    if [ -z "$QUEUE_STATUS" ]; then
        echo "✓ All jobs completed!"
        break
    fi
    
    echo "$(date +%H:%M:%S) - Jobs running:"
    echo "$QUEUE_STATUS"
    sleep 30
done
echo ""

# Step 5: Run evaluation
echo "─────────────────────────────────────────────────────────"
echo "Step 5/5: Running evaluation on best model..."
echo "─────────────────────────────────────────────────────────"
$SSH_CMD "cd ~/coldstart && ./submit-job.sh 'python evaluate.py' --gpu --name eval_$JOB_NAME"
echo ""

# Wait for evaluation to complete
echo "Waiting for evaluation to complete..."
sleep 10
while true; do
    EVAL_STATUS=$($SSH_CMD "squeue -u team02 --format='%.30j %.8T' | grep 'eval_' || true")
    
    if [ -z "$EVAL_STATUS" ]; then
        echo "✓ Evaluation completed!"
        break
    fi
    
    echo "$(date +%H:%M:%S) - Evaluation status: $EVAL_STATUS"
    sleep 15
done
echo ""

# Step 6: Display results
echo "─────────────────────────────────────────────────────────"
echo "FINAL RESULTS"
echo "─────────────────────────────────────────────────────────"
$SSH_CMD "cd ~/coldstart && tail -50 slurm-*.out | grep -A 20 'EVALUATION' || echo 'Evaluation output not found yet'"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  - Review full logs: ssh to cluster and check ~/logs/"
echo "  - View models: ls -lh /home/team02/models/"
echo "  - Check status: ./check_status.sh"
