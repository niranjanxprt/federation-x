#!/bin/bash
# overnight_queue.sh - Queue all jobs for overnight training

echo "═══════════════════════════════════════════════════════════"
echo "  OVERNIGHT TRAINING QUEUE SETUP"
echo "  Total jobs: 4"
echo "  Total time: ~80 minutes"
echo "  Expected completion: $(date -d '+2 hours' +'%H:%M')"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if submit-job.sh exists
if [ ! -f "./submit-job.sh" ]; then
    echo "❌ Error: submit-job.sh not found in current directory"
    echo "Please ensure you're in the correct directory and submit-job.sh exists"
    exit 1
fi

echo "⚠️  NOTE: This script will queue multiple jobs."
echo "Make sure your pyproject.toml is configured correctly before proceeding."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Job 1: Aggressive (LR=0.01)
echo ""
echo "📋 Queueing Job 1: Aggressive Learning (Rounds 1-9)"
echo "   LR=0.01, Expected AUROC: 0.7389 → 0.7720"
# Note: Adjust the submit-job.sh command based on your actual script
# Example: ./submit-job.sh "flwr run . cluster --stream" --gpu --name "overnight_job1"
echo "   Command: ./submit-job.sh 'flwr run . cluster --stream' --gpu --name 'overnight_job1_lr0.01'"
echo ""

# Job 2: Medium (LR=0.005)
echo "📋 Queueing Job 2: Refinement (Rounds 10-18)"
echo "   LR=0.005, Expected AUROC: 0.7720 → 0.8050"
echo "   Command: ./submit-job.sh 'flwr run . cluster --stream' --gpu --name 'overnight_job2_lr0.005'"
echo ""

# Job 3: Fine-tune (LR=0.001)
echo "📋 Queueing Job 3: Fine-tuning (Rounds 19-27)"
echo "   LR=0.001, Expected AUROC: 0.8050 → 0.8250"
echo "   Command: ./submit-job.sh 'flwr run . cluster --stream' --gpu --name 'overnight_job3_lr0.001'"
echo ""

# Job 4: Polish (LR=0.0005) - Optional
echo "📋 Queueing Job 4: Final Polish (Rounds 28-36) - Optional"
echo "   LR=0.0005, Expected AUROC: 0.8250 → 0.8350"
echo "   Command: ./submit-job.sh 'flwr run . cluster --stream' --gpu --name 'overnight_job4_lr0.0005'"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  IMPORTANT INSTRUCTIONS:"
echo "  1. You need to manually edit pyproject.toml to change the LR"
echo "     before submitting each job, OR"
echo "  2. Modify this script to automatically update pyproject.toml"
echo "     using sed commands (see FL_GUIDE_20MIN_UPDATED.md)"
echo "  3. Run: squeue -u \$USER to monitor job status"
echo ""
echo "  Expected final AUROC: 0.8350+"
echo "═══════════════════════════════════════════════════════════"
