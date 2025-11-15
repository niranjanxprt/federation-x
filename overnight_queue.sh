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

echo "⚠️  NOTE: This script is for REFERENCE ONLY."
echo "It shows example commands but doesn't actually submit jobs."
echo "Use quick_submit.sh or deploy_and_test.sh for actual submissions."
echo ""
read -p "Press Enter to view examples or Ctrl+C to cancel..."

# Job configurations
ROUNDS="9"
EPOCHS="3"

# Job 1: Aggressive (LR=0.01)
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📋 Job 1: Aggressive Learning (Rounds 1-9)"
echo "═══════════════════════════════════════════════════════════"
echo "   LR=0.01, Expected AUROC: 0.7389 → 0.7720"
echo ""
echo "   Manual command:"
echo "   ./submit-job.sh \"flwr run . cluster-gpu --stream --run-config \\\"num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=0.01\\\"\" --gpu --name \"overnight_job1\""
echo ""
echo "   OR use quick_submit.sh:"
echo "   ./quick_submit.sh 1 overnight_job1"
echo ""

# Job 2: Medium (LR=0.005)
echo "═══════════════════════════════════════════════════════════"
echo "📋 Job 2: Refinement (Rounds 10-18)"
echo "═══════════════════════════════════════════════════════════"
echo "   LR=0.005, Expected AUROC: 0.7720 → 0.8050"
echo ""
echo "   Manual command:"
echo "   ./submit-job.sh \"flwr run . cluster-gpu --stream --run-config \\\"num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=0.005\\\"\" --gpu --name \"overnight_job2\""
echo ""
echo "   OR use quick_submit.sh:"
echo "   ./quick_submit.sh 2 overnight_job2"
echo ""

# Job 3: Fine-tune (LR=0.001)
echo "═══════════════════════════════════════════════════════════"
echo "📋 Job 3: Fine-tuning (Rounds 19-27)"
echo "═══════════════════════════════════════════════════════════"
echo "   LR=0.001, Expected AUROC: 0.8050 → 0.8250"
echo ""
echo "   Manual command:"
echo "   ./submit-job.sh \"flwr run . cluster-gpu --stream --run-config \\\"num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=0.001\\\"\" --gpu --name \"overnight_job3\""
echo ""
echo "   OR use quick_submit.sh:"
echo "   ./quick_submit.sh 3 overnight_job3"
echo ""

# Job 4: Polish (LR=0.0005) - Optional
echo "═══════════════════════════════════════════════════════════"
echo "📋 Job 4: Final Polish (Rounds 28-36) - Optional"
echo "═══════════════════════════════════════════════════════════"
echo "   LR=0.0005, Expected AUROC: 0.8250 → 0.8350"
echo ""
echo "   Manual command:"
echo "   ./submit-job.sh \"flwr run . cluster-gpu --stream --run-config \\\"num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=0.0005\\\"\" --gpu --name \"overnight_job4\""
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  RECOMMENDED WORKFLOW:"
echo "  Use quick_submit.sh for easy job submission:"
echo "    ./quick_submit.sh 1 job1"
echo "    (wait for completion)"
echo "    ./quick_submit.sh 2 job2"
echo "    (wait for completion)"
echo "    ./quick_submit.sh 3 job3"
echo ""
echo "  Monitor: ./check_status.sh"
echo "  Expected final AUROC: 0.8250+"
echo "═══════════════════════════════════════════════════════════"
