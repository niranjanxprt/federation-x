#!/bin/bash
# preflight_20min.sh - Comprehensive pre-flight check for 20-min jobs

echo "═══════════════════════════════════════════════════════════"
echo "  PRE-FLIGHT CHECKLIST (20-Minute Configuration)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# File checks
echo "📁 FILE CHECKS:"
test -f cold_start_hackathon/losses.py && echo "  ✓ losses.py exists" || echo "  ✗ CREATE losses.py"
grep -q "FocalLoss" cold_start_hackathon/task.py && echo "  ✓ Focal Loss imported" || echo "  ✗ ADD import to task.py"
grep -q "IMAGENET1K_V1" cold_start_hackathon/task.py && echo "  ✓ Pre-trained weights" || echo "  ✗ UPDATE to pre-trained"
grep -q "FedProx" cold_start_hackathon/server_app.py && echo "  ✓ FedProx strategy" || echo "  ✗ CHANGE to FedProx"
echo ""

# Configuration checks
echo "⚙️  CONFIGURATION:"
ROUNDS=$(grep "num-server-rounds" pyproject.toml | grep -o '[0-9]*')
EPOCHS=$(grep "local-epochs" pyproject.toml | grep -o '[0-9]*')
LR=$(grep "^lr" pyproject.toml | grep -o '[0-9.]*')
echo "  Rounds:       $ROUNDS"
echo "  Local epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo ""

# Infrastructure checks
echo "🔧 INFRASTRUCTURE:"
test -d /home/team02/checkpoints && echo "  ✓ Checkpoint dir exists" || echo "  ⚠️  CREATE: mkdir -p /home/team02/checkpoints"
squeue -u $USER &>/dev/null && echo "  ✓ Cluster access" || echo "  ⚠️  Check cluster connection"

QUEUE_COUNT=$(($(squeue -u $USER 2>/dev/null | wc -l) - 1))
if [ $QUEUE_COUNT -lt 95 ]; then
    echo "  ✓ Queue space: $QUEUE_COUNT / 100"
else
    echo "  ⚠️  Queue almost full: $QUEUE_COUNT / 100"
fi
echo ""

# Disk space
echo "💾 DISK SPACE:"
DISK_USED=$(df -h ~ | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USED -lt 80 ]; then
    echo "  ✓ Disk usage: ${DISK_USED}%"
else
    echo "  ⚠️  High disk usage: ${DISK_USED}%"
fi
echo ""

# Python imports test
echo "🐍 PYTHON CHECKS:"
python -c "from cold_start_hackathon.losses import FocalLoss; print('  ✓ FocalLoss import OK')" 2>&1 || echo "  ✗ FocalLoss import failed"
python -c "from cold_start_hackathon.task import Net; Net(); print('  ✓ Model init OK')" 2>&1 || echo "  ✗ Model init failed"
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════"
echo "  RECOMMENDATION:"
if [ -f cold_start_hackathon/losses.py ]; then
    echo "  ✅ Core files ready"
    echo ""
    echo "  Next: Submit job using your submit-job.sh script"
else
    echo "  ⚠️  SOME CHECKS FAILED - Review errors above"
fi
echo "═══════════════════════════════════════════════════════════"
