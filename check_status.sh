#!/bin/bash
# check_status.sh - Check training status on cluster

SSH_HOST="team02@129.212.178.168"
SSH_PORT="32605"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  CLUSTER STATUS CHECK${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

ssh -p $SSH_PORT $SSH_HOST << 'ENDSSH'

echo "🔄 ACTIVE JOBS:"
echo "───────────────────────────────────────────────────────────"
squeue -u team02 --format="%-12i %-30j %-8T %-10M %-10l" 2>/dev/null || echo "No jobs running or squeue unavailable"
echo ""

echo "🏆 TOP 5 RECENT MODELS:"
echo "───────────────────────────────────────────────────────────"
ls -lht ~/models/*.pt 2>/dev/null | head -n 5 | while read -r line; do
    filename=$(echo "$line" | awk '{print $9}')
    size=$(echo "$line" | awk '{print $5}')
    date=$(echo "$line" | awk '{print $6, $7, $8}')
    basename=$(basename "$filename")

    # Try to extract AUROC
    auroc=$(echo "$basename" | grep -oP 'auroc\K[0-9]+' || echo "????")
    round=$(echo "$basename" | grep -oP 'round\K[0-9]+' || echo "?")

    printf "  Round %-3s | AUROC 0.%-4s | %8s | %s\n" "$round" "$auroc" "$size" "$date"
done || echo "No models found"
echo ""

echo "💾 CHECKPOINTS:"
echo "───────────────────────────────────────────────────────────"
if [ -d "/home/team02/checkpoints" ]; then
    ls -lht /home/team02/checkpoints/*.pt 2>/dev/null | head -n 5 | while read -r line; do
        filename=$(echo "$line" | awk '{print $9}')
        size=$(echo "$line" | awk '{print $5}')
        printf "  %-50s %8s\n" "$(basename $filename)" "$size"
    done || echo "No checkpoints found"
else
    echo "Checkpoint directory not found (will be created on first run)"
fi
echo ""

echo "📋 RECENT LOGS (last 3):"
echo "───────────────────────────────────────────────────────────"
ls -lt ~/logs/*.out 2>/dev/null | head -n 3 | while read -r line; do
    filename=$(echo "$line" | awk '{print $9}')
    size=$(echo "$line" | awk '{print $5}')
    printf "  %-60s %8s\n" "$(basename $filename)" "$size"
done || echo "No logs found"
echo ""

echo "💿 DISK USAGE:"
echo "───────────────────────────────────────────────────────────"
df -h ~ 2>/dev/null | awk 'NR==2 {printf "  Home:     %s / %s (%s used)\n", $3, $2, $5}' || echo "  Unable to check disk usage"
du -sh ~/models 2>/dev/null | awk '{printf "  Models:   %s\n", $1}' || echo "  Models:   0B"
du -sh ~/coldstart 2>/dev/null | awk '{printf "  Code:     %s\n", $1}' || echo "  Code:     0B"
echo ""

echo "📊 GIT STATUS:"
echo "───────────────────────────────────────────────────────────"
cd ~/coldstart 2>/dev/null && {
    echo "  Branch: $(git branch --show-current)"
    echo "  Commit: $(git log -1 --oneline)"
} || echo "  Unable to check git status"

ENDSSH

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Status check complete${NC}"
echo ""
