#!/bin/bash
# show_config.sh - Show current pyproject.toml configuration on cluster

SSH_HOST="team02@129.212.178.168"
SSH_PORT="32605"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  CURRENT CONFIGURATION (pyproject.toml)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

ssh -p $SSH_PORT $SSH_HOST << 'ENDSSH'
cd ~/coldstart

if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found"
    exit 1
fi

echo "📋 Current FL Configuration:"
echo "───────────────────────────────────────────────────────────"

# Extract values
ROUNDS=$(grep "num-server-rounds" pyproject.toml | grep -oP '= \K[0-9]+' || echo "NOT FOUND")
EPOCHS=$(grep "local-epochs" pyproject.toml | grep -oP '= \K[0-9]+' || echo "NOT FOUND")
LR=$(grep "^lr " pyproject.toml | grep -oP '= \K[0-9.]+' || echo "NOT FOUND")
IMAGE_SIZE=$(grep "image-size" pyproject.toml | grep -oP '= \K[0-9]+' || echo "NOT FOUND")

printf "  %-20s %s\n" "num-server-rounds:" "$ROUNDS"
printf "  %-20s %s\n" "local-epochs:" "$EPOCHS"
printf "  %-20s %s\n" "lr:" "$LR"
printf "  %-20s %s\n" "image-size:" "$IMAGE_SIZE"

echo ""
echo "🎯 Recommended Values by Job:"
echo "───────────────────────────────────────────────────────────"
echo "  Job 1 (Aggressive):  lr = 0.01   | 9 rounds | 3 epochs"
echo "  Job 2 (Refinement):  lr = 0.005  | 9 rounds | 3 epochs"
echo "  Job 3 (Fine-tune):   lr = 0.001  | 9 rounds | 3 epochs"

echo ""
echo "📝 Current Config Matches:"
echo "───────────────────────────────────────────────────────────"

# Determine which job this matches
if [ "$LR" = "0.01" ] && [ "$ROUNDS" = "9" ] && [ "$EPOCHS" = "3" ]; then
    echo "  ✅ Job 1 (Aggressive Learning)"
elif [ "$LR" = "0.005" ] && [ "$ROUNDS" = "9" ] && [ "$EPOCHS" = "3" ]; then
    echo "  ✅ Job 2 (Refinement)"
elif [ "$LR" = "0.001" ] && [ "$ROUNDS" = "9" ] && [ "$EPOCHS" = "3" ]; then
    echo "  ✅ Job 3 (Fine-tuning)"
else
    echo "  ⚠️  Custom configuration"
    if [ "$ROUNDS" != "9" ]; then
        echo "     WARNING: num-server-rounds should be 9 for 20-min jobs"
    fi
    if [ "$EPOCHS" != "3" ]; then
        echo "     WARNING: local-epochs should be 3 for best results"
    fi
    if [ "$LR" != "0.01" ] && [ "$LR" != "0.005" ] && [ "$LR" != "0.001" ]; then
        echo "     WARNING: Recommended LR values are 0.01, 0.005, or 0.001"
    fi
fi

echo ""
echo "📂 Full config section:"
echo "───────────────────────────────────────────────────────────"
grep -A 5 "\[tool.flwr.app.config\]" pyproject.toml

ENDSSH

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "⚠️  NOTE: pyproject.toml is FIXED and cannot be edited"
echo ""
echo "To use different config, pass via --run-config flag:"
echo "  flwr run . cluster --stream --run-config \"num-server-rounds=9 local-epochs=3 lr=0.01\""
echo ""
echo "Or use the automation scripts:"
echo "  ./quick_submit.sh 1    # Job 1 (LR=0.01)"
echo "  ./quick_submit.sh 2    # Job 2 (LR=0.005)"
echo "  ./quick_submit.sh 3    # Job 3 (LR=0.001)"
echo ""
