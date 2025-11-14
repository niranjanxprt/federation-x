#!/bin/bash
# monitor_20min.sh - Enhanced monitoring dashboard for 20-min jobs

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo "  FEDERATED LEARNING DASHBOARD (20-min jobs) - $(date +'%H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo ""

    # Active jobs with more detail
    echo "🔄 ACTIVE JOBS:"
    squeue -u $USER --format="%-12i %-30j %-8T %-10M %-10l %-10P" 2>/dev/null | head -n 6 || echo "  No jobs running"
    echo ""

    # Job queue status
    QUEUED=$(squeue -u $USER -t PENDING 2>/dev/null | wc -l)
    RUNNING=$(squeue -u $USER -t RUNNING 2>/dev/null | wc -l)
    echo "📋 QUEUE STATUS:"
    echo "  Running: $RUNNING / 4 parallel slots"
    echo "  Queued:  $((QUEUED-1)) / 100 max queue"
    echo ""

    # Top models by AUROC
    echo "🏆 TOP 5 MODELS:"
    ls -t ~/models/*.pt 2>/dev/null | head -n 5 | while read model; do
        auroc=$(basename "$model" | grep -oP 'auroc\K[0-9]+' || echo "????")
        round=$(basename "$model" | grep -oP 'round\K[0-9]+' || echo "?")
        size=$(du -h "$model" | cut -f1)
        printf "  Round %-3s: AUROC 0.%-4s  Size: %-6s  %s\n" \
               "$round" "$auroc" "$size" "$(basename $model | cut -c1-50)"
    done || echo "  No models yet"
    echo ""

    # Checkpoint status
    echo "💾 CHECKPOINTS:"
    if [ -d /home/team02/checkpoints ]; then
        ls -lh /home/team02/checkpoints/*.pt 2>/dev/null | \
            awk '{printf "  %-50s %8s  %s %s\n", $9, $5, $6, $7}' | \
            head -n 5 || echo "  No checkpoints yet"
    else
        echo "  Directory not found"
    fi
    echo ""

    # Disk usage
    echo "💿 DISK USAGE:"
    df -h ~ | awk 'NR==2 {printf "  Home:     %s / %s (%s used)\n", $3, $2, $5}'
    du -sh ~/models 2>/dev/null | awk '{printf "  Models:   %s\n", $1}' || echo "  Models:   0B"
    du -sh /home/team02/checkpoints 2>/dev/null | awk '{printf "  Checkpts: %s\n", $1}' || echo "  Checkpts: 0B"
    echo ""

    # GPU status (if available)
    if command -v nvidia-smi &> /dev/null; then
        echo "🖥️  GPU STATUS:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null | \
            awk -F', ' '{printf "  GPU %s: %-20s | %3s%% util | %5s/%5s MB\n", $1, $2, $3, $4, $5}'
        echo ""
    fi

    # Next recommended action
    echo "🎯 NEXT ACTION:"
    num_jobs=$(squeue -u $USER 2>/dev/null | wc -l)
    if [ $((num_jobs-1)) -gt 0 ]; then
        echo "  → Jobs running, monitoring in progress..."
    else
        echo "  → No jobs running. Check logs or submit next job."
    fi
    echo ""

    echo "───────────────────────────────────────────────────────────────────────────"
    echo "  W&B: https://wandb.ai/niranjanxprt-niranjanxprt/flower-federated-learning"
    echo "  Refreshing every 15s... (Ctrl+C to exit)"
    echo "═══════════════════════════════════════════════════════════════════════════"

    sleep 15
done
