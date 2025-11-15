#!/bin/bash
# deploy_and_test.sh - Automated deployment and testing script for cluster

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
SSH_HOST="team02@129.212.178.168"
SSH_PORT="32605"
REMOTE_DIR="~/coldstart"
VENV_PATH="~/hackathon-venv"
BRANCH="claude/test-01TPDEivdvegb7uMnXnhx9U7"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

# ============================================================================
# Main Script
# ============================================================================

print_header "FEDERATED LEARNING - DEPLOY & TEST AUTOMATION"

echo "SSH Target: $SSH_HOST:$SSH_PORT"
echo "Remote Directory: $REMOTE_DIR"
echo "Branch: $BRANCH"
echo ""

# Check if SSH key is set up
print_info "Checking SSH connection..."
if ssh -p $SSH_PORT -o BatchMode=yes -o ConnectTimeout=5 $SSH_HOST echo "OK" 2>/dev/null; then
    print_success "SSH key authentication is set up"
    USE_KEY_AUTH=true
else
    print_warning "SSH key authentication not set up - you'll need to enter password multiple times"
    print_info "To avoid this, set up SSH keys with: ssh-copy-id -p $SSH_PORT $SSH_HOST"
    USE_KEY_AUTH=false
    echo ""
fi

# ============================================================================
# Step 1: Deploy Code to Cluster
# ============================================================================

print_header "STEP 1: DEPLOYING CODE TO CLUSTER"

print_info "Connecting to cluster and pulling latest changes..."

ssh -p $SSH_PORT $SSH_HOST << 'ENDSSH'
set -e

# Navigate to project directory
cd ~/coldstart

# Show current status
echo "Current directory: $(pwd)"
echo "Current branch: $(git branch --show-current)"

# Fetch latest changes
echo "Fetching latest changes..."
git fetch origin

# Check if branch exists
if git show-ref --verify --quiet refs/heads/$BRANCH; then
    echo "Branch exists locally, switching to it..."
    git checkout $BRANCH
    git pull origin $BRANCH
else
    echo "Branch doesn't exist locally, checking out from remote..."
    git checkout -b $BRANCH origin/$BRANCH
fi

echo "✓ Code deployment complete"
ENDSSH

if [ $? -eq 0 ]; then
    print_success "Code deployed successfully"
else
    print_error "Code deployment failed"
    exit 1
fi

# ============================================================================
# Step 2: Run Pre-flight Checks
# ============================================================================

print_header "STEP 2: RUNNING PRE-FLIGHT CHECKS"

ssh -p $SSH_PORT $SSH_HOST << 'ENDSSH'
set -e

cd ~/coldstart

# Make scripts executable
chmod +x preflight_20min.sh monitor_20min.sh overnight_queue.sh 2>/dev/null || true

# Run pre-flight checks if available
if [ -f "preflight_20min.sh" ]; then
    echo "Running pre-flight checks..."
    ./preflight_20min.sh
else
    echo "Pre-flight script not found, running basic checks..."

    # Check files
    echo "📁 FILE CHECKS:"
    [ -f "cold_start_hackathon/losses.py" ] && echo "  ✓ losses.py exists" || echo "  ✗ losses.py missing"
    [ -f "cold_start_hackathon/task.py" ] && echo "  ✓ task.py exists" || echo "  ✗ task.py missing"
    [ -f "cold_start_hackathon/server_app.py" ] && echo "  ✓ server_app.py exists" || echo "  ✗ server_app.py missing"

    # Check configuration
    echo ""
    echo "⚙️  CONFIGURATION:"
    grep "num-server-rounds" pyproject.toml || echo "  Config not found"

    # Check virtual environment
    echo ""
    echo "🐍 PYTHON ENVIRONMENT:"
    if [ -d "~/hackathon-venv" ]; then
        echo "  ✓ Virtual environment exists"
    else
        echo "  ✗ Virtual environment not found"
    fi
fi

ENDSSH

if [ $? -eq 0 ]; then
    print_success "Pre-flight checks complete"
else
    print_warning "Pre-flight checks completed with warnings"
fi

# ============================================================================
# Step 3: Job Configuration Selection
# ============================================================================

print_header "STEP 3: SELECT JOB CONFIGURATION"

print_info "Note: pyproject.toml is fixed - config will be overridden at runtime via --run-config"
echo ""

echo "Select job configuration:"
echo ""
echo "  1) Job 1 - Aggressive Learning (LR=0.01, 9 rounds, 3 epochs)"
echo "  2) Job 2 - Refinement (LR=0.005, 9 rounds, 3 epochs)"
echo "  3) Job 3 - Fine-tuning (LR=0.001, 9 rounds, 3 epochs)"
echo "  4) Custom configuration"
echo "  5) Skip job submission"
echo ""
read -p "Enter choice [1-5]: " CONFIG_CHOICE

case $CONFIG_CHOICE in
    1)
        LR="0.01"
        ROUNDS="9"
        EPOCHS="3"
        DESCRIPTION="Job 1 (Aggressive Learning)"
        ;;
    2)
        LR="0.005"
        ROUNDS="9"
        EPOCHS="3"
        DESCRIPTION="Job 2 (Refinement)"
        ;;
    3)
        LR="0.001"
        ROUNDS="9"
        EPOCHS="3"
        DESCRIPTION="Job 3 (Fine-tuning)"
        ;;
    4)
        read -p "Enter learning rate (e.g., 0.01): " LR
        read -p "Enter num-server-rounds (e.g., 9): " ROUNDS
        read -p "Enter local-epochs (e.g., 3): " EPOCHS
        DESCRIPTION="Custom Configuration"
        ;;
    5)
        print_info "Skipping job submission"
        print_header "DEPLOYMENT COMPLETE"
        print_success "Code deployed to cluster"
        print_success "Pre-flight checks completed"
        exit 0
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_info "Selected: $DESCRIPTION"
echo "  Learning Rate:      $LR"
echo "  Rounds:             $ROUNDS"
echo "  Local Epochs:       $EPOCHS"
echo ""

print_header "STEP 4: JOB SUBMISSION"

echo "What would you like to do?"
echo ""
echo "  1) Submit job with this config"
echo "  2) Check job status only"
echo "  3) Monitor training (real-time)"
echo "  4) Cancel"
echo ""
read -p "Enter choice [1-4]: " CHOICE

case $CHOICE in
    1)
        print_info "Submitting job..."

        # Get job name
        read -p "Enter job name (or press Enter for auto-generated): " JOB_NAME
        if [ -z "$JOB_NAME" ]; then
            JOB_NAME="fl_job_$(date +%Y%m%d_%H%M%S)"
        fi

        print_info "Submitting: $JOB_NAME"
        print_info "Runtime config: num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=$LR"

        ssh -p $SSH_PORT $SSH_HOST << ENDSSH
            cd ~/coldstart
            source ~/hackathon-venv/bin/activate

            echo "Submitting job: $JOB_NAME"
            echo "Runtime config overrides:"
            echo "  num-server-rounds=$ROUNDS"
            echo "  local-epochs=$EPOCHS"
            echo "  lr=$LR"
            echo ""

            # Submit job with runtime config
            if [ -f "submit-job.sh" ]; then
                ./submit-job.sh "flwr run . cluster-gpu --stream --run-config \"num-server-rounds=$ROUNDS local-epochs=$EPOCHS lr=$LR\"" --gpu --name "$JOB_NAME"
            else
                echo "Error: submit-job.sh not found"
                exit 1
            fi
ENDSSH
        ;;

    2)
        print_info "Checking job status..."
        ssh -p $SSH_PORT $SSH_HOST << 'ENDSSH'
            echo "═══════════════════════════════════════════════════════════"
            echo "  CURRENT JOBS STATUS"
            echo "═══════════════════════════════════════════════════════════"
            squeue -u team02 2>/dev/null || echo "No jobs found or squeue unavailable"
            echo ""
            echo "Recent models:"
            ls -lht ~/models/*.pt 2>/dev/null | head -n 5 || echo "No models found"
ENDSSH
        ;;

    3)
        print_info "Starting real-time monitoring..."
        print_warning "Press Ctrl+C to exit monitoring"
        sleep 2
        ssh -p $SSH_PORT -t $SSH_HOST "cd ~/coldstart && ./monitor_20min.sh"
        ;;

    4)
        print_info "Skipping job submission"
        ;;

    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

# ============================================================================
# Final Summary
# ============================================================================

print_header "DEPLOYMENT COMPLETE"

print_success "Code deployed to cluster"
print_success "Pre-flight checks completed"

echo ""
print_info "Next steps:"
echo "  1. SSH to cluster: ssh -p $SSH_PORT $SSH_HOST"
echo "  2. Navigate to project: cd ~/coldstart"
echo "  3. Check job status: squeue -u team02"
echo "  4. Monitor training: ./monitor_20min.sh"
echo "  5. View logs: tail -f ~/logs/*.out"
echo ""
echo "W&B Dashboard: https://wandb.ai/niranjanxprt-niranjanxprt/flower-federated-learning"
echo ""

print_success "All done! 🚀"
