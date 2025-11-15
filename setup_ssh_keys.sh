#!/bin/bash
# setup_ssh_keys.sh - Set up SSH key authentication for passwordless login

SSH_HOST="team02@129.212.178.168"
SSH_PORT="32605"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  SSH KEY SETUP FOR PASSWORDLESS LOGIN${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

echo "This script will set up SSH key authentication to avoid"
echo "entering passwords multiple times."
echo ""

# Check if SSH key exists
if [ -f ~/.ssh/id_rsa.pub ] || [ -f ~/.ssh/id_ed25519.pub ]; then
    echo -e "${GREEN}✓${NC} SSH key found"

    if [ -f ~/.ssh/id_ed25519.pub ]; then
        KEY_FILE=~/.ssh/id_ed25519.pub
        KEY_TYPE="ed25519"
    else
        KEY_FILE=~/.ssh/id_rsa.pub
        KEY_TYPE="rsa"
    fi

    echo -e "${GREEN}✓${NC} Using $KEY_TYPE key: $KEY_FILE"
else
    echo -e "${YELLOW}⚠${NC}  No SSH key found. Generating one..."

    # Generate SSH key
    ssh-keygen -t ed25519 -C "team02@hackathon-cluster" -f ~/.ssh/id_ed25519 -N ""

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} SSH key generated successfully"
        KEY_FILE=~/.ssh/id_ed25519.pub
    else
        echo -e "${RED}✗${NC} Failed to generate SSH key"
        exit 1
    fi
fi

echo ""
echo "Now copying SSH key to cluster..."
echo "You will be asked for your password ONE TIME."
echo ""

# Copy SSH key to server
ssh-copy-id -p $SSH_PORT $SSH_HOST

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓${NC} SSH key copied successfully!"
    echo ""
    echo "Testing passwordless login..."

    # Test connection
    if ssh -p $SSH_PORT -o BatchMode=yes $SSH_HOST echo "Connection successful" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Passwordless SSH is working!"
        echo ""
        echo "You can now use these scripts without entering password:"
        echo "  - ./deploy_and_test.sh"
        echo "  - ./quick_submit.sh"
        echo "  - ./check_status.sh"
    else
        echo -e "${YELLOW}⚠${NC}  SSH key copied but authentication test failed"
        echo "Please try manually: ssh -p $SSH_PORT $SSH_HOST"
    fi
else
    echo -e "${RED}✗${NC} Failed to copy SSH key"
    echo ""
    echo "You can try manually:"
    echo "  ssh-copy-id -p $SSH_PORT $SSH_HOST"
    exit 1
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo ""
