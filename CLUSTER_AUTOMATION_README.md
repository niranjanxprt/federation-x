# 🚀 Cluster Automation Scripts

Automated deployment and testing scripts for the federated learning cluster.

## 📋 Overview

These scripts automate the entire workflow of deploying code to the cluster, running pre-flight checks, submitting jobs, and monitoring training progress.

**Cluster Details:**
- Host: `team02@129.212.178.168` (or `team02@134.199.193.89`)
- Port: `32605`
- Remote directory: `~/coldstart` (⚠️ must be exactly this name!)
- Virtual environment: `~/hackathon-venv` (read-only, pre-configured)
- Federation: `cluster-gpu` (for GPU jobs)
- Job submission: Only via `submit-job.sh` wrapper

---

## 🔑 Quick Start

### 1. Set Up SSH Keys (Recommended - Do This First!)

This eliminates the need to enter your password multiple times:

```bash
./setup_ssh_keys.sh
```

**What it does:**
- Generates SSH key if you don't have one
- Copies your public key to the cluster
- Tests passwordless authentication
- You'll only need to enter password ONCE

**After this, all other scripts will work without password prompts!**

---

### 2. Deploy and Test (Main Script)

```bash
./deploy_and_test.sh
```

**What it does:**
1. ✅ Connects to cluster
2. ✅ Pulls latest code from git branch
3. ✅ Runs pre-flight checks
4. ✅ Interactive menu for job submission
5. ✅ Provides monitoring options

**Interactive menu options:**
- Submit Job 1 (Aggressive Learning, LR=0.01)
- Submit Job 2 (Refinement, LR=0.005)
- Submit Job 3 (Fine-tuning, LR=0.001)
- Check job status
- Monitor training in real-time
- Skip job submission

---

### 3. Quick Job Submission

For quick job submission without going through the full menu:

```bash
# Job 1 (Aggressive)
./quick_submit.sh 1 my_job_name

# Job 2 (Medium)
./quick_submit.sh 2 my_job_name

# Job 3 (Fine-tune)
./quick_submit.sh 3 my_job_name
```

**Usage:**
```bash
./quick_submit.sh [1|2|3] [optional_job_name]
```

---

### 4. Check Status

Quick status check without submitting jobs:

```bash
./check_status.sh
```

**Shows:**
- 🔄 Active jobs (via squeue)
- 🏆 Top 5 recent models with AUROC scores
- 💾 Checkpoint files
- 📋 Recent log files
- 💿 Disk usage
- 📊 Git branch and commit info

---

## 📂 Available Scripts

### On Your Local Machine

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `setup_ssh_keys.sh` | Set up passwordless SSH | **Do this first!** |
| `deploy_and_test.sh` | Full deployment & testing workflow | Main automation script |
| `quick_submit.sh` | Quick job submission | When you need to submit jobs fast |
| `check_status.sh` | Check cluster status | Check without deploying |

### On the Cluster (in ~/coldstart)

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `monitor_20min.sh` | Real-time monitoring dashboard | Watch training progress |
| `preflight_20min.sh` | Pre-flight validation checks | Before submitting jobs |
| `overnight_queue.sh` | Queue multiple jobs overnight | Automated sequential training |

---

## 🎯 Typical Workflow

### First Time Setup

```bash
# 1. Set up SSH keys (one time only)
./setup_ssh_keys.sh

# 2. Deploy and run Job 1
./deploy_and_test.sh
# Select option 1 from menu

# 3. Monitor progress
# (Option 6 from menu, or separately:)
ssh -p 32605 -t team02@129.212.178.168 'cd ~/coldstart && ./monitor_20min.sh'
```

### Subsequent Jobs

```bash
# Quick check status
./check_status.sh

# Submit Job 2 after Job 1 completes
./quick_submit.sh 2 job2_refinement

# Submit Job 3 after Job 2 completes
./quick_submit.sh 3 job3_finetune
```

---

## 📊 Monitoring Training

### Option 1: Real-time Dashboard (Recommended)

```bash
ssh -p 32605 -t team02@129.212.178.168 'cd ~/coldstart && ./monitor_20min.sh'
```

**Shows:**
- Active jobs with status
- Queue status (running/pending)
- Top 5 models by AUROC
- Checkpoint files
- Disk usage
- GPU status (if available)
- Next recommended action
- Refreshes every 15 seconds

Press `Ctrl+C` to exit.

### Option 2: Check Status Periodically

```bash
./check_status.sh
```

### Option 3: Direct SLURM Commands

```bash
# Check job queue
ssh -p 32605 team02@129.212.178.168 'squeue -u team02'

# View recent logs
ssh -p 32605 team02@129.212.178.168 'tail -f ~/logs/*.out'

# Check models
ssh -p 32605 team02@129.212.178.168 'ls -lht ~/models/*.pt | head'
```

---

## 🔧 Configuration

### Before Running Jobs

Make sure `pyproject.toml` on the cluster is configured correctly:

```toml
[tool.flwr.app.config]
image-size = 128              # Keep for speed
num-server-rounds = 9         # 20-min optimized
local-epochs = 3              # Balance of speed/quality
lr = 0.01                     # Change per job (0.01 → 0.005 → 0.001)
```

**Job-specific LR values:**
- Job 1: `lr = 0.01` (Aggressive)
- Job 2: `lr = 0.005` (Medium)
- Job 3: `lr = 0.001` (Fine-tune)

### Updating Config Between Jobs

SSH to cluster and edit manually:

```bash
ssh -p 32605 team02@129.212.178.168
cd ~/coldstart
nano pyproject.toml  # or vim
# Change the lr value
# Save and exit
```

Or use sed remotely:

```bash
ssh -p 32605 team02@129.212.178.168 "cd ~/coldstart && sed -i 's/lr = .*/lr = 0.005/' pyproject.toml"
```

---

## 🐛 Troubleshooting

### SSH Password Prompts

**Problem:** Script asks for password multiple times

**Solution:**
```bash
./setup_ssh_keys.sh
```

### Connection Timeout

**Problem:** SSH connection times out

**Solution:**
- Cluster may be temporarily unavailable
- Check your network connection
- Try again in a few minutes
- Contact cluster admin if persistent

### Job Not Submitting

**Problem:** Job submission fails

**Solution:**
1. Check queue isn't full: `squeue -u team02`
2. Verify submit-job.sh exists on cluster
3. Check virtual environment is activated
4. Review error messages in output

### Script Permission Denied

**Problem:** `bash: ./script.sh: Permission denied`

**Solution:**
```bash
chmod +x deploy_and_test.sh quick_submit.sh check_status.sh setup_ssh_keys.sh
```

### Pre-flight Checks Fail

**Problem:** Pre-flight checks show errors

**Solution:**
1. Ensure code was deployed: `./deploy_and_test.sh`
2. Check git branch is correct
3. Verify all files exist on cluster
4. Review specific error messages

---

## 📝 Manual SSH Commands

If you prefer manual control:

```bash
# Connect to cluster
ssh -p 32605 team02@129.212.178.168

# Once connected:
cd ~/coldstart
git pull origin claude/fl-guide-subagents-plan-01TPDEivdvegb7uMnXnhx9U7
source ~/hackathon-venv/bin/activate
./preflight_20min.sh
./submit-job.sh "flwr run . cluster --stream" --gpu --name "my_job"
```

---

## 🔗 Useful Links

- **W&B Dashboard:** https://wandb.ai/niranjanxprt-niranjanxprt/flower-federated-learning
- **GitHub Branch:** `claude/fl-guide-subagents-plan-01TPDEivdvegb7uMnXnhx9U7`

---

## 📈 Expected Results

Following the 3-job strategy:

```
Job 1: Rounds 1-9   | LR=0.01   | 20min | AUROC 0.7389 → 0.7720 (+0.033)
Job 2: Rounds 10-18 | LR=0.005  | 20min | AUROC 0.7720 → 0.8050 (+0.033)
Job 3: Rounds 19-27 | LR=0.001  | 20min | AUROC 0.8050 → 0.8250 (+0.020)
───────────────────────────────────────────────────────────────────────
Total: 60 minutes | Final AUROC: 0.8250 | Improvement: +0.086
```

---

## 💡 Tips

1. **Always run setup_ssh_keys.sh first** - saves time
2. **Check status before submitting** - avoid queue conflicts
3. **Monitor first job closely** - verify everything works
4. **Update LR between jobs** - critical for optimal results
5. **Watch W&B dashboard** - best way to track AUROC progress
6. **Keep terminal open during jobs** - or use screen/tmux
7. **Save model checkpoints** - enable resuming if interrupted

---

## 🆘 Getting Help

If you encounter issues:

1. Run `./check_status.sh` to see current state
2. Check recent logs on cluster: `~/logs/*.out`
3. Review W&B dashboard for training metrics
4. Verify pyproject.toml configuration
5. Check this README for troubleshooting section

---

**Happy Training! 🚀**
