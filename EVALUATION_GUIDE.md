# Evaluation Guide - How to Run evaluate.py

## ❌ Why `./submit-job.sh "python evaluate.py" --gpu` Doesn't Work Locally

### Problem
You're trying to run a **cluster-specific command** from your local machine or wrong directory.

**The command will ONLY work**:
- ✅ ON THE CLUSTER (ssh to team02@129.212.178.168)
- ✅ IN THE DIRECTORY: `~/coldstart/`
- ✅ AFTER: You've trained a model

**It will NOT work**:
- ❌ On your local machine
- ❌ In any other directory
- ❌ If `submit-job.sh` doesn't exist in current directory

---

## ✅ How to Run Evaluation Correctly

### Step 1: SSH to the Cluster
```bash
ssh -p 32605 team02@129.212.178.168
# Enter your password
```

### Step 2: Navigate to Your Project Directory
```bash
cd ~/coldstart
```

### Step 3: Pull Latest Code (with fixed evaluate.py)
```bash
git checkout claude/test-01TPDEivdvegb7uMnXnhx9U7
git pull origin claude/test-01TPDEivdvegb7uMnXnhx9U7
```

### Step 4: Run Evaluation

#### Option 1: Auto-detect Best Model (Easiest) ✨
```bash
./submit-job.sh "python evaluate.py" --gpu
```

**What it does**:
- Automatically finds the model with highest AUROC in `~/models/`
- Example: Finds `job1234_round9_auroc8456.pt` (AUROC 0.8456)
- Uses that model for evaluation

#### Option 2: Specify Exact Model Path
```bash
./submit-job.sh "python evaluate.py --model /home/team02/models/your_specific_model.pt" --gpu
```

**Use when**:
- You want to evaluate a specific model
- You know the exact filename

#### Option 3: Use Latest Checkpoint
```bash
./submit-job.sh "python evaluate.py --checkpoint job_name" --gpu
```

**What it does**:
- Loads from `/home/team02/checkpoints/job_name_latest.pt`
- Useful for evaluating a job that just finished

---

## 📋 Complete Example Workflow

### Scenario: You just finished Job 1, want to evaluate it

```bash
# 1. SSH to cluster
ssh -p 32605 team02@129.212.178.168

# 2. Go to project directory
cd ~/coldstart

# 3. Check what models you have
ls -lh ~/models/
# Output example:
# job1048_job1_pretrained_round3_auroc7456.pt
# job1048_job1_pretrained_round6_auroc8123.pt
# job1048_job1_pretrained_round9_auroc8456.pt  <- Best one!

# 4. Option A: Auto-detect best model
./submit-job.sh "python evaluate.py" --gpu

# 4. Option B: Specify exact model
./submit-job.sh "python evaluate.py --model /home/team02/models/job1048_job1_pretrained_round9_auroc8456.pt" --gpu

# 5. Check job status
squeue -u team02

# 6. View results (once job completes)
tail -100 ~/logs/job_XXXX_*.out
```

---

## 🔍 Expected Output

```
================================================================================
MODEL EVALUATION
================================================================================

No model specified, auto-detecting best model...
Found best model: /home/team02/models/job1048_job1_pretrained_round9_auroc8456.pt

Loading model from /home/team02/models/job1048_job1_pretrained_round9_auroc8456.pt...
Model loaded on cuda.

Evaluating...
  Hospital A      AUROC: 0.8532 (n=1234)
  Hospital B      AUROC: 0.7823 (n=987)
  Hospital C      AUROC: 0.8691 (n=1456)
  Eval Avg        AUROC: 0.8456
  Test A          AUROC: 0.8123 (n=500)
  Test B          AUROC: 0.7945 (n=500)
  Test C          AUROC: 0.8234 (n=500)
  Test D (OOD)    AUROC: 0.7512 (n=500)
  Test Avg        AUROC: 0.7954

================================================================================
```

---

## 🐛 Troubleshooting

### Error: `./submit-job.sh: No such file or directory`

**Problem**: You're not on the cluster or in wrong directory

**Solution**:
```bash
# Are you on the cluster?
hostname
# Should show: gpu-login-0 or similar

# Are you in the right directory?
pwd
# Should show: /home/team02/coldstart

# If not, navigate there
cd ~/coldstart

# Verify submit-job.sh exists
ls -la submit-job.sh
```

---

### Error: `Model file not found at ...`

**Problem**: The auto-detected model doesn't exist

**Solution**:
```bash
# Check what models you actually have
ls -lh ~/models/

# Specify the correct model path
./submit-job.sh "python evaluate.py --model /home/team02/models/ACTUAL_MODEL_NAME.pt" --gpu
```

---

### Error: `DATASET_DIR environment variable not set`

**Problem**: Running outside of Slurm environment

**Solution**:
```bash
# Don't run directly - use submit-job.sh wrapper
# ❌ Wrong:
python evaluate.py

# ✅ Correct:
./submit-job.sh "python evaluate.py" --gpu
```

---

### Error: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Problem**: Model architecture mismatch - trained on different branch than evaluating

**Symptoms**:
- Trained on **main branch** (no pretrained, 1-channel)
- Evaluating on **test branch** (pretrained, 3-channel)
- OR vice versa

**Solution**:
```bash
# Check which branch you trained on
cat ~/logs/job_XXXX_*.out | grep -i "pretrain"

# If you see "pretrained" or "IMAGENET1K_V1":
# -> Your model is from test branch
# -> Evaluate on test branch:
git checkout claude/test-01TPDEivdvegb7uMnXnhx9U7

# If you DON'T see "pretrained":
# -> Your model is from main branch
# -> Evaluate on main branch:
git checkout main
```

**Important**: The branch you evaluate on must match the branch you trained on!

---

## 📊 Understanding the Output

### Per-Hospital Metrics
```
Hospital A      AUROC: 0.8532 (n=1234)
Hospital B      AUROC: 0.7823 (n=987)
Hospital C      AUROC: 0.8691 (n=1456)
```
- Individual performance on each hospital's validation set
- `n=` number of samples in that hospital

### Eval Avg (Most Important for Hackathon)
```
Eval Avg        AUROC: 0.8456
```
- **This is your primary metric**
- Aggregated AUROC across all 3 hospitals
- This is what judges will use to rank you

### Test Sets (Hidden from Participants)
```
Test A          AUROC: 0.8123 (n=500)
Test B          AUROC: 0.7945 (n=500)
Test C          AUROC: 0.8234 (n=500)
Test D (OOD)    AUROC: 0.7512 (n=500)  <- Out-of-distribution
```
- These may not show up for you (test data hidden)
- Judges will evaluate on these
- Test D is "out-of-distribution" - hardest test

---

## 🎯 Best Practices

### 1. Always Use Auto-Detection
```bash
# ✅ Easiest and safest
./submit-job.sh "python evaluate.py" --gpu
```

### 2. Evaluate After Each Job Completes
```bash
# Job 1 finishes
./submit-job.sh "python evaluate.py" --gpu --name eval_job1

# Job 2 finishes
./submit-job.sh "python evaluate.py" --gpu --name eval_job2

# Job 3 finishes
./submit-job.sh "python evaluate.py" --gpu --name eval_job3
```

### 3. Track Your Progress
Create a file to log your results:
```bash
# After each evaluation, save the result
echo "Job 1 - AUROC: 0.7823" >> ~/evaluation_log.txt
echo "Job 2 - AUROC: 0.8234" >> ~/evaluation_log.txt
echo "Job 3 - AUROC: 0.8456" >> ~/evaluation_log.txt

# View history
cat ~/evaluation_log.txt
```

---

## 🚀 Quick Reference Commands

```bash
# SSH to cluster
ssh -p 32605 team02@129.212.178.168

# Go to project
cd ~/coldstart

# Pull latest code
git pull origin claude/test-01TPDEivdvegb7uMnXnhx9U7

# Evaluate (auto-detect best model)
./submit-job.sh "python evaluate.py" --gpu

# Check job status
squeue -u team02

# View results
tail -100 ~/logs/job_*_*.out | grep -A 20 "EVALUATION"
```

---

## 📝 Summary

**Why your command didn't work**:
1. ❌ `submit-job.sh` only exists on the cluster (not in git repo)
2. ❌ Must run from `~/coldstart/` directory on cluster
3. ❌ Old hardcoded MODEL_PATH was invalid

**Fixed in latest version**:
1. ✅ Auto-detects best model from `~/models/`
2. ✅ Supports manual model specification
3. ✅ Supports checkpoint loading
4. ✅ Better error messages

**To use**:
1. SSH to cluster
2. `cd ~/coldstart`
3. `git pull`
4. `./submit-job.sh "python evaluate.py" --gpu`

**Last Updated**: 2025-11-15
**Branch**: claude/test-01TPDEivdvegb7uMnXnhx9U7
