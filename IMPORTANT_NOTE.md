# ⚠️ IMPORTANT: pyproject.toml is FIXED

## 🔒 Key Constraint

**`pyproject.toml` CANNOT be modified** - it's managed by the cluster system and uses a fixed virtual environment.

---

## ✅ Solution: Runtime Configuration Override

Use the `--run-config` flag to override configuration at job submission time:

```bash
flwr run . cluster-gpu --stream --run-config "num-server-rounds=9 local-epochs=3 lr=0.01"
```

**Note**: Use `cluster-gpu` federation (not just `cluster`) for GPU jobs!

---

## 🚀 Easy Way: Use the Automation Scripts

The automation scripts **automatically handle config overrides** for you:

### Quick Submit
```bash
# Job 1: Aggressive Learning (LR=0.01)
./quick_submit.sh 1 my_job_name

# Job 2: Refinement (LR=0.005)
./quick_submit.sh 2 my_job_name

# Job 3: Fine-tuning (LR=0.001)
./quick_submit.sh 3 my_job_name
```

### Deploy and Test (Interactive)
```bash
./deploy_and_test.sh
```

Then select from menu:
1. Job 1 - Aggressive Learning (LR=0.01, 9 rounds, 3 epochs)
2. Job 2 - Refinement (LR=0.005, 9 rounds, 3 epochs)
3. Job 3 - Fine-tuning (LR=0.001, 9 rounds, 3 epochs)
4. Custom configuration

**The scripts will automatically pass the correct `--run-config` parameters!**

---

## 📋 What Gets Overridden

The scripts override these pyproject.toml defaults:

| Parameter | Default (in pyproject.toml) | Job 1 | Job 2 | Job 3 |
|-----------|---------------------------|-------|-------|-------|
| `num-server-rounds` | 100 | **9** | **9** | **9** |
| `local-epochs` | 1 | **3** | **3** | **3** |
| `lr` | 0.01 | **0.01** | **0.005** | **0.001** |

---

## 🔍 Verify Current Config

Check what's in pyproject.toml on the cluster:

```bash
./show_config.sh
```

**But remember**: You don't need to edit it - just override at runtime!

---

## 📚 Complete Workflow

### Step-by-Step for 3 Jobs:

```bash
# 1. First time setup (one-time only)
./setup_ssh_keys.sh

# 2. Deploy code and submit Job 1
./deploy_and_test.sh
# Select: 1 (Job 1 - Aggressive Learning)
# Then: 1 (Submit job)

# 3. Wait for Job 1 to complete (~20 min)
./check_status.sh

# 4. Submit Job 2
./quick_submit.sh 2 job2_refinement

# 5. Wait for Job 2 to complete (~20 min)
./check_status.sh

# 6. Submit Job 3
./quick_submit.sh 3 job3_finetune

# 7. Final results (~20 min)
./check_status.sh
```

**Total Time**: ~60 minutes
**Expected AUROC**: 0.7389 → 0.8250 (+0.086)

---

## 🎯 Technical Details

### How --run-config Works

Flower's `--run-config` flag allows runtime parameter overrides:

```bash
flwr run . cluster --stream \
  --run-config "num-server-rounds=9 local-epochs=3 lr=0.01"
```

These values override whatever is in `pyproject.toml` at runtime, so:
- ✅ No need to edit files on cluster
- ✅ pyproject.toml stays unchanged
- ✅ Each job can have different config
- ✅ Works with managed/fixed environments

### In the Code

The values are accessed in `server_app.py`:

```python
num_rounds: int = context.run_config["num-server-rounds"]
lr: float = context.run_config["lr"]
local_epochs: int = context.run_config["local-epochs"]
```

The `--run-config` values take precedence over `pyproject.toml`.

---

## ❓ FAQ

**Q: Can I edit pyproject.toml?**
A: No - it's managed by the cluster system and uses a fixed virtual environment.

**Q: How do I change the learning rate?**
A: Use `./quick_submit.sh 1/2/3` or pass `--run-config "lr=0.005"` at runtime.

**Q: What if I need custom config?**
A: Use `./deploy_and_test.sh` and select option 4 (Custom configuration).

**Q: Do the automation scripts handle this automatically?**
A: Yes! They pass the correct `--run-config` parameters automatically.

**Q: What about the FL guide recommendations?**
A: All implemented! The scripts use the recommended values (9 rounds, 3 epochs, LR sequence).

---

## ✅ Summary

- 🔒 **pyproject.toml is FIXED** - don't try to edit it
- ✅ **Use `--run-config`** to override at runtime
- 🚀 **Automation scripts handle it** - just use `quick_submit.sh` or `deploy_and_test.sh`
- 📊 **Same results expected** - AUROC 0.7389 → 0.8250 in 60 minutes

**You're all set! Start with `./setup_ssh_keys.sh` then `./deploy_and_test.sh`**

---

## 🏗️ Cluster Information

### Key Constraints
- **Virtual Environment**: Read-only `hackathon-venv` (pre-configured)
- **Home Directory**: `~/coldstart` (must be named exactly this!)
- **Job Submission**: Only via `submit-job.sh` wrapper script
- **pyproject.toml**: FIXED - cannot be edited
- **Federation Name**: `cluster-gpu` (for GPU jobs)

### Job Submission Command Structure
```bash
./submit-job.sh "<command>" [--name <job-name>] [--gpu]
```

**Examples:**
```bash
# Basic (uses default config from pyproject.toml)
./submit-job.sh "flwr run . cluster-gpu" --gpu

# With runtime config override (what our scripts use)
./submit-job.sh "flwr run . cluster-gpu --stream --run-config \"num-server-rounds=9 local-epochs=3 lr=0.01\"" --gpu --name "my_job"
```

### Monitoring Jobs
```bash
# Check job status
squeue -u team02

# View job history
sacct -u team02

# Watch logs (real-time)
tail -f ~/logs/job*.out

# Or use our monitoring script
./monitor_20min.sh
```

### Useful Slurm Commands
| Command | Description |
|---------|-------------|
| `squeue -u team02` | Show running/queued jobs |
| `scancel <job_id>` | Cancel specific job |
| `scontrol show job <job_id>` | Detailed job info |
| `sinfo` | Show cluster status |
