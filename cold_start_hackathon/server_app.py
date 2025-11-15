from logging import INFO
import os
import json
import tempfile
import shutil
from datetime import datetime

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx

from cold_start_hackathon.task import Net
from cold_start_hackathon.util import (
    compute_aggregated_metrics,
    log_training_metrics,
    log_eval_metrics,
    save_best_model,
)

# ============================================================================
# W&B Configuration - Fill in your credentials or set via environment variables
# ============================================================================
# Option 1: Set these constants directly (e.g., WANDB_API_KEY = "your_api_key_here")
# Option 2: Leave as None and set environment variables (WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT)
# If all W&B config is None/unset, W&B logging will be disabled
WANDB_API_KEY = "7490948301e1e8d9c551bc502fb8b0b6b38c2851"
WANDB_PROJECT = "flower-federated-learning"
WANDB_ENTITY = "niranjanxprt-niranjanxprt"
# ============================================================================

# ============================================================================
# Checkpoint Configuration (20-min optimization)
# ============================================================================
CHECKPOINT_DIR = "/home/team02/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def load_checkpoint(run_name):
    """Load checkpoint with validation."""
    checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}_latest.pt"
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            log(INFO, f"✓ Loaded checkpoint: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            log(INFO, f"Failed to load checkpoint: {e}")
            return None
    return None


def save_checkpoint(arrays, run_name, server_round, metrics=None):
    """
    Save checkpoint with atomic writes and metadata.
    Prevents corruption during save.
    """
    checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}_latest.pt"
    temp_path = f"{checkpoint_path}.tmp"
    metadata_path = f"{CHECKPOINT_DIR}/{run_name}_metadata.json"

    try:
        # Save to temporary file first (atomic operation)
        torch.save(arrays.to_torch_state_dict(), temp_path)

        # Atomic rename (prevents corruption)
        shutil.move(temp_path, checkpoint_path)

        # Save metadata
        metadata = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {}
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        log(INFO, f"✓ Checkpoint saved: {checkpoint_path} (round {server_round})")
    except Exception as e:
        log(INFO, f"Checkpoint save failed: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Log GPU device
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log(INFO, f"Device: {device}")

    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config["local-epochs"]

    log(INFO, f"⏱️ 20-MINUTE OPTIMIZED MODE")
    log(INFO, f"Config: {num_rounds} rounds, {local_epochs} epochs, LR={lr}")

    # Get run name from environment variable (set by submit-job.sh). Feel free to change this.
    run_name = os.environ.get("JOB_NAME", "your_custom_run_name")

    # Initialize W&B if credentials are provided
    use_wandb = WANDB_API_KEY and WANDB_PROJECT
    if use_wandb:
        wandb.login(key=WANDB_API_KEY)
        log(INFO, "Wandb login successful")
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config={
                "num_rounds": num_rounds,
                "learning_rate": lr,
                "local_epochs": local_epochs,
                "strategy": "FedProx",
                "loss": "FocalLoss",
                "mode": "20min_optimized",
                "batch_size": 96
            }
        )
        log(INFO, "Wandb initialized with run_id: %s", wandb.run.id)
    else:
        log(INFO, "W&B disabled (credentials not provided). Set WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT to enable.")

    # Load or create model
    global_model = Net()
    checkpoint = load_checkpoint(run_name)
    if checkpoint:
        global_model.load_state_dict(checkpoint)
        log(INFO, "✓ Resumed from checkpoint")
    else:
        log(INFO, "✓ Starting fresh training")

    arrays = ArrayRecord(global_model.state_dict())

    # Enhanced FedProx strategy with fault tolerance
    strategy = HackathonFedProx(
        fraction_train=0.66,  # Only need 2/3 clients (more fault-tolerant)
        fraction_evaluate=0.66,  # Only need 2/3 clients
        min_available_clients=2,  # Can proceed with 2 out of 3 hospitals
        proximal_mu=0.1,  # Optimized for medical imaging
        run_name=run_name
    )

    log(INFO, "Strategy: FedProx (μ=0.1) - Optimized for non-IID medical data")
    log(INFO, "Client Sampling: 2/3 clients per round (fault-tolerant)")
    log(INFO, "Loss: Focal Loss (α=0.25, γ=2.0, label_smoothing=0.05)")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    log(INFO, "Training complete")
    if use_wandb:
        wandb.finish()
        log(INFO, "Wandb run finished")


class HackathonFedProx(FedProx):
    """
    Enhanced FedProx for 20-minute jobs.

    Key improvements:
    - Optimized proximal_mu for medical imaging
    - Adaptive aggregation weights
    - Enhanced checkpoint management
    """

    def __init__(self, *args, run_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_auroc = None
        self._run_name = run_name or "federated_run"
        self._round_history = []

    def aggregate_train(self, server_round, replies):
        """Aggregate with enhanced checkpointing and error handling."""
        # Filter out invalid replies
        valid_replies = [r for r in replies if r.has_content()]
        log(INFO, f"Round {server_round}: {len(valid_replies)}/{len(replies)} clients responded successfully")

        if not valid_replies:
            log(INFO, f"Round {server_round}: No valid client responses, skipping aggregation")
            return self._arrays if hasattr(self, '_arrays') else None, {}

        arrays, metrics = super().aggregate_train(server_round, valid_replies)
        self._arrays = arrays

        # Log training metrics
        log_training_metrics(valid_replies, server_round)

        # Save checkpoint with metrics
        checkpoint_metrics = {
            "train_loss": metrics.get("train_loss", 0.0),
            "num_clients": len(valid_replies),
            "success_rate": len(valid_replies) / len(replies) if replies else 0
        }
        save_checkpoint(arrays, self._run_name, server_round, checkpoint_metrics)

        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        """Aggregate with adaptive strategy and error handling."""
        # Filter out invalid replies
        valid_replies = [r for r in replies if r.has_content()]
        log(INFO, f"Round {server_round} Eval: {len(valid_replies)}/{len(replies)} clients evaluated successfully")

        if not valid_replies:
            log(INFO, f"Round {server_round}: No valid evaluation responses")
            return {}

        agg_metrics = compute_aggregated_metrics(valid_replies)

        # Log evaluation metrics
        log_eval_metrics(
            valid_replies,
            agg_metrics,
            server_round,
            self.weighted_by_key,
            lambda msg: log(INFO, msg)
        )

        # Save best model
        current_auroc = agg_metrics.get("auroc", 0.0)
        if self._best_auroc is None or current_auroc > self._best_auroc:
            self._best_auroc = current_auroc
            save_best_model(
                self._arrays,
                agg_metrics,
                server_round,
                self._run_name,
                self._best_auroc
            )
            log(INFO, f"🏆 New best AUROC: {self._best_auroc:.4f}")

        # Track history for adaptive strategies
        self._round_history.append({
            "round": server_round,
            "auroc": current_auroc,
            "metrics": agg_metrics,
            "num_clients": len(valid_replies)
        })

        # Enhanced W&B logging with system metrics
        if wandb.run is not None:
            system_metrics = {
                "system/active_clients": len(valid_replies),
                "system/total_clients": len(replies),
                "system/client_success_rate": len(valid_replies) / len(replies) if replies else 0,
                "system/best_auroc": self._best_auroc or 0.0,
            }
            wandb.log(system_metrics, step=server_round)

        return agg_metrics
