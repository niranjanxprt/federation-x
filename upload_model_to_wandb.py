#!/usr/bin/env python3
"""Upload all trained models to Weights & Biases."""

import os
import glob
import wandb
import torch

# W&B Configuration
WANDB_API_KEY = "7490948301e1e8d9c551bc502fb8b0b6b38c2851"
WANDB_PROJECT = "flower-federated-learning"
WANDB_ENTITY = "niranjanxprt-niranjanxprt"

# Models directory
MODELS_DIR = "models"


def extract_model_metadata(filename):
    """Extract AUROC and round from filename like 'job127_145241_round6_auroc7389.pt'."""
    basename = os.path.splitext(filename)[0]

    # Try to extract round and auroc from filename
    round_val = 0
    auroc_val = 0.0

    if "round" in basename:
        try:
            parts = basename.split("round")
            if len(parts) >= 2:
                round_part = parts[1]
                # Extract first digits after "round"
                round_digits = ""
                for char in round_part:
                    if char.isdigit():
                        round_digits += char
                    else:
                        break
                if round_digits:
                    round_val = int(round_digits)
        except:
            pass

    if "auroc" in basename:
        try:
            parts = basename.split("auroc")
            if len(parts) >= 2:
                auroc_part = parts[1]
                # Extract all digits after "auroc"
                auroc_digits = ""
                for char in auroc_part:
                    if char.isdigit():
                        auroc_digits += char
                    else:
                        break
                if auroc_digits:
                    auroc_val = int(auroc_digits) / 10000.0
        except:
            pass

    return round_val, auroc_val


def upload_all_models():
    """Upload all models from the models directory to W&B."""

    # Login to W&B
    wandb.login(key=WANDB_API_KEY, verify=True)
    print(f"✓ Logged into W&B as niranjanxprt (niranjanxprt-niranjanxprt)\n")

    # Find all .pt files in models directory
    if not os.path.exists(MODELS_DIR):
        print(f"ERROR: Models directory '{MODELS_DIR}' not found")
        return False

    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))

    if not model_files:
        print(f"ERROR: No .pt model files found in '{MODELS_DIR}' directory")
        return False

    print(f"Found {len(model_files)} model(s) to upload:\n")

    uploaded_count = 0
    failed_count = 0

    for model_path in sorted(model_files):
        filename = os.path.basename(model_path)

        print(f"Uploading: {filename}")

        try:
            # Get model file size
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  ✓ Size: {model_size_mb:.1f} MB")

            # Extract metadata from filename
            round_val, auroc_val = extract_model_metadata(filename)
            print(f"  ✓ Metadata - Round: {round_val}, AUROC: {auroc_val:.4f}")

            # Load and verify model
            state_dict = torch.load(model_path, map_location="cpu")
            print(f"  ✓ Loaded: {len(state_dict)} parameter tensors")

            # Create artifact name from filename
            artifact_name = os.path.splitext(filename)[0]

            # Initialize W&B run for this model
            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"upload-{artifact_name}",
                config={
                    "auroc": auroc_val,
                    "round": round_val,
                    "source": "remote_ssh",
                    "filename": filename,
                },
                job_type="model-upload"
            )

            # Create W&B artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Trained federated learning model - {filename}",
                metadata={
                    "auroc": auroc_val,
                    "round": round_val,
                    "filename": filename,
                    "model_source": "remote_ssh",
                    "framework": "Flower (Federated Learning)",
                    "task": "Chest X-Ray Classification"
                }
            )

            # Add model file to artifact
            artifact.add_file(model_path)
            print(f"  ✓ Created artifact: {artifact_name}")

            # Log artifact to W&B
            wandb.log_artifact(artifact)
            print(f"  ✓ Uploaded to W&B!")
            print(f"  → https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/artifacts/model/{artifact_name}\n")

            # Finish the run
            wandb.finish()
            uploaded_count += 1

        except Exception as e:
            print(f"  ✗ Error uploading: {e}\n")
            failed_count += 1
            if wandb.run is not None:
                wandb.finish()

    # Print summary
    print("=" * 70)
    print(f"Upload Summary:")
    print(f"  ✓ Successfully uploaded: {uploaded_count} model(s)")
    if failed_count > 0:
        print(f"  ✗ Failed to upload: {failed_count} model(s)")
    print(f"  Total processed: {len(model_files)}")
    print("=" * 70)
    print(f"\n✓ View all models at:")
    print(f"  https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/artifacts?artifactType=model")

    return failed_count == 0


if __name__ == "__main__":
    success = upload_all_models()
    exit(0 if success else 1)
