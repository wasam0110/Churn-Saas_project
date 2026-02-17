# ============================================================
# src/models/registry.py
# Model registry: versioning, metadata tracking, and
# lifecycle management for trained models.
# ============================================================

import json                                        # JSON for serializing model metadata
import hashlib                                     # Hashing for model fingerprints
import joblib                                      # Model serialization
from datetime import datetime                      # Timestamps for model versions
from pathlib import Path                           # Object-oriented file paths
from loguru import logger                          # Structured logging
from typing import Dict, Any, Optional, List       # Type hints
import pandas as pd                               # DataFrame utilities for registry listings


class ModelRegistry:
    """
    Simple file-based model registry for tracking trained models.

    Tracks model versions, metadata (hyperparameters, metrics),
    and provides easy loading of any previous version.

    For production, consider using MLflow's model registry instead.
    """

    def __init__(self, registry_dir: str = "models"):
        """
        Initialize the model registry.

        Parameters
        ----------
        registry_dir : str
            Base directory for storing models and metadata.
        """
        # Allow passing the full config dict (backwards compatibility):
        # If a dict is passed, extract model_registry.dir
        if isinstance(registry_dir, dict):
            registry_dir = registry_dir.get("model_registry", {}).get("dir", "models")

        # Set the registry directory path
        self.registry_dir = Path(registry_dir)
        # Create the directory if it doesn't exist
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        # Path to the registry index file (tracks all versions)
        self.index_path = self.registry_dir / "registry_index.json"
        # Load existing registry index or create a new one
        self.index = self._load_index()
        # Log initialization
        logger.info(f"ModelRegistry initialized at: {self.registry_dir}")

    def _load_index(self) -> Dict:
        """
        Load the registry index from disk, or create a new empty one.

        Returns
        -------
        dict
            The registry index containing all model version metadata.
        """
        # Check if the index file exists
        if self.index_path.exists():
            # Load the existing index
            with open(self.index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            # Log the loaded index
            logger.info(f"Registry index loaded: {len(index.get('models', {}))} models tracked")
            # Return the loaded index
            return index
        else:
            # Create a new empty index
            logger.info("Creating new registry index")
            return {"models": {}, "production_model": None}

    def _save_index(self) -> None:
        """
        Save the registry index to disk.
        """
        # Write the index as formatted JSON
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2, default=str)
        # Log the save
        logger.debug("Registry index saved")

    def _generate_version_id(self, model_name: str) -> str:
        """
        Generate a unique version ID for a model.

        Format: {model_name}_v{number}_{timestamp}

        Parameters
        ----------
        model_name : str
            Name of the model algorithm.

        Returns
        -------
        str
            Unique version identifier string.
        """
        # Count existing versions of this model
        existing_versions = [
            key for key in self.index["models"]        # Iterate over all registered models
            if key.startswith(model_name)              # Filter by model name prefix
        ]
        # Increment version number
        version_num = len(existing_versions) + 1
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Build the version ID
        version_id = f"{model_name}_v{version_num}_{timestamp}"
        # Return the version ID
        return version_id

    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        description: str = "",
        config: Optional[Dict] = None,
    ) -> int:
        """
        Register a trained model with its metadata.

        Parameters
        ----------
        model : object
            The trained model object to serialize.
        model_name : str
            Name of the algorithm (e.g., 'xgboost').
        metrics : dict
            Performance metrics from evaluation.
        hyperparameters : dict, optional
            Hyperparameters used for training. Defaults to empty dict.
        feature_names : list of str, optional
            Names of input features. Defaults to empty list.
        description : str, optional
            Human-readable description of this model version.
        config : dict, optional
            Full project config (for backward compatibility).

        Returns
        -------
        int
            The numeric version number assigned to this model.
        """
        # Handle defaults for optional parameters
        if hyperparameters is None:
            hyperparameters = {}
        if feature_names is None:
            feature_names = []
        # Generate a unique version ID
        version_id = self._generate_version_id(model_name)

        # Build the model file path
        model_path = self.registry_dir / f"{version_id}.joblib"

        # Serialize and save the model to disk
        joblib.dump(model, model_path)

        # Build the metadata dictionary
        metadata = {
            "version_id": version_id,                  # Unique version identifier
            "model_name": model_name,                  # Algorithm name
            "model_path": str(model_path),             # Path to serialized model
            "registered_at": datetime.now().isoformat(),  # Registration timestamp
            "metrics": metrics,                         # Performance metrics
            "hyperparameters": hyperparameters,         # Training config
            "feature_names": feature_names,             # Input feature names
            "description": description,                 # Human description
            "stage": "staging",                         # Initial stage (staging, not production)
        }

        # Add to the registry index
        self.index["models"][version_id] = metadata

        # Save the updated index
        self._save_index()

        # Log the registration
        logger.info(f"Model registered: {version_id}")
        logger.info(f"  Path: {model_path}")
        logger.info(f"  Metrics: {metrics}")

        # Extract numeric version for backward compatibility
        # version_id format: "model_name_v{num}_{timestamp}"
        import re
        match = re.search(r'_v(\d+)_', version_id)
        numeric_version = int(match.group(1)) if match else len(self.index["models"])

        # Return the numeric version
        return numeric_version

    def promote_to_production(self, version_id: str) -> None:
        """
        Promote a model version to production status.

        Only one model can be in production at a time. The previous
        production model is moved to 'archived' stage.

        Parameters
        ----------
        version_id : str
            The version ID to promote.
        """
        # Check that the version exists
        if version_id not in self.index["models"]:
            raise ValueError(f"Model version '{version_id}' not found in registry")

        # Demote the current production model (if any)
        current_prod = self.index["production_model"]
        if current_prod and current_prod in self.index["models"]:
            # Move current production to archived
            self.index["models"][current_prod]["stage"] = "archived"
            logger.info(f"Previous production model archived: {current_prod}")

        # Promote the new model
        self.index["models"][version_id]["stage"] = "production"
        self.index["production_model"] = version_id

        # Also save as 'best_model.joblib' for easy access
        source_path = Path(self.index["models"][version_id]["model_path"])
        best_model_path = self.registry_dir / "best_model.joblib"
        # Copy the model file
        import shutil
        shutil.copy2(source_path, best_model_path)

        # Save the updated index
        self._save_index()

        # Log the promotion
        logger.info(f"Model promoted to production: {version_id}")
        logger.info(f"  Also saved as: {best_model_path}")

    def load_model(self, version_id: Optional[str] = None) -> Any:
        """
        Load a model from the registry.

        Parameters
        ----------
        version_id : str, optional
            Specific version to load. If None, loads the production model.

        Returns
        -------
        object
            The deserialized model object.
        """
        # If no version specified, load the production model
        if version_id is None:
            version_id = self.index.get("production_model")
            if version_id is None:
                raise ValueError("No production model set. Specify a version_id.")

        # Check that the version exists
        if version_id not in self.index["models"]:
            raise ValueError(f"Model version '{version_id}' not found")

        # Get the model path
        model_path = self.index["models"][version_id]["model_path"]

        # Load and return the model
        model = joblib.load(model_path)
        logger.info(f"Model loaded: {version_id}")

        return model

    def get_model_info(self, version_id: str) -> Dict:
        """
        Get metadata for a specific model version.

        Parameters
        ----------
        version_id : str
            The version ID to query.

        Returns
        -------
        dict
            Model metadata dictionary.
        """
        # Check that the version exists
        if version_id not in self.index["models"]:
            raise ValueError(f"Model version '{version_id}' not found")

        # Return a copy of the metadata
        return self.index["models"][version_id].copy()

    def list_models(self, stage: Optional[str] = None) -> pd.DataFrame:
        """
        List all registered models, optionally filtered by stage.

        Parameters
        ----------
        stage : str, optional
            Filter by stage: 'staging', 'production', or 'archived'.

        Returns
        -------
        pd.DataFrame
            DataFrame listing all matching model versions with metadata.
        """
        # Import pandas here to avoid circular imports
        import pandas as pd

        # Collect model info into a list
        models_list = []
        for vid, meta in self.index["models"].items():
            # Apply stage filter if specified
            if stage is not None and meta.get("stage") != stage:
                continue
            # Add model info to the list
            models_list.append({
                "version_id": vid,
                "model_name": meta["model_name"],
                "stage": meta["stage"],
                "registered_at": meta["registered_at"],
                "pr_auc": meta["metrics"].get("pr_auc", None),
                "roc_auc": meta["metrics"].get("roc_auc", None),
                "f1": meta["metrics"].get("f1", None),
            })

        # Convert to DataFrame
        df = pd.DataFrame(models_list)

        # Log the count
        logger.info(f"Listed {len(df)} models" + (f" (stage={stage})" if stage else ""))

        # Return the DataFrame
        return df
