# ============================================================
# src/utils/helpers.py
# Shared utility functions used across the entire project.
# ============================================================

import yaml                                        # YAML parser for reading config files
import joblib                                      # Efficient serialization for numpy arrays and sklearn models
from pathlib import Path                           # Object-oriented filesystem paths
from loguru import logger                          # Enhanced logging with colors, rotation, and formatting


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.
    Walks up from this file's location until it finds the 'config' folder.
    """
    # Start from the directory containing this helpers.py file
    current = Path(__file__).resolve().parent
    # Walk upward until we find the config/ directory (project root marker)
    while current != current.parent:
        # Check if the 'config' directory exists at this level
        if (current / "config").exists():
            # Found the project root, return it
            return current
        # Move one directory up
        current = current.parent
    # Fallback: if config/ not found, return 2 levels up from this file
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = None) -> dict:
    """
    Load the YAML configuration file and return it as a dictionary.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to config.yaml. If None, uses the default
        location at <project_root>/config/config.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary with all project settings.
    """
    # If no explicit path provided, build the default path
    if config_path is None:
        # Construct the path: <project_root>/config/config.yaml
        config_path = get_project_root() / "config" / "config.yaml"
    else:
        # Convert the provided string to a Path object
        config_path = Path(config_path)

    # Verify the config file actually exists before trying to read it
    if not config_path.exists():
        # Raise a clear error if the file is missing
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Open and parse the YAML file
    with open(config_path, "r", encoding="utf-8") as f:
        # yaml.safe_load prevents arbitrary code execution from YAML
        config = yaml.safe_load(f)

    # Log that we successfully loaded the config
    logger.info(f"Configuration loaded from: {config_path}")

    # Return the parsed configuration dictionary
    return config


def save_model(model, filepath: str) -> None:
    """
    Serialize and save a trained model to disk using joblib.

    Parameters
    ----------
    model : object
        The trained model object (sklearn, xgboost, etc.)
    filepath : str
        Destination file path for the serialized model.
    """
    # Convert to Path object for robust path handling
    filepath = Path(filepath)
    # Create parent directories if they don't exist (e.g., models/)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Serialize the model to disk using joblib (efficient for numpy arrays)
    joblib.dump(model, filepath)
    # Log the save location for traceability
    logger.info(f"Model saved to: {filepath}")


def load_model(filepath: str):
    """
    Load a serialized model from disk.

    Parameters
    ----------
    filepath : str
        Path to the serialized model file.

    Returns
    -------
    object
        The deserialized model object, ready for prediction.
    """
    # Convert to Path object
    filepath = Path(filepath)
    # Verify the model file exists
    if not filepath.exists():
        # Raise an error if model file is missing
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    # Deserialize and return the model
    model = joblib.load(filepath)
    # Log the load event
    logger.info(f"Model loaded from: {filepath}")
    # Return the deserialized model
    return model


def setup_logging(log_dir: str = "logs", log_file: str = "app.log") -> None:
    """
    Configure loguru for structured, rotated file logging and console output.

    Parameters
    ----------
    log_dir : str
        Directory where log files will be stored.
    log_file : str
        Name of the log file.
    """
    # Build the full path to the log file
    log_path = Path(get_project_root()) / log_dir / log_file
    # Create the log directory if it doesn't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove any existing loguru handlers to avoid duplicate logs
    logger.remove()
    # Add a console handler with colorized output and INFO level
    logger.add(
        sink=lambda msg: print(msg, end=""),       # Print to stdout
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # Timestamp in green
               "<level>{level: <8}</level> | "      # Log level
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "  # Source location
               "<level>{message}</level>",           # The actual log message
        level="INFO",                                # Minimum level for console
        colorize=True,                               # Enable ANSI color codes
    )
    # Add a file handler with rotation and retention policies
    logger.add(
        sink=str(log_path),                          # Write to this file
        format="{time:YYYY-MM-DD HH:mm:ss} | "      # Timestamp (no colors in file)
               "{level: <8} | "                      # Log level
               "{name}:{function}:{line} | "         # Source location
               "{message}",                          # The log message
        level="DEBUG",                               # Capture all levels in file
        rotation="10 MB",                            # Rotate when file reaches 10 MB
        retention="30 days",                         # Keep logs for 30 days
        compression="zip",                           # Compress rotated files
    )
    # Log that logging has been configured
    logger.info("Logging configured successfully.")


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        The directory path to ensure exists.

    Returns
    -------
    Path
        The Path object pointing to the ensured directory.
    """
    # Convert string to Path object
    dir_path = Path(path)
    # Create the directory and any missing parents
    dir_path.mkdir(parents=True, exist_ok=True)
    # Return the Path object for chaining
    return dir_path
