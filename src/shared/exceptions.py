"""Custom exceptions for the Belgian Energy Demand Forecasting system."""


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


class DataPipelineError(Exception):
    """Raised when data ingestion or transformation fails."""


class ModelError(Exception):
    """Raised when model loading, training, or serving fails."""


class ValidationError(Exception):
    """Raised when data validation fails."""
