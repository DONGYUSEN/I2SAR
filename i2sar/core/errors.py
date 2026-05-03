class I2SARError(Exception):
    """Base exception for I2SAR."""


class SchemaError(I2SARError):
    """Raised when an HDF5 file does not match the I2SAR schema."""


class WorkflowError(I2SARError):
    """Raised when a workflow task cannot be executed."""
