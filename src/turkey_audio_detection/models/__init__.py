"""Model subpackage."""

from turkey_audio_detection.models.sed import CnnSed, download_panns_weights

__all__ = ["CnnSed", "download_panns_weights"]
