"""
voice_camouflage_detection package
====================================
KTU S5 ECE Digital Signal Processing Project.

Public API:
    from voice_camouflage_detection import pipeline
    result = pipeline.run_from_file("sample.wav")
"""
from . import (
    config,
    utils,
    audio_io,
    preprocessing,
    transforms,
    features,
    classifier,
    visualize,
    pipeline,
)

__version__ = "1.0.0"
__all__ = [
    "config", "utils", "audio_io", "preprocessing",
    "transforms", "features", "classifier", "visualize", "pipeline",
]
