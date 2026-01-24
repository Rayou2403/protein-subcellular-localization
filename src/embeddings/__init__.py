"""Embeddings extraction and fusion utilities.

Modules:
  - fetch_esmc: Extract ESM-C embeddings from FASTA to HDF5.
  - fetch_prostt5: Extract ProstT5 embeddings with configurable pooling.
  - fetch_protbert: Extract ProtBert embeddings with configurable pooling.
  - concat: Concatenate (fuse) ESM-C and ProstT5 embeddings.
  - fetch_embeddings: Orchestrator CLI for the full pipeline.
"""
def run_esmc_extraction(*args, **kwargs):
    from .fetch_esmc import run_esmc_extraction as _run

    return _run(*args, **kwargs)


def run_prost_extraction(*args, **kwargs):
    from .fetch_prostt5 import run_prost_extraction as _run

    return _run(*args, **kwargs)

def run_protbert_extraction(*args, **kwargs):
    from .fetch_protbert import run_protbert_extraction as _run

    return _run(*args, **kwargs)


def run_fusion(*args, **kwargs):
    from .concat import run_fusion as _run

    return _run(*args, **kwargs)


__all__ = ["run_esmc_extraction", "run_prost_extraction", "run_protbert_extraction", "run_fusion"]
