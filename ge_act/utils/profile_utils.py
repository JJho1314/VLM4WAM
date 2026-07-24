def should_save_final_checkpoint(*, cuda_profile_enabled: bool) -> bool:
    """Avoid multi-gigabyte final checkpoints for throwaway profiling runs."""

    return not cuda_profile_enabled
