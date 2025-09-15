def resolve_model(model_name: str):
    """
    Map your CLI model names to classes in networks.py.
    """
    import networks as N
    mapping = {}

    # Common names used in this repo; add/adjust if needed
    if hasattr(N, 'KinematicModel'):
        mapping['kinematic-model'] = N.KinematicModel
    if hasattr(N, 'CustomKinematicModel'):
        mapping['multi-segment-model'] = N.CustomKinematicModel
    if hasattr(N, 'PSDCoeffModel'):
        mapping['power-spectral-coeff-model'] = N.PSDCoeffModel
        mapping['power-spectral-difference-model'] = N.PSDCoeffModel
        mapping['power-spectral-no-eeg-model'] = N.PSDCoeffModel
        mapping['power-spectral-no-kinematic-model'] = N.PSDCoeffModel

    if model_name not in mapping:
        raise ValueError(f'Unknown model name: {model_name}. Known: {sorted(mapping)}')
    return mapping[model_name]
