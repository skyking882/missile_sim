"""Independent Python reconstruction and common resource-scaled responses."""

def load_profile(missile_id):
    """Load an included public profile without depending on the working directory."""
    from .resources import read
    import copy
    if not isinstance(missile_id,str) or not missile_id.replace('_','').isalnum():
        raise ValueError('Invalid missile identifier')
    return copy.deepcopy(read('profiles/'+missile_id+'.json'))


def library_profiles():
    from .resources import DATA,Parameters
    profiles=[]
    for path in sorted((DATA/'profiles').glob('*.json')):
        profile=load_profile(path.stem)
        try:
            params=Parameters(path.stem);g=params.rocket['guidance'];radar=g.get('radarSeeker')
            if radar:
                if radar.get('absDopplerSpeed') or radar.get('scanHalfSector') or radar.get('useTargetId'):
                    raise NotImplementedError('absolute Doppler / scan / target-ID branch')
                for key in ['mainBeamNotchWidth','zeroDopplerNotchWidth']:
                    if radar.get(key,-2)>=0:raise NotImplementedError('clutter spectrum / notch branch')
                if not radar['active'] and not radar['semiActive']:
                    raise NotImplementedError('passive anti-radiation seeker')
            profile['_reconstructed_reason']='IR: explicit thermal scene; SARH: explicit PD illumination; branch-specific candidate, no per-missile trajectory validation'
            profile['_reconstructed_supported']=True
        except (NotImplementedError,KeyError) as exc:
            profile['_reconstructed_supported']=False;profile['_reconstructed_reason']=str(exc)
        profiles.append(profile)
    return profiles
