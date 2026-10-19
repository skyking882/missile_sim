"""Choose a bounded raw-order history, never silently overwrite duplicate ticks."""
def select_window(history, epoch_tick, duration_ticks=288):
    if not history:
        raise ValueError('Empty own-state history')
    stop=min(history[-1]['tick'],epoch_tick+duration_ticks)
    issues=[]
    for a,b in zip(history,history[1:]):
        if b['tick']<=a['tick']:
            issue=dict(index0=a['index'],index1=b['index'],tick0=a['tick'],tick1=b['tick'])
            if min(a['tick'],b['tick'])<=stop:
                raise ValueError('Ambiguous own-state time inside scored window: '+str(issue))
            issues.append(issue)
    selected=[]
    for r in history:
        selected.append(r)
        if r['tick']>=stop:
            break
    if len(selected)<2:
        raise ValueError('Window lacks bracketing own-state records')
    return selected,dict(requested_end_tick=stop,retained_records=len(selected),original_records=len(history),nonincreasing_pairs_outside_window=issues)
