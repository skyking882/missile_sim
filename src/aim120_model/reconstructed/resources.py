"""Package-local original inputs and verified initialized loader values.

Raw .80 weapon values take precedence. Omitted scalar defaults were recorded
as 'original caller default' by the .80 loader, not fitted to trajectories.
Context-dependent defaults are evaluated at their owner (N and semiActive).
"""
import copy
import json
from functools import lru_cache
from pathlib import Path
from .python_math118 import f

DATA = Path(__file__).with_name('data')
RAW = DATA/'raw'


@lru_cache(maxsize=None)
def read(name):
    return json.loads((DATA/name).read_text(encoding='utf-8'))


def loaded_curve(resource, prefix, *, timeout=None):
    points = [v for k,v in resource.items() if k.startswith(prefix) and k[len(prefix):].isdigit()]
    if points: return sorted([list(map(f,p)) for p in points])
    # Original empty gain table is a unit multiplier. timeOut supplies the
    # separate launch guard; it is not a guessed ramp or another missile's fit.
    return [[0.,1.],[1.,1.]]


def merge(a,b):
    value=copy.deepcopy(a)
    for k,v in b.items():
        value[k]=merge(value[k],v) if isinstance(v,dict) and isinstance(value.get(k),dict) else copy.deepcopy(v)
    return value


class Parameters:
    def __init__(self, missile_id, *, loft=False):
        index=read('resources.json')['missiles']
        if missile_id not in index: raise ValueError('No original reconstructed resource for '+missile_id)
        self.missile_id=missile_id;self.entry=index[missile_id]
        if self.entry['unsupported_features']:
            raise NotImplementedError('Unsupported branch: '+', '.join(self.entry['unsupported_features']))
        self.original=read('raw/'+self.entry['file'])['rocket']
        self.defaults=read('loader_defaults.json')
        self.rocket=copy.deepcopy(self.original)
        self.gameparams=read('raw/gameparams.blk');self.gameplay=read('raw/gameplay.blk')
        self.path=missile_id+'/rocket'
        self.guidance_path=self.path+'/guidance';self.sensor_path=self.guidance_path+'/radarSeeker'
        g=self.rocket['guidance'];a=g['guidanceAutopilot']
        a['loftEnabled']=bool(loft and a.get('loftEnabled',False))
        # Only caller defaults are merged; raw per-missile values always win.
        if 'radarSeeker' in g:
            radar=g['radarSeeker'];radar.setdefault('active',False)
            radar.setdefault('semiActive',not radar['active'])
            g['radarSeeker']=merge(self.radar_defaults(),radar)
        for k in (['lockTimeOut','lockDistance'] if g.get('radarSeeker',{}).get('active') else []):
            if k not in g:g[k]=self.get('guidance',k)

    def get(self,section,key):
        node=self.rocket
        for part in section.split('/') if section else []:node=node.get(part,{})
        if key in node:return node[key]
        if section=='guidance/guidanceAutopilot' and key=='omegaToAccelMult':
            return node['propNavMult']
        if section=='guidance/radarSeeker' and key=='semiActive':return not node.get('active',False)
        try:return self.defaults[section][key]
        except KeyError:raise NotImplementedError('Missing verified loader value: '+section+'/'+key) from None

    def radar_defaults(self):
        result=copy.deepcopy(self.defaults['guidance/radarSeeker'])
        for section,values in self.defaults.items():
            if section.startswith('guidance/radarSeeker/'):
                result[section.rsplit('/',1)[1]]=copy.deepcopy(values)
        result.setdefault('distance',{'presents':False})
        result['distance'].setdefault('presents',False)
        return result

    def physical(self):
        import math
        r=self.rocket;b=self.gameparams['shellBallisticsParams'];mass=f(r['mass'])
        d,length,wing=map(f,(r['caliber'],r['length'],r['wingAreaMult']))
        cy=f(self.get('','CyK'));cxaoa=f(r.get('CxAoA',b['props']['CxAoA']))
        if b['applyWingAreaMultToCxAoA']:cxaoa=f(cxaoa*f(wing*wing))
        S=f(f(d*d)*f(math.pi/4));Sw=f(f(f(f(.3)*wing)*length)*d);radius2=f(f(d*.5)**2)
        inertia=[f(mass*radius2)]+[f(mass*f((length*length+3*radius2)/12))]*2
        table=r.get('CyMachMult',{})
        points=list(table.values()) if isinstance(table,dict) else list(table)
        points += [v for k,v in r.items() if k.startswith('CyMachMult') and k[10:].isdigit()]
        if not points:points=[[0.,1.],[1.,1.]] # 1435F2ED5 constant-one table.
        lost=mass-f(r.get('massEnd1',r['massEnd']))
        kfin=f(f(f(r['finsLatAccel']*f(9.8100004196167))*mass)/f(Sw*cy))
        # 14142B647..678 scales loaded fin capability by burned-out/initial mass.
        kfin=f(kfin*f(f(mass-lost)/mass))
        return dict(mass=mass,massEnd=f(r['massEnd']),force=f(r['force']),timeFire=f(r['timeFire']),
            fireDelay=f(r['fireDelay']),endSpeed=f(r['endSpeed']),caliber=d,length=length,S=S,Sw=Sw,
            lever=f(f(r['distFromCmToStab'])/wing),CxK=f(r['CxK']),CxAoA=cxaoa,CyK=cy,
            CyMaxAoA=f(self.get('','CyMaxAoA')),WdK=list(map(f,self.get('','WdK'))),
            finsAoa=list(map(f,[r['finsAoaHor'],r['finsAoaVer']])),Kfin=kfin,
            CyMachMult=sorted(points),inertia0=inertia,useCxiMach=b['useCxiMach'],
            maxAcceleration=f(b['maxAcceleration']),gravity=f(9.8100004196167),precision=self.get('','precision'))
