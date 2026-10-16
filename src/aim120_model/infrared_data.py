"""Read frozen game IR sources without updating a sparse datamine checkout.

Missing sparse files may be staged, at the exact commit, in the source cache.
Runtime never downloads or mutates the dataset. A version mismatch is an error.
"""
from __future__ import annotations

import json
import re
import copy
from functools import lru_cache
from pathlib import Path


VERSION_COMMITS = {'2.59.0.27': '4ac255b829a27346b9300fccdc138362c2535fbe'}


@lru_cache(maxsize=8)
def _asset(path):
    return json.loads(Path(path).read_text())


class InfraredGameData:
    def __init__(self, version='2.59.0.27', *, project_dir=None):
        if version not in VERSION_COMMITS:
            raise ValueError('Unsupported frozen IR dataset version: '+str(version))
        self.version, self.commit = version, VERSION_COMMITS[version]
        self.root = Path(project_dir) if project_dir is not None else Path(__file__).resolve().parents[2]
        self.datamine = self.root/'data'/('datamine_'+version)
        self.cache = self.root/'outputs/ir_datamine_20261016'
        self.assets = self.root/'data/profile_sets'/version/'infrared'

    def _versioned_asset(self, name):
        data = _asset(str(self.assets/name))
        if data.get('version') != self.version or data.get('commit') != self.commit:
            raise ValueError('IR versioned asset does not match the requested frozen dataset')
        return data

    def coverage(self):
        """Source-indexed Air RB >=13.0 roster, mounts and explicit limitations."""
        return copy.deepcopy(self._versioned_asset('coverage.json'))

    def aircraft(self, name):
        name = self.identifier(name)
        entry = self._versioned_asset('coverage.json')['aircraft'].get(name)
        if entry is None:
            raise ValueError(f'{name} is not in the frozen {self.version} Air RB >=13.0 aircraft set')
        return copy.deepcopy(entry)

    @staticmethod
    def identifier(value):
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise ValueError('Game model identifier must contain letters, digits, underscore or hyphen')
        return value

    def read(self, path):
        relative = Path(path)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Game source path must be relative to the frozen dataset')
        local = self.datamine/relative
        if local.is_file():
            return json.loads(local.read_text())
        fields = self.assets/'game_fields.json'
        if fields.is_file():
            entry = self._versioned_asset('game_fields.json')['sources'].get(str(relative))
            if entry is not None:
                return copy.deepcopy(entry['data'])
        cached = self.cache/relative
        manifest = self.cache/'source_version.json'
        if cached.is_file() and manifest.is_file():
            metadata = json.loads(manifest.read_text())
            if metadata.get('version') != self.version or metadata.get('commit') != self.commit:
                raise ValueError('IR source cache version does not match the requested frozen dataset')
            return json.loads(cached.read_text())
        raise FileNotFoundError(
            f'Missing game IR source {path} at {self.version} ({self.commit}). '
            f'The sparse checkout may omit it; stage the exact git-show source at {cached} '
            'with a matching source_version.json. No other aircraft or version is substituted.')

    def weapon_path(self, name):
        name = self.identifier(name)
        coverage = self.assets/'coverage.json'
        if coverage.is_file():
            weapons = self._versioned_asset('coverage.json')['weapons']
            if name in weapons:
                return weapons[name]['source_path']
            # bulletName may differ from the actual launcher filename (pl_5e2).
            matches = [(n,v) for n,v in weapons.items() if v['missile_id'] == name]
            if matches:
                return min(matches,key=lambda x:(x[0] != name,'default' in x[0],'bol_pod' in x[0],x[0]))[1]['source_path']
        return 'aces.vromfs.bin_u/gamedata/weapons/rocketguns/'+name+'.blkx'

    def weapon(self, name):
        return self.read(self.weapon_path(name))

    def mounted_weapon(self, name, aircraft_model):
        """Resolve a launcher or bulletName, then verify its actual mount relation."""
        path = self.weapon_path(name)
        entry = self.aircraft(aircraft_model)
        allowed = entry['infrared_weapons']
        index = self._versioned_asset('coverage.json')['weapons']
        missile_id = index.get(Path(path).stem,{}).get('missile_id')
        if not any(n == Path(path).stem or index[n]['missile_id'] == missile_id for n in allowed):
            raise ValueError(f'{name} is not an IR AAM mounted by {aircraft_model} in {self.version}')
        return path

    def aircraft_fm(self, aircraft_model):
        """Resolve the real unit fmFile; shared/extensionless/rooted paths are retained."""
        unit_path = 'aces.vromfs.bin_u/gamedata/flightmodels/'+self.identifier(aircraft_model)+'.blkx'
        unit = self.read(unit_path)
        name = unit['fmFile'].lstrip('/')
        relative = Path(name)
        if '..' in relative.parts or relative.is_absolute():
            raise ValueError('Aircraft fmFile must stay within the frozen flightmodels tree')
        name = name.removesuffix('.blk')+'.blkx' if name.endswith('.blk') else name+'.blkx'
        fm_path = 'aces.vromfs.bin_u/gamedata/flightmodels/'+name
        return self.read(fm_path), fm_path, unit_path

    def missile_profile(self, name, *, launch_aircraft_model=None):
        path = self.mounted_weapon(name,launch_aircraft_model) if launch_aircraft_model else self.weapon_path(name)
        entry = self._versioned_asset('coverage.json')['weapons'].get(Path(path).stem)
        if entry is None:
            raise ValueError('No covered IR AAM profile for '+name)
        if entry['runtime_profile'] is None:
            raise ValueError(f"{name}: {entry['runtime_limitation']}; IR seeker is supported, flight model remains excluded")
        profile = json.loads((self.root/'data/profile_sets'/self.version/entry['runtime_profile']).read_text())
        if profile['provenance']['source_version'] != self.version or profile['provenance']['source_commit'] != self.commit:
            raise ValueError('IR missile profile version/commit mismatch')
        return profile

    def provenance(self, path):
        result = dict(version=self.version, commit=self.commit, source_path=path,
                      url=f'https://github.com/gszabi99/War-Thunder-Datamine/blob/{self.commit}/{path}')
        if (self.assets/'game_fields.json').is_file():
            entry = self._versioned_asset('game_fields.json')['sources'].get(path)
            if entry is not None:
                result.update(sha256=entry['sha256'],extracted_fields=entry['fields'])
        return result


def infrared_coverage(*, version='2.59.0.27', project_dir=None):
    return InfraredGameData(version,project_dir=project_dir).coverage()


def load_infrared_missile_profile(weapon, *, launch_aircraft_model=None, version='2.59.0.27', project_dir=None):
    return InfraredGameData(version,project_dir=project_dir).missile_profile(
        weapon,launch_aircraft_model=launch_aircraft_model)


__all__ = ['InfraredGameData', 'infrared_coverage', 'load_infrared_missile_profile']
