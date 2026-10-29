"""Frozen .27 thermal scene inputs, distinct from the .80 missile resources."""
import copy
import re
from pathlib import PurePosixPath
from .resources import read


class InfraredGameData:
    def __init__(self,version='2.59.0.27',**kwargs):
        if version!='2.59.0.27':raise ValueError('Thermal dataset is explicitly 2.59.0.27')
        self.version=version;self.commit='4ac255b829a27346b9300fccdc138362c2535fbe'

    @staticmethod
    def identifier(value):
        if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',value):raise ValueError('Invalid aircraft identifier')
        return value

    def coverage(self):return copy.deepcopy(read('infrared/coverage.json'))

    def aircraft(self,name):
        try:return self.coverage()['aircraft'][self.identifier(name)]
        except KeyError:raise ValueError('Aircraft is outside the frozen thermal dataset: '+name) from None

    def aircraft_fm(self,name):
        record=self.aircraft(name)
        path=record['fm_source'];unit=record['unit_source']
        return self.read(path),path,unit

    def read(self,path):
        # PurePosixPath keeps dataset keys identical on Windows and Unix.
        key=PurePosixPath(str(path).replace('\\','/')).as_posix()
        try:return copy.deepcopy(read('infrared/game_fields.json')['sources'][key]['data'])
        except KeyError:raise ValueError('Missing frozen thermal source: '+key) from None

    def provenance(self,path):
        return dict(path=path,version=self.version,commit=self.commit,representation='frozen JSON fields')
