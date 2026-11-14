"""D=1 regression: no plotting imports; resume after a saved point.

python -m kagome_hexagon_triangle.dmi.tests.test_data_only_scan
"""
import csv
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


class NoPlotting(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if (fullname=='matplotlib' or fullname.startswith('matplotlib.')
                or fullname.startswith('kagome_hexagon_triangle.dmi.analysis')):
            raise ModuleNotFoundError(f'Plotting must not be imported by a server scan: {fullname}')


def main():
    blocker=NoPlotting();sys.meta_path.insert(0,blocker)
    try:
        from ..src import independent_random_fields as scan
        from ..src.plateau_explore import collect as collect_old_scan
        with tempfile.TemporaryDirectory(prefix='dmi_data_only_') as tmp:
            root=Path(tmp)
            cfg=scan.OPTIONS|dict(dims=[1],Dz=[.02],fields=[.38],trials=[1,2],
                measurement_device='cpu',threads=2,chi=2,ctm_max=8,
                fresh=[(.01,.01)],out=str(root))
            point=scan.point

            def stop_before_second(*args,**kwargs):
                if args[4]=='random2':raise RuntimeError('simulated interruption')
                return point(*args,**kwargs)

            # Reproduce an interrupted two-seed plan after point 1 was saved.
            with patch.object(scan,'point',side_effect=stop_before_second):
                try:scan.run(cfg)
                except RuntimeError as exc:
                    assert str(exc)=='simulated interruption'
                else:raise AssertionError('Expected the simulated interruption')
            first=scan.folder_for(cfg,1,.02,.38,1)
            saved={p:p.read_bytes() for p in (first/'independent_result.json',first/'initial.pt',first/'final.pt')}
            settings=(root/'settings.json').read_bytes()

            # Resume the identical settings and retain the already saved point.
            with patch.object(scan,'point',wraps=point) as calls:
                scan.run(cfg)
                assert calls.call_count==1 and calls.call_args.args[4]=='random2'
            assert (root/'settings.json').read_bytes()==settings
            assert json.loads(settings)==json.loads(json.dumps(cfg))
            assert all(p.read_bytes()==value for p,value in saved.items())
            with (root/'progress.csv').open(newline='',encoding='utf-8') as f:
                rows=list(csv.DictReader(f))
            assert len(rows)==2 and {r['trial'] for r in rows}=={'1','2'}
            assert all(r['warm_start']=='False' and r['measurement_device']=='cpu' for r in rows)
            assert not (root/'analysis').exists() and (root/'COMPLETED.txt').is_file()
            # The older continuation workflow must also remain data-only.
            old_root=root/'old_scan';old_root.mkdir()
            collect_old_scan(old_root)
        print('PASS: no matplotlib/report imports; saved point/checkpoints unchanged; identical-settings resume; both seeds in progress.csv.')
    finally:
        sys.meta_path.remove(blocker)


if __name__=='__main__':main()
