"""Independent repeat of measurement on the SAVED, unchanged final states.

Also records transverse magnetization and SHA256 of the input state.
One measurement per selected state, no further imaginary-time evolution.
"""
import hashlib
import json
from pathlib import Path
import time
import torch
from .simple_update import BlockState
from .honeycomb_ctm import HoneycombEnvironment, layers
from .observables import measure

ROOT = Path(__file__).parent/'results'
INPUTS = [ROOT/'D5_D6_c3_h038'/f'D{D}_seed1701'/'final.pt' for D in (5, 6)]
INPUTS += [ROOT/'D5_continuation_h038'/'D5_seed1701'/'final.pt']
INPUTS += [ROOT/'D6_continuation_h038'/'D6_seed1701'/'final.pt']
CHI = 24


def verify_one(path, chi=CHI):
    with torch.no_grad():
        if not path.exists():
            raise FileNotFoundError(path)
        state = BlockState.load(path)
        t0 = time.perf_counter()
        env = HoneycombEnvironment.initial(state.D).run(layers(state), chi, 50, 1e-6)
        obs = measure(state, env)
        obs['chi'] = chi
        obs['measurement_seconds'] = time.perf_counter()-t0
        obs['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = path.parent/f'observables_chi{chi}.json'
        if previous.exists():
            old = json.loads(previous.read_text())
            obs['repeat_energy_total_error'] = abs(obs['energy_total']-old['energy_total'])
            obs['repeat_magnetization_error'] = abs(obs['magnetization_normalized']-old['magnetization_normalized'])
        target = path.parent/f'verified_observables_chi{chi}.json'
        target.write_text(json.dumps(obs, indent=2)+'\n')
        print(str(path), json.dumps({k:v for k,v in obs.items() if k not in ('sites', 'bonds')}), flush=True)
        return obs


def main():
    torch.set_num_threads(10)
    torch.set_grad_enabled(False)
    for path in INPUTS:
        verify_one(path)
    verify_one(INPUTS[-1], 32)


if __name__ == '__main__':
    main()
