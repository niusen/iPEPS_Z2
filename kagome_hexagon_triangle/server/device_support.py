"""Move only the CTMRG/measurement phase to CUDA; keep SU/checkpoints on CPU.

Existing SU gates remain CPU tensors because evolution stays on CPU. The
existing double_layer function moves small observable operators to the
state's device. All environment tensors are explicitly moved here once.
No global default device or legacy kernel is modified.
"""
import torch
from ..simple_update import BlockState
from ..honeycomb_ctm import HoneycombEnvironment


def resolve_measurement_device(name):
    device = torch.device(name)
    if device.type == 'cpu':
        if device.index is not None:
            raise ValueError('Use measurement_device="cpu", without an index')
        return device
    if device.type != 'cuda':
        raise ValueError('measurement_device must be cpu or cuda:N')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but this Python/PyTorch environment has no available CUDA device')
    index = device.index if device.index is not None else torch.cuda.current_device()
    if not 0 <= index < torch.cuda.device_count():
        raise RuntimeError(f'CUDA device {index} requested, but only {torch.cuda.device_count()} visible device(s) exist')
    return torch.device('cuda', index)


def measurement_state(state, device):
    return BlockState([t.to(device) for t in state.A],
                      [t.to(device) for t in state.B],
                      [[w.to(device) for w in row] for row in state.weights],
                      state.field, state.seed, steps=state.steps, c3=state.c3)


def initial_environment(D, device):
    env = HoneycombEnvironment.initial(D)
    for face in env.faces:
        face.C = [t.to(device) for t in face.C]
        face.Ta = [t.to(device) for t in face.Ta]
        face.Tb = [t.to(device) for t in face.Tb]
    return env


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def immutable_settings(config):
    # Execution device does not change the physics or SU state. Excluding
    # this new key preserves existing v1 CPU checkpoint fingerprints and
    # permits selecting a GPU for still-unfinished measurements on resume.
    return {k:v for k,v in config.items() if k != 'measurement_device'}
