"""
Timing of energy, forces and virial for small boxes of q-TIP4P/F water, following
what i-PI's metatomic driver does at each step: neighbor lists from vesin, evaluation
of the exported model, and forces and virial from automatic differentiation with
respect to positions and strain.

    python benchmarks/benchmark.py --device cpu --threads 1
    python benchmarks/benchmark.py --device cuda --sizes 64 --batch 1 8 --profile
"""

import argparse
import json
import math
import os
import statistics
import tempfile
import time

import numpy as np
import torch
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)
from vesin.metatomic import NeighborList


# number of Angstrom^3 per molecule at 0.997 g/cm^3
VOLUME_PER_MOLECULE = 18.015 / (0.997 * 0.602214076)


def random_rotation(rng):
    q = rng.normal(size=4)
    a, b, c, d = q / np.linalg.norm(q)
    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
        ]
    )


def water_box(n_molecules, seed=0):
    """Randomly placed and oriented molecules (O-O > 2.4 A) in a cubic box"""
    rng = np.random.default_rng(seed)
    length = (n_molecules * VOLUME_PER_MOLECULE) ** (1 / 3)
    theta = math.radians(107.4)
    local = 0.9419 * np.array(
        [[0, 0, 0], [1, 0, 0], [math.cos(theta), math.sin(theta), 0]]
    )
    oxygens = []
    while len(oxygens) < n_molecules:
        trial = rng.uniform(0, length, size=3)
        if oxygens:
            d = np.array(oxygens) - trial
            d -= length * np.round(d / length)
            if np.min(np.linalg.norm(d, axis=1)) < 2.4:
                continue
        oxygens.append(trial)
    positions = np.concatenate([o + local @ random_rotation(rng).T for o in oxygens])
    return np.array([8, 1, 1] * n_molecules), positions, length * np.eye(3)


RNG = np.random.default_rng(0)


def make_systems(types, positions, cell, batch, device):
    """
    Systems with positions and cell depending on a strain, as in i-PI. Positions are
    randomly displaced, so that the number of neighbor pairs changes between calls as
    in a simulation.
    """
    all_positions, strains, systems = [], [], []
    for _ in range(batch):
        displaced = positions + RNG.normal(scale=0.02, size=positions.shape)
        pos = torch.tensor(displaced, device=device, requires_grad=True)
        strain = torch.eye(3, dtype=torch.float64, device=device, requires_grad=True)
        systems.append(
            System(
                types=torch.tensor(types, dtype=torch.int32, device=device),
                positions=pos @ strain,
                cell=torch.tensor(cell, device=device) @ strain,
                pbc=torch.tensor([True, True, True], device=device),
            )
        )
        all_positions.append(pos)
        strains.append(strain)
    return all_positions, strains, systems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 216])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--model", default=None, help="exported model to use")
    parser.add_argument("--profile", action="store_true", help="count ops & syncs")
    parser.add_argument("--json", default=None, help="save the results here")
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    if args.model is None:
        from metatomic_qtip4pf import get_model

        path = os.path.join(tempfile.mkdtemp(), "qtip4pf.pt")
        get_model().save(path)
    else:
        path = args.model
    model = load_atomistic_model(path).to(device)

    options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={"energy": ModelOutput(unit="eV", sample_kind="system")},
    )
    calculators = [
        NeighborList(nl, "angstrom") for nl in model.requested_neighbor_lists()
    ]

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def step(types, positions, cell, batch):
        all_positions, strains, systems = make_systems(
            types, positions, cell, batch, device
        )
        synchronize()
        t0 = time.perf_counter()
        for calculator in calculators:
            calculator.add_neighbor_list(systems)
        synchronize()
        t1 = time.perf_counter()
        outputs = model(systems, options, check_consistency=False)
        energy = outputs["energy"].block().values.sum()
        synchronize()
        t2 = time.perf_counter()
        torch.autograd.grad(energy, all_positions + strains)
        synchronize()
        t3 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2

    print(
        f"torch {torch.__version__}, device {device}, threads "
        f"{torch.get_num_threads()}, model {path}"
    )
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device))
    print(
        f"{'waters':>7} {'batch':>6} {'NL/ms':>8} {'fwd/ms':>8} {'bwd/ms':>8} "
        f"{'model/ms':>9} {'model/ms/system':>16}"
    )
    results = []
    for n_molecules in args.sizes:
        types, positions, cell = water_box(n_molecules)
        for batch in args.batch:
            for _ in range(5):
                step(types, positions, cell, batch)
            timings = np.array(
                [step(types, positions, cell, batch) for _ in range(args.repeats)]
            )
            nl, fwd, bwd = (1e3 * statistics.median(t) for t in timings.T)
            print(
                f"{n_molecules:7d} {batch:6d} {nl:8.2f} {fwd:8.2f} {bwd:8.2f} "
                f"{fwd + bwd:9.2f} {(fwd + bwd) / batch:16.3f}"
            )
            result = dict(waters=n_molecules, batch=batch, nl=nl, fwd=fwd, bwd=bwd)

            if args.profile:
                result.update(
                    profile(model, options, types, positions, cell, batch, device)
                )
                print(
                    "        ops: {ops}, CUDA kernels: {kernels}, "
                    "syncs: {syncs}".format(**result)
                )
            results.append(result)

    if args.json is not None:
        with open(args.json, "w") as fd:
            json.dump(
                dict(
                    torch=torch.__version__,
                    device=str(device),
                    gpu=torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None,
                    threads=torch.get_num_threads(),
                    results=results,
                ),
                fd,
                indent=2,
            )


def profile(model, options, types, positions, cell, batch, device):
    """Number of aten ops, CUDA kernels and device -> host syncs in fwd + bwd"""
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    all_positions, strains, systems = make_systems(
        types, positions, cell, batch, device
    )
    for nl in model.requested_neighbor_lists():
        NeighborList(nl, "angstrom").add_neighbor_list(systems)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with torch_profile(activities=activities) as prof:
        outputs = model(systems, options, check_consistency=False)
        energy = outputs["energy"].block().values.sum()
        torch.autograd.grad(energy, all_positions + strains)
        if device.type == "cuda":
            torch.cuda.synchronize()

    events = prof.key_averages()
    ops = sum(e.count for e in events if e.key.startswith("aten::"))
    kernels = sum(e.count for e in events if e.device_type.name == "CUDA")
    syncs = sum(
        e.count
        for e in events
        if e.key in ("aten::_local_scalar_dense", "aten::nonzero")
    )
    return dict(ops=ops, kernels=kernels, syncs=syncs)


if __name__ == "__main__":
    main()
