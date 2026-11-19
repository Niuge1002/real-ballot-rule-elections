"""Stage 2 (remote): train a 'Democratic AI' PB rule by evolution strategies.

Objective (Koster et al. 2022 analogue): expected share of voters who prefer the learned rule's
outcome to a rival rule's outcome, averaged over training elections and over a rival set.
Voters vote by a logistic model on their normalised utility difference (temperature TAU),
indifferent voters split 0.5/0.5.  Optional penalty LAM * (exclusion rate of learned outcome).

Usage: python3 train_dai.py KIND RIVALS SEED LAM NWORK TAG [HOLD]
  KIND in {card, cost, sqrt}; RIVALS in {league, league_distinct, GreedyAV, MES}; LAM >= 0
  HOLD: optional municipality held out (leave-one-municipality-out); training then uses the train and in-distribution
  test partitions of all other non-Warsaw municipalities, and validation their validation partition.
  'league_distinct' averages over the DISTINCT rival outcomes (rules with identical outcomes count once).
"""
import sys, json, zipfile, pickle, time
import numpy as np
from multiprocessing import Pool
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, _HERE)
import pbcore

KIND, RIVALS, SEED, LAM, NWORK, TAG = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
HOLD = sys.argv[7] if len(sys.argv) > 7 else ''
TAU = 0.05
POP_PAIRS, GENS, BATCH, SIGMA, LR, INIT_SD = 32, 200, 64, 0.10, 0.05, 0.5
NMAX_TRAIN = 5000

if RIVALS in ('league', 'league_distinct'):
    RIV = list(pbcore.INCUMBENTS)
elif RIVALS == 'GreedyAV':
    RIV = ['GreedyAV']
else:
    RIV = ['MES-' + KIND]

_I = {}


def _load(files):
    zf = zipfile.ZipFile('approval.zip')
    outs = pickle.load(open('out/stage1_outcomes.pkl', 'rb'))
    for f in files:
        inst = pbcore.build_instance(pbcore.parse_pb(zf.read(f).decode('utf-8')), f)
        norm = pbcore.ballot_norm(inst, KIND)
        if RIVALS == 'league_distinct':
            cls = pbcore.outcome_classes(RIV, outs[f])
            U_riv = np.stack([pbcore.utilities(inst, outs[f][c[0]], KIND) for c in cls])
        else:
            U_riv = np.stack([pbcore.utilities(inst, outs[f][r], KIND) for r in RIV])
        _I[f] = (inst, norm, U_riv)


def init_worker(files):
    _load(files)


def score(args):
    theta, f, hard = args
    inst, norm, U_riv = _I[f]
    W = pbcore.rule_learned(inst, theta)
    u = pbcore.utilities(inst, W, KIND)
    d = (u[None, :] - U_riv) / np.maximum(norm, 1e-12)[None, :]
    if hard:
        share = (np.sign(d) * 0.5 + 0.5).mean()
    else:
        share = (1.0 / (1.0 + np.exp(-d / TAU))).mean()
    excl = (u <= 0).mean()
    return share - LAM * excl, share, excl


if __name__ == '__main__':
    recs = json.load(open('out/stage1_instances.json'))
    split = json.load(open('out/split.json'))
    if HOLD == 'TEMPORAL':
        # genuine temporal split: develop on non-Warsaw editions voted in 2022 or earlier, test on 2023 or later
        yr = lambda r: int(str(r['year'])[:4])
        pool_ = [r for r in recs if split.get(r['file']) and r['city'] != 'Warszawa' and yr(r) <= 2022 and r['n'] <= NMAX_TRAIN]
        train = sorted(r['file'] for r in pool_ if split[r['file']] in ('train', 'test_id'))
        val = sorted(r['file'] for r in pool_ if split[r['file']] == 'val')
    elif HOLD:
        pool_ = [r for r in recs if split.get(r['file']) and r['city'] not in ('Warszawa', HOLD) and r['n'] <= NMAX_TRAIN]
        train = sorted(r['file'] for r in pool_ if split[r['file']] in ('train', 'test_id'))
        val = sorted(r['file'] for r in pool_ if split[r['file']] == 'val')
    else:
        train = sorted(r['file'] for r in recs if split.get(r['file']) == 'train' and r['n'] <= NMAX_TRAIN)
        val = sorted(r['file'] for r in recs if split.get(r['file']) == 'val' and r['n'] <= NMAX_TRAIN)
    rng = np.random.default_rng(1000 + SEED)
    D = pbcore.n_params()
    theta = rng.normal(0, INIT_SD, D)
    m_, v_ = np.zeros(D), np.zeros(D)
    log = []
    best = (-np.inf, theta.copy(), -1)
    t0 = time.time()
    with pbcore.make_pool(NWORK, initializer=init_worker, initargs=(train + val,)) as pool:
        def evaluate(thetas, files, hard):
            jobs = [(th, f, hard) for th in thetas for f in files]
            res = np.array(pool.map(score, jobs, chunksize=max(1, len(jobs) // (4 * NWORK))))
            return res.reshape(len(thetas), len(files), 3).mean(1)
        for g in range(GENS + 1):
            if g % 10 == 0:
                vr = evaluate([theta], val, True)[0]
                tr = evaluate([theta], train, True)[0]
                log.append(dict(gen=g, val_obj=vr[0], val_share=vr[1], val_excl=vr[2], train_share=tr[1], train_excl=tr[2], secs=time.time() - t0))
                print(TAG, log[-1], flush=True)
                if vr[0] > best[0]:
                    best = (vr[0], theta.copy(), g)
            if g == GENS:
                break
            batch = rng.choice(len(train), size=min(BATCH, len(train)), replace=False)
            files = [train[i] for i in batch]
            eps = rng.normal(0, 1, (POP_PAIRS, D))
            thetas = [theta + SIGMA * e for e in eps] + [theta - SIGMA * e for e in eps]
            fit = evaluate(thetas, files, False)[:, 0]
            ranks = np.empty(len(fit)); ranks[np.argsort(fit)] = np.arange(len(fit))
            shaped = ranks / (len(fit) - 1) - 0.5
            grad = ((shaped[:POP_PAIRS] - shaped[POP_PAIRS:])[:, None] * eps).sum(0) / (2 * POP_PAIRS * SIGMA)
            grad -= 0.001 * theta
            m_ = 0.9 * m_ + 0.1 * grad; v_ = 0.999 * v_ + 0.001 * grad ** 2
            mh = m_ / (1 - 0.9 ** (g + 1)); vh = v_ / (1 - 0.999 ** (g + 1))
            theta = theta + LR * mh / (np.sqrt(vh) + 1e-8)
    json.dump(dict(tag=TAG, kind=KIND, rivals=RIVALS, seed=SEED, lam=LAM, hold=HOLD, train_files=train, val_files=val, best_gen=best[2], best_val=best[0],
                   theta=best[1].tolist(), theta_final=theta.tolist(), log=log, n_train=len(train), n_val=len(val),
                   hyper=dict(TAU=TAU, POP_PAIRS=POP_PAIRS, GENS=GENS, BATCH=BATCH, SIGMA=SIGMA, LR=LR, INIT_SD=INIT_SD, NMAX_TRAIN=NMAX_TRAIN)),
              open(f'out/dai_{TAG}.json', 'w'))
