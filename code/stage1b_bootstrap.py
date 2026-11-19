"""Stage 1b: full voter bootstrap. Each resample redraws the electorate with replacement, RE-COMPUTES all eight
incumbent outcomes on the resampled ballots, and re-runs the rule-level election under each utility model.
Usage: python3 stage1b_bootstrap.py approval.zip NPROC NBOOT
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, _HERE)
import sys, json, zipfile, pickle, traceback, zlib
import numpy as np
import pbcore

ZIP, NPROC, NBOOT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
INC = pbcore.INCUMBENTS


def subinst(inst, rows):
    A = inst['A'][rows]
    J = dict(inst); J['A'] = A.tocsr(); J['Ac'] = A.tocsc(); J['n'] = A.shape[0]
    return J


def summarize(inst, outs):
    res = {}
    for kind in pbcore.KINDS:
        cls = pbcore.outcome_classes(INC, outs)
        U = np.stack([pbcore.utilities(inst, outs[c[0]], kind) for c in cls])
        P = pbcore.pairwise_counts(U)
        lab = pbcore.classify(P)
        cw, weak, smith = pbcore.condorcet(P)
        res[kind] = dict(label=lab, cw=sorted(cls[cw]) if cw >= 0 else None)
    return res


def work(fn):
    try:
        inst = pbcore.build_instance(pbcore.parse_pb(zipfile.ZipFile(ZIP).read(fn).decode('utf-8')), fn)
        base = summarize(inst, pbcore.run_incumbents(inst))
        rng = np.random.default_rng(zlib.crc32(fn.encode()) + 7)
        tally = {k: dict(labels={}, same_cw=0, overlap_cw=0, cw_exists=0, member={r: 0 for r in INC}) for k in pbcore.KINDS}
        nb = NBOOT if inst['n'] <= 20000 else max(NBOOT // 2, 20)
        for b in range(nb):
            rows = rng.integers(0, inst['n'], inst['n'])
            J = subinst(inst, rows)
            s = summarize(J, pbcore.run_incumbents(J))
            for k in pbcore.KINDS:
                t = tally[k]; lab = s[k]['label']
                t['labels'][lab] = t['labels'].get(lab, 0) + 1
                if s[k]['cw'] is not None:
                    t['cw_exists'] += 1
                    t['same_cw'] += int(base[k]['cw'] is not None and set(s[k]['cw']) == set(base[k]['cw']))
                    t['overlap_cw'] += int(base[k]['cw'] is not None and bool(set(s[k]['cw']) & set(base[k]['cw'])))
                    for r in s[k]['cw']:
                        t['member'][r] += 1
        return dict(file=fn, base=base, boot=tally, nboot=nb)
    except Exception:
        return dict(file=fn, error=traceback.format_exc()[-800:])


if __name__ == '__main__':
    outs = pickle.load(open('out/stage1_outcomes.pkl', 'rb'))
    recs = {r['file']: r for r in json.load(open('out/stage1_instances.json'))}
    files = sorted(outs, key=lambda f: -recs[f]['n'] * recs[f]['m'])
    res = []
    with pbcore.make_pool(NPROC) as pool:
        for k, r in enumerate(pool.imap_unordered(work, files, chunksize=1)):
            res.append(r)
            if k % 100 == 0 or 'error' in r:
                print(k, r['file'], r.get('error', '')[:300], flush=True)
    json.dump(res, open('out/bootstrap_full.json', 'w'))
    print('done', len(res), sum('error' in r for r in res), flush=True)
