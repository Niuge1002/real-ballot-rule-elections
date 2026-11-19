"""Stage 1 (v2): instance selection, 8 incumbent rules + official outcome, covariates, ballot caps, solver log,
SHA-256 manifest of input files, and a GROUPED data partition (groups = municipality x year 'editions').
Usage: python3 stage1_rules.py approval.zip catalogue.json NPROC
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, _HERE)
import sys, os, json, zipfile, time, pickle, traceback, hashlib
import numpy as np
import pbcore

ZIP, CAT, NPROC = sys.argv[1], sys.argv[2], int(sys.argv[3])
cat = {t['file_name']: t for t in json.load(open(CAT))}
SEED = 20260928


def eligible(fn):
    t = cat.get(fn)
    return (t is not None and t['vote_type'] == 'approval' and str(t['experimental']) not in ('True', '1')
            and t.get('checker_status') != 'invalid')


def work(fn):
    t0 = time.time()
    try:
        raw = zipfile.ZipFile(ZIP).read(fn)
        inst = pbcore.build_instance(pbcore.parse_pb(raw.decode('utf-8')), fn)
        c, B = inst['cost'], inst['B']
        rec = dict(file=fn, sha256=hashlib.sha256(raw).hexdigest(), n=inst['n'], m=inst['m'], B=B, total_cost=float(c.sum()))
        if not np.isfinite(B) or inst['n'] < 10 or inst['m'] < 3:
            rec['excluded'] = 'size_or_budget'; return rec, None, []
        if c.sum() <= B:
            rec['excluded'] = 'fully_fundable'; return rec, None, []
        pbcore.SOLVER_LOG.clear()
        outs = pbcore.run_incumbents(inst)
        log = list(pbcore.SOLVER_LOG)
        off = pbcore.official_outcome(inst)
        if off is not None:
            outs['Official'] = off
        rec.update(pbcore.instance_covariates(inst))
        rec.update(pbcore.ballot_cap_info(inst))
        rec['has_official'] = off is not None
        for k in ('age', 'sex', 'nbh'):
            rec['has_' + k] = bool(np.mean([a not in ('', None) for a in inst[k]]) > 0.8)
        rec['n_classes'] = len(pbcore.outcome_classes(pbcore.INCUMBENTS, outs))
        rec['meta_rule'] = inst['meta'].get('rule', '')
        rec['meta_comment'] = inst['meta'].get('comment', '')[:300]
        rec['secs'] = time.time() - t0
        return rec, {k: v.astype(bool) for k, v in outs.items()}, log
    except Exception:
        return dict(file=fn, error=traceback.format_exc()[-800:]), None, []


if __name__ == '__main__':
    files = [f for f in zipfile.ZipFile(ZIP).namelist() if f.endswith('.pb') and eligible(f)]
    files.sort(key=lambda f: -float(cat[f].get('num_votes_raw') or 0) * float(cat[f].get('num_projects_raw') or 1))
    print('eligible', len(files), flush=True)
    recs, outs, slog = [], {}, []
    with pbcore.make_pool(NPROC) as pool:
        for k, (rec, o, log) in enumerate(pool.imap_unordered(work, files, chunksize=1)):
            t = cat[rec['file']]
            rec.update(country=t['country'], city=t['city'], year=t['year'], rule_raw=t.get('rule_raw'))
            rec['edition'] = f"{t['city']}|{t['year']}"
            recs.append(rec); slog += log
            if o is not None:
                outs[rec['file']] = o
            if k % 200 == 0:
                print(k, rec['file'], rec.get('secs'), rec.get('excluded'), rec.get('error', '')[:200], flush=True)
    # grouped partition: all Warsaw elections are the held-out municipality; other editions are assigned whole
    rng = np.random.default_rng(SEED)
    kept = sorted(outs)
    ed = {r['file']: r['edition'] for r in recs}
    city = {r['file']: r['city'] for r in recs}
    groups = sorted({ed[f] for f in kept if city[f] != 'Warszawa'})
    u = dict(zip(groups, rng.random(len(groups))))
    split = {}
    for f in kept:
        if city[f] == 'Warszawa':
            split[f] = 'test_ood_city'
        else:
            x = u[ed[f]]
            split[f] = 'train' if x < 0.50 else ('val' if x < 0.65 else 'test_id')
    for r in recs:
        r['split'] = split.get(r['file'], '')
    os.makedirs('out', exist_ok=True)
    json.dump(recs, open('out/stage1_instances.json', 'w'))
    pickle.dump(outs, open('out/stage1_outcomes.pkl', 'wb'))
    json.dump(split, open('out/split.json', 'w'))
    json.dump(slog, open('out/solver_log.json', 'w'))
    json.dump({r['file']: r.get('sha256') for r in recs}, open('out/manifest_sha256.json', 'w'), indent=0)
    bad = [s for s in slog if not s.get('primary_kept', True) or s.get('status1', 0) != 0]
    print('kept', len(kept), 'groups', len(groups), 'errors', sum('error' in r for r in recs), 'solver_issues', len(bad), flush=True)
