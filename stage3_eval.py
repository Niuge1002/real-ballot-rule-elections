"""Stage 3 (v2): model-implied rule-level elections, benchmark selectors, proportionality and minority audits.
Usage: python3 stage3_eval.py NPROC dai_*.json
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, _HERE)
import sys, json, zipfile, pickle, time, traceback, zlib
import numpy as np
import pbcore

NPROC = int(sys.argv[1])
DAI = {}
for fn in sys.argv[2:]:
    d = json.load(open(fn)); DAI[d['tag']] = d
OUTS = pickle.load(open('out/stage1_outcomes.pkl', 'rb'))
INC = pbcore.INCUMBENTS
KINDS = pbcore.KINDS
NBOOT = 200
EXACT_M, EXACT_N = 20, 30000


def main_tag(kind):
    c = [d for d in DAI.values() if d['kind'] == kind and d['rivals'] == 'league' and d['lam'] == 0 and not d.get('hold')]
    return max(c, key=lambda d: d['best_val'])['tag'] if c else None


MAIN = {k: main_tag(k) for k in KINDS if main_tag(k)}


def shares_vs(U, ia, ib, n):
    ab = int((U[ia] > U[ib]).sum()); ba = int((U[ib] > U[ia]).sum())
    return ab, ba


def election(U, names, outs, rng):
    classes = pbcore.outcome_classes(names, outs)
    rep = [names.index(c[0]) for c in classes]
    Uc = U[rep]
    P = pbcore.pairwise_counts(Uc)
    lab = pbcore.classify(P)
    cw, weak, smith = pbcore.condorcet(P)
    res = dict(classes=classes, label=lab, cw=cw, weak=weak, smith=smith, ml=pbcore.maximal_lottery(P).tolist(),
               P=P.astype(int).tolist())
    K = len(classes)
    if K > 1:
        iu = [(a, b) for a in range(K) for b in range(a + 1, K)]
        S = np.stack([np.sign(Uc[a] - Uc[b]) for a, b in iu], axis=1)
        n = U.shape[1]
        Wb = rng.multinomial(n, np.ones(n) / n, size=NBOOT).T.astype(float)
        Mb = S.T @ Wb
        same = exists = 0
        for t in range(NBOOT):
            M = np.zeros((K, K))
            for q, (a, b) in enumerate(iu):
                M[a, b] = Mb[q, t]; M[b, a] = -Mb[q, t]
            w = [a for a in range(K) if all(M[a, b] > 0 for b in range(K) if b != a)]
            if w:
                exists += 1; same += int(w[0] == cw)
        res['cond_boot_exists'] = exists / NBOOT; res['cond_boot_same'] = same / NBOOT
    return res


def groups_age(age):
    out = np.full(len(age), '', dtype=object)
    for i, a in enumerate(age):
        try:
            x = float(a)
        except (TypeError, ValueError):
            continue
        if 10 <= x <= 110:
            out[i] = '<30' if x < 30 else ('30-49' if x < 50 else ('50-64' if x < 65 else '65+'))
    return out


def quintiles(score):
    """Equal-size quintile groups by rank (ties broken by voter order), 0 = least mainstream."""
    r = np.empty(len(score), int); r[np.argsort(score, kind='stable')] = np.arange(len(score))
    return np.minimum((5 * r) // len(score), 4)


def work(fn):
    t0 = time.time()
    try:
        inst = pbcore.build_instance(pbcore.parse_pb(zipfile.ZipFile('approval.zip').read(fn).decode('utf-8')), fn)
        outs = dict(OUTS[fn])
        for tag, d in DAI.items():
            outs['DAI:' + tag] = pbcore.rule_learned(inst, np.array(d['theta']))
        n, c, B = inst['n'], inst['cost'], inst['B']
        app = np.asarray(inst['Ac'].sum(0)).ravel()
        blen = np.asarray(inst['A'].sum(1)).ravel()
        mainstream = (inst['A'] @ (app / n)) / blen
        qv = quintiles(mainstream)
        agegrp = groups_age(inst['age'])
        sex = np.array([s if s in ('M', 'F', 'K') else '' for s in inst['sex']], dtype=object); sex[sex == 'K'] = 'F'
        nbh = inst['nbh']
        rng = np.random.default_rng(zlib.crc32(fn.encode()))
        small = inst['m'] <= EXACT_M and n <= EXACT_N
        rec = dict(file=fn, main=MAIN, kinds={}, small=small)
        ejr_cache = {}
        for kind in KINDS:
            # selectors among incumbents, using the common-eight convention (self-comparison counts 0.5)
            Ui = np.stack([pbcore.utilities(inst, outs[r], kind) for r in INC])
            Si = np.zeros((8, 8))
            for a in range(8):
                Si[a] = ((Ui[a][None, :] > Ui).sum(1) + 0.5 * (Ui[a][None, :] == Ui).sum(1)) / n
            score = Si.mean(1)
            sel = INC[int(np.argmax(score))]
            ejr_inc = {}
            for r in INC:
                key = (outs[r].tobytes(), kind)
                if key not in ejr_cache:
                    ejr_cache[key] = pbcore.ejr_restricted(inst, outs[r], kind)
                ejr_inc[r] = ejr_cache[key]
            okr = [i for i, r in enumerate(INC) if ejr_inc[r]['n_exact'] == 0]
            selejr = INC[max(okr, key=lambda i: score[i])] if okr else ('MES-card' if kind == 'card' else 'MES-cost')
            outs['Sel:' + kind] = outs[sel].copy(); outs['SelEJR:' + kind] = outs[selejr].copy()
            rec.setdefault('selector_choice', {})[kind] = dict(sel=sel, selejr=selejr, n_ejr_ok=len(okr))
        names = list(outs)
        rec['rules'] = names
        for kind in KINDS:
            U = np.stack([pbcore.utilities(inst, outs[r], kind) for r in names])
            norm = np.maximum(pbcore.ballot_norm(inst, kind), 1e-12)
            Un = U / norm[None, :]
            K = len(names)
            ii = [names.index(r) for r in INC]
            vs_ab = np.zeros((K, 8), int); vs_ba = np.zeros((K, 8), int); s2 = np.zeros((K, 8)); s8 = np.zeros((K, 8))
            for b, ib in enumerate(ii):
                d = U - U[ib][None, :]
                vs_ab[:, b] = (d > 0).sum(1); vs_ba[:, b] = (d < 0).sum(1)
                dn = Un - Un[ib][None, :]
                s2[:, b] = (1 / (1 + np.exp(-2 * dn))).mean(1); s8[:, b] = (1 / (1 + np.exp(-8 * dn))).mean(1)
            core = INC + [r for r in names if r == 'Official' or r.endswith(':' + kind) or
                          (r.startswith('DAI:') and (DAI[r[4:]]['kind'] == kind or r[4:] in MAIN.values()))]
            ci = [names.index(r) for r in core]
            Pc = pbcore.pairwise_counts(U[ci]).astype(int)
            K_ = dict(vs_ab=vs_ab.tolist(), vs_ba=vs_ba.tolist(), soft2=s2.round(5).tolist(), soft8=s8.round(5).tolist(),
                      core=core, P_core=Pc.tolist(), elections={}, per_rule={}, decomp={})
            sets = {'incumbents': INC}
            if kind in MAIN:
                sets['inc+DAI'] = INC + ['DAI:' + MAIN[kind]]
            for sname, rl in sets.items():
                idx = [names.index(r) for r in rl]
                K_['elections'][sname] = election(U[idx], rl, {r: outs[r] for r in rl}, rng)
            # mechanism decomposition: who prefers which outcome, by mainstream quintile
            if kind in MAIN:
                a = names.index('DAI:' + MAIN[kind])
                for rival in ['MES-card' if kind == 'card' else 'MES-cost', 'GreedyAV', 'SelEJR:' + kind, 'Sel:' + kind]:
                    b = names.index(rival)
                    K_['decomp'][rival] = [[int(((U[a] > U[b]) & (qv == q)).sum()), int(((U[a] < U[b]) & (qv == q)).sum()),
                                            int(((U[a] == U[b]) & (qv == q)).sum())] for q in range(5)]
            done = {}
            for k, r in enumerate(names):
                key = outs[r].tobytes()
                if key in done:
                    K_['per_rule'][r] = dict(same_as=done[key]); continue
                done[key] = r
                ek = (key, kind)
                if ek not in ejr_cache:
                    ejr_cache[ek] = pbcore.ejr_restricted(inst, outs[r], kind)
                e = ejr_cache[ek]
                g = dict(n_funded=int(outs[r].sum()), spent=float(c[outs[r]].sum() / B), mean_u=float(Un[k].mean()),
                         excl=float((U[k] <= 0).mean()), ejr_n=e['n_exact'], ejr_up1_n=e['n_up1'], ejr_worst=e['worst_exact'],
                         ejr_cert=e['cert'], q_u=[float(Un[k][qv == q].mean()) for q in range(5)],
                         q_excl=[float((U[k][qv == q] <= 0).mean()) for q in range(5)])
                if small and kind in ('card', 'cost') and (r in INC or r.startswith('Sel') or
                                                          (r.startswith('DAI:') and r[4:] in MAIN.values())):
                    ex = pbcore.ejr_exact_small(inst, outs[r], kind, max_nodes=2_000_000)
                    g.update(ejrx_n=ex['n_exact'], ejrx_up1_n=ex['n_up1'], ejrx_complete=ex['complete'])
                if (agegrp != '').mean() > 0.8:
                    g['age'] = {a_: [float(Un[k][agegrp == a_].mean()), float((U[k][agegrp == a_] <= 0).mean()), int((agegrp == a_).sum())]
                                for a_ in ('<30', '30-49', '50-64', '65+') if (agegrp == a_).sum() >= 5}
                if (sex != '').mean() > 0.8:
                    g['sex'] = {a_: [float(Un[k][sex == a_].mean()), float((U[k][sex == a_] <= 0).mean()), int((sex == a_).sum())]
                                for a_ in ('M', 'F') if (sex == a_).sum() >= 5}
                if (nbh != '').mean() > 0.8 and len(set(nbh)) >= 5:
                    g['nbh'] = {str(a_): [float(Un[k][nbh == a_].mean()), float((U[k][nbh == a_] <= 0).mean()), int((nbh == a_).sum())]
                                for a_ in sorted(set(nbh)) if a_ != '' and (nbh == a_).sum() >= 20}
                K_['per_rule'][r] = g
            K_['decisive_vs_inc'] = None
            rec['kinds'][kind] = K_
        rec['q_sizes'] = [int((qv == q).sum()) for q in range(5)]
        rec['secs'] = time.time() - t0
        return rec
    except Exception:
        return dict(file=fn, error=traceback.format_exc()[-1500:])


if __name__ == '__main__':
    recs = {r['file']: r for r in json.load(open('out/stage1_instances.json'))}
    files = sorted(OUTS, key=lambda f: -recs[f]['n'] * recs[f]['m'] ** 1.5)
    if _os.environ.get('S3_MINYEAR'):  # optional restriction (temporal-split evaluation)
        files = [f for f in files if int(str(recs[f]['year'])[:4]) >= int(_os.environ['S3_MINYEAR']) and recs[f]['city'] != 'Warszawa']
    OUTNAME = _os.environ.get('S3_OUT', 'out/stage3_results.json')
    print('instances', len(files), 'DAI', len(DAI), 'MAIN', MAIN, flush=True)
    res = []
    with pbcore.make_pool(NPROC) as pool:
        for k, r in enumerate(pool.imap_unordered(work, files, chunksize=1)):
            res.append(r)
            if k % 100 == 0 or 'error' in r:
                print(k, r['file'], r.get('secs'), r.get('error', '')[:600], flush=True)
    json.dump(res, open(OUTNAME, 'w'))
    print('done', len(res), 'errors', sum('error' in r for r in res), flush=True)
