"""Revision aggregation: every number reported in the manuscript is computed here from the raw stage outputs.

Usage: python analysis.py ROOT CATALOGUE OUTDIR
  ROOT must contain out/stage1_instances.json, out/split.json, out/solver_log.json, out/stage3_results.json,
  out/bootstrap_full.json, out/dai_*.json, and optionally out/validation_pabutools.json and out/stage3_temporal.json.
All estimands are defined in the docstrings of the functions below and in Methods.
"""
import sys, os, json, glob, re, collections, math
import numpy as np
import pandas as pd

ROOT, CATF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261003)
NB = 2000  # cluster-bootstrap replicates
INC = ['GreedyAV', 'GreedyRatio', 'MaxCard', 'MaxCost', 'MES-card', 'MES-cost', 'Phragmen', 'GreedyNash']
KINDS = ['card', 'cost', 'sqrt']
MES = {'card': 'MES-card', 'cost': 'MES-cost', 'sqrt': 'MES-cost'}
N = {}  # numbers for the manuscript


def J(p):
    return json.load(open(os.path.join(ROOT, 'out', p)))


inst = {r['file']: r for r in J('stage1_instances.json')}
split = J('split.json')
s3 = [r for r in J('stage3_results.json')]
N['stage3_errors'] = sum('error' in r for r in s3)
s3 = {r['file']: r for r in s3 if 'error' not in r}
boot = {r['file']: r for r in J('bootstrap_full.json') if 'error' not in r}
DAI = {}
for f in glob.glob(os.path.join(ROOT, 'out', 'dai_*.json')):
    d = json.load(open(f)); DAI[d['tag']] = d
cat = {e['file_name']: e for e in json.load(open(CATF))}


def com(f):
    x = cat[f]['comments']
    return ' '.join(map(str, x)) if isinstance(x, list) else str(x)


# ---------------------------------------------------------------- sample definitions
E = pd.DataFrame(inst.values()).set_index('file')
E['split'] = [split.get(f, '') for f in E.index]
N['n_stage1_records'] = len(E)
E = E[E['split'] != ''].copy()  # records excluded in stage 1 (size/budget, fully fundable) carry no partition
E['yr'] = [int(str(y)[:4]) for y in E['year']]
E['event'] = [re.sub(r'_vote_(approvals|knapsacks)\.pb$', '', f) for f in E.index]
E['knapsack_dup'] = [bool(re.search(r'_vote_knapsacks\.pb$', f)) and (E['event'] == ev).sum() > 1 for f, ev in zip(E.index, E['event'])]
E['municipality'] = E['city']
E['checker_warning'] = [cat[f]['checker_status'] != 'valid' for f in E.index]
FLAG = re.compile(r'offline|paper|threshold|categor|exclusiv|extra', re.I)
E['institutional_flag'] = [bool(FLAG.search(com(f))) for f in E.index]
E['clean'] = ~E['checker_warning'] & ~E['institutional_flag']
E['heldout'] = E['split'].isin(['test_id', 'test_ood_city'])
E['in_s3'] = [f in s3 for f in E.index]
assert E['in_s3'].all(), E.index[~E['in_s3']][:5].tolist()
P = E[~E['knapsack_dup']].copy()  # primary analysis sample: one record per budgeting event
N['n_kept'] = len(E); N['n_primary'] = len(P); N['n_knapsack_dup'] = int(E['knapsack_dup'].sum())
N['n_voters_primary'] = int(P['n'].sum()); N['n_voters_kept'] = int(E['n'].sum())
N['n_municipalities_primary'] = int(P['city'].nunique()); N['n_editions_primary'] = int(P['edition'].nunique())
N['split_counts_primary'] = P['split'].value_counts().to_dict()
N['split_counts_kept'] = E['split'].value_counts().to_dict()
N['heldout_primary'] = int(P['heldout'].sum())
N['n_clean_primary'] = int(P['clean'].sum()); N['n_checker_warning_primary'] = int(P['checker_warning'].sum())
N['n_institutional_flag_primary'] = int(P['institutional_flag'].sum())
N['n_cap_gt20_primary'] = int((P['share_at_cap'] > 0.2).sum())
N['countries_primary'] = P['country'].value_counts().to_dict()
N['cities_primary_top'] = P['city'].value_counts().head(12).to_dict()
N['median_n'] = float(P['n'].median()); N['median_m'] = float(P['m'].median())
N['max_n'] = int(P['n'].max()); N['max_m'] = int(P['m'].max())
N['n_official_primary'] = int(P['has_official'].sum()); N['n_age_primary'] = int(P['has_age'].sum())
N['n_sex_primary'] = int(P['has_sex'].sum()); N['n_nbh_primary'] = int(P['has_nbh'].sum())
N['year_range'] = [int(P['yr'].min()), int(P['yr'].max())]
N['edition_year_mismatch'] = int(sum(str(E.loc[f, 'year'])[:4] not in f for f in E.index if re.search(r'_(20\d\d)', f)))
# grouped split integrity: no edition and no budgeting event crosses partitions
dev = {'train', 'val'}
cross_ed = E.groupby('edition')['split'].nunique().gt(1).sum()
cross_ev = E.groupby('event')['split'].nunique().gt(1).sum()
N['editions_crossing_partitions'] = int(cross_ed); N['events_crossing_partitions'] = int(cross_ev)
P.reset_index().to_csv(os.path.join(OUT, 'T_elections.csv'), index=False)

# ---------------------------------------------------------------- generic helpers


def cboot(df, col, cluster='city', nb=NB, alpha=0.05):
    """Cluster bootstrap of an election-level mean: resample municipalities with replacement (percentile CI)."""
    d = df[[col, cluster]].dropna()
    g = d.groupby(cluster)[col].agg(['sum', 'count'])
    s, c = g['sum'].values, g['count'].values
    est = float(d[col].mean())
    pick = rng.integers(0, len(s), size=(nb, len(s)))
    bs = s[pick].sum(1) / c[pick].sum(1)
    return est, float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2))), bs


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_two_sided(k, n):
    from math import comb
    if n == 0:
        return 1.0
    pk = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return float(min(1.0, sum(p for p in pk if p <= pk[k] * (1 + 1e-9))))


def holm(pvals):
    keys = sorted(pvals, key=lambda k: pvals[k]); m = len(keys); out = {}; run = 0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k])); out[k] = run
    return out


def pr(rec, kind, rule):
    """per-rule audit record, following same_as links (identical outcomes are audited once)."""
    d = rec['kinds'][kind]['per_rule'][rule]
    while 'same_as' in d:
        d = rec['kinds'][kind]['per_rule'][d['same_as']]
    return d


def league(rec, kind, rule, n):
    """Common-eight league share: mean over the eight incumbents b of [#prefer rule + 0.5 #indifferent]/n.
    Self-comparison (rule == b, or identical outcome) contributes exactly 0.5."""
    K = rec['kinds'][kind]; k = rec['rules'].index(rule)
    ab = np.array(K['vs_ab'][k], float); ba = np.array(K['vs_ba'][k], float)
    return float(np.mean((ab + 0.5 * (n - ab - ba)) / n))


def decisive(rec, kind, rule):
    K = rec['kinds'][kind]; k = rec['rules'].index(rule)
    ab = np.array(K['vs_ab'][k], float); ba = np.array(K['vs_ba'][k], float)
    return float(ab.sum() / (ab + ba).sum()) if (ab + ba).sum() > 0 else np.nan


def contests(rec, kind, rule):
    """win/loss/tie counts of rule against each incumbent whose outcome differs for at least one voter."""
    K = rec['kinds'][kind]; k = rec['rules'].index(rule)
    w = l = t = 0
    for b, rb in enumerate(INC):
        if rb == rule:
            continue
        ab, ba = K['vs_ab'][k][b], K['vs_ba'][k][b]
        if ab + ba == 0:
            continue
        w += ab > ba; l += ab < ba; t += ab == ba
    return w, l, t


def rule_table(files, kind, rules):
    rows = []
    for f in files:
        rec = s3[f]; n = inst[f]['n']
        for r in rules:
            if r not in rec['rules']:
                continue
            a = pr(rec, kind, r); w, l, t = contests(rec, kind, r)
            q = a['q_u']
            rows.append(dict(file=f, city=inst[f]['city'], edition=inst[f]['edition'], n=n, rule=r, L=league(rec, kind, r, n),
                             dec=decisive(rec, kind, r), win=w, loss=l, tie=t, mean_u=a['mean_u'], excl=a['excl'],
                             q1_u=q[0], q1_excl=a['q_excl'][0], q1_ratio=q[0] / a['mean_u'] if a['mean_u'] > 0 else np.nan,
                             ejr=int(a['ejr_n'] > 0), ejr1=int(a['ejr_up1_n'] > 0), spent=a['spent'], n_funded=a['n_funded'],
                             ejrx=(int(a['ejrx_n'] > 0) if a.get('ejrx_complete') else np.nan),
                             ejrx1=(int(a['ejrx_up1_n'] > 0) if a.get('ejrx_complete') else np.nan)))
    return pd.DataFrame(rows)


def summarize(T, by='rule', cluster=True):
    out = []
    for r, g in T.groupby(by, sort=False):
        d = dict(rule=r, n_elections=len(g), L=g.L.mean(), L_vw=np.average(g.L, weights=g.n), dec=g.dec.mean(),
                 win_rate=g.win.sum() / max(1, (g.win + g.loss + g.tie).sum()), wins=int(g.win.sum()),
                 contests=int((g.win + g.loss + g.tie).sum()), mean_u=g.mean_u.mean(), excl=g.excl.mean(),
                 q1_ratio_mean=g.q1_ratio.mean(), q1_ratio_rm=g.q1_u.mean() / g.mean_u.mean(), q1_excl=g.q1_excl.mean(),
                 ejr_rate=g.ejr.mean(), ejr_k=int(g.ejr.sum()), ejr1_rate=g.ejr1.mean(), ejr1_k=int(g.ejr1.sum()),
                 ejrx_rate=g.ejrx.mean(), ejrx_n=int(g.ejrx.notna().sum()), spent=g.spent.mean())
        if cluster and g.city.nunique() > 1:
            e, lo, hi, _ = cboot(g, 'L', nb=500)
            d.update(L_lo=lo, L_hi=hi)
        out.append(d)
    return pd.DataFrame(out)


def paired(T, a, b, col, files=None, nb=NB, cluster='city'):
    """paired election-level difference mean(col[a] - col[b]) with municipality-cluster bootstrap CI and sign-flip P."""
    A = T[T.rule == a].set_index('file'); Bt = T[T.rule == b].set_index('file')
    ix = A.index.intersection(Bt.index)
    D = pd.DataFrame(dict(d=A.loc[ix, col] - Bt.loc[ix, col], city=A.loc[ix, cluster], n=A.loc[ix, 'n'])).dropna()
    est, lo, hi, bs = cboot(D, 'd', nb=nb)
    # cluster sign-flip randomisation test (cluster-level sums)
    cs = D.groupby('city').d.sum().values; obs = abs(cs.sum())
    flips = rng.choice([-1, 1], size=(20000, len(cs)))
    p = float(((np.abs(flips @ cs) >= obs - 1e-12).sum() + 1) / (20000 + 1))
    return dict(a=a, b=b, col=col, n=len(D), n_clusters=int(D.city.nunique()), diff=float(est), lo=lo, hi=hi, p=p,
                diff_vw=float(np.average(D.d, weights=D.n)))


# ---------------------------------------------------------------- 1. rule-level elections among incumbents
clsrows = []; memb = []
for kind in KINDS:
    for f in P.index:
        el = s3[f]['kinds'][kind]['elections']['incumbents']
        lab = el['label']; cwm = el['classes'][el['cw']] if el.get('cw') is not None and lab in ('strict_cw', 'single') else []
        weak = sorted({r for i in (el.get('weak') or []) for r in el['classes'][i]})
        smith = sorted({r for i in (el.get('smith') or []) for r in el['classes'][i]})
        b = boot.get(f, {}).get('boot', {}).get(kind)
        nbt = boot.get(f, {}).get('nboot', np.nan)
        clsrows.append(dict(file=f, city=inst[f]['city'], n=inst[f]['n'], kind=kind, label=lab, K=len(el['classes']),
                            n_smith=len(el.get('smith') or []),
                            boot_same=(b['same_cw'] / nbt if b else np.nan), boot_exists=(b['cw_exists'] / nbt if b else np.nan),
                            boot_overlap=(b['overlap_cw'] / nbt if b else np.nan),
                            fixed_same=el.get('cond_boot_same', np.nan)))
        for r in INC:
            memb.append(dict(file=f, city=inst[f]['city'], kind=kind, rule=r, cw=int(r in cwm), weak=int(r in weak), smith=int(r in smith),
                             multi=int(lab != 'single'), strict=int(lab == 'strict_cw'),
                             boot_member=(b['member'][r] / nbt if b else np.nan),
                             ml=float(sum(el['ml'][i] for i, c in enumerate(el['classes']) if r in c))))
CL = pd.DataFrame(clsrows); MB = pd.DataFrame(memb)
CL.to_csv(os.path.join(OUT, 'T_election_labels.csv'), index=False)
for kind in KINDS:
    g = CL[CL.kind == kind]; multi = g[g.label != 'single']
    cnt = g.label.value_counts().to_dict()
    k = int((multi.label == 'strict_cw').sum()); nm = len(multi)
    est, lo, hi, _ = cboot(multi.assign(x=(multi.label == 'strict_cw').astype(float)), 'x')
    sc = multi[multi.label == 'strict_cw']
    N[f'elec_{kind}'] = dict(counts=cnt, n_multi=nm, strict_cw=k, strict_cw_pct=100 * k / nm, wilson=[100 * x for x in wilson(k, nm)],
                             cluster_ci=[100 * lo, 100 * hi], cycles=int(cnt.get('cycle', 0)), weak_only=int(cnt.get('weak_cw', 0)),
                             tie_only=int(cnt.get('tie_only', 0)),
                             vw_strict_cw_pct=100 * np.average(multi.label == 'strict_cw', weights=multi.n),
                             boot_same_mean=float(sc.boot_same.mean()), boot_exists_mean=float(sc.boot_exists.mean()),
                             boot_same_ge95=float((sc.boot_same >= 0.95).mean()), boot_overlap_mean=float(sc.boot_overlap.mean()),
                             fixed_same_mean=float(sc.fixed_same.mean()), n_boot_elections=int(sc.boot_same.notna().sum()))
mrows = []
for (kind, r), g in MB[MB.multi == 1].groupby(['kind', 'rule'], sort=False):
    mrows.append(dict(kind=kind, rule=r, cw_pct=100 * g.cw.mean(), weak_pct=100 * g.weak.mean(), smith_pct=100 * g.smith.mean(),
                      boot_member=100 * g.boot_member.mean(), ml_mass=g.ml.mean()))
MT = pd.DataFrame(mrows); MT.to_csv(os.path.join(OUT, 'T_cw_membership.csv'), index=False)
N['cw_membership'] = {f'{a}|{b}': round(c, 2) for a, b, c in MT[['kind', 'rule', 'cw_pct']].itertuples(index=False)}


# pairwise contest matrix among incumbents (and Official where recorded), all primary elections
for kind in KINDS:
    rows = []
    cand = INC + ['Official']
    for f in P.index:
        rec = s3[f]; K = rec['kinds'][kind]
        for ra in cand:
            if ra not in rec['rules']:
                continue
            ka = rec['rules'].index(ra)
            for b, rb in enumerate(INC):
                if rb == ra:
                    continue
                ab, ba = K['vs_ab'][ka][b], K['vs_ba'][ka][b]
                if ab + ba == 0:
                    continue
                rows.append((ra, rb, int(ab > ba), int(ab < ba)))
    PW = pd.DataFrame(rows, columns=['row', 'col', 'win', 'loss'])
    PW = PW.groupby(['row', 'col']).agg(win=('win', 'sum'), loss=('loss', 'sum'), contests=('win', 'size')).reset_index()
    PW['win_pct'] = 100 * PW.win / PW.contests
    PW.to_csv(os.path.join(OUT, f'T_pairwise_{kind}.csv'), index=False)

# cycle correlates (strict cycles, cost utility): logistic regression with municipality-clustered SEs
try:
    import statsmodels.api as sm
    feats = ['jaccard', 'lam2', 'm', 'n', 'budget_share', 'mean_ballot', 'gini_app']
    for kind in ['card', 'cost']:
        g = CL[(CL.kind == kind) & (CL.label != 'single')].set_index('file')
        X = P.loc[g.index, feats].astype(float).copy(); X['m'] = np.log(X['m']); X['n'] = np.log(X['n'])
        X = (X - X.mean()) / X.std(); X = sm.add_constant(X)
        y = (g.label == 'cycle').astype(float)
        if y.sum() >= 5:
            fit = sm.Logit(y, X).fit(disp=0, cov_type='cluster', cov_kwds=dict(groups=pd.factorize(g.city)[0]))
            N[f'cycle_logit_{kind}'] = {v: dict(OR=float(np.exp(fit.params[v])), lo=float(np.exp(fit.conf_int().loc[v, 0])),
                                                hi=float(np.exp(fit.conf_int().loc[v, 1])), p=float(fit.pvalues[v])) for v in feats}
            N[f'cycle_logit_{kind}']['n'] = int(len(y)); N[f'cycle_logit_{kind}']['events'] = int(y.sum())
except Exception as ex:  # pragma: no cover
    N['cycle_logit_error'] = repr(ex)

# ---------------------------------------------------------------- 2. MES vs GreedyAV (equivalence)
for kind in ['card', 'cost', 'sqrt']:
    rows = []
    mi = INC.index(MES[kind]); gi = INC.index('GreedyAV')
    for f in P.index:
        K = s3[f]['kinds'][kind]; n = inst[f]['n']; ab = K['vs_ab'][mi][gi]; ba = K['vs_ba'][mi][gi]
        rows.append(dict(file=f, city=inst[f]['city'], n=n, s=(ab + 0.5 * (n - ab - ba)) / n, win=int(ab > ba), dist=int(ab + ba > 0)))
    D = pd.DataFrame(rows); Dd = D[D.dist == 1]
    est, lo, hi, _ = cboot(D, 's', alpha=0.10)
    _, lo95, hi95, _ = cboot(D, 's')
    w = int(Dd.win.sum())
    N[f'mes_vs_gav_{kind}'] = dict(share=est, ci90=[lo, hi], ci95=[lo95, hi95], equivalent_2pp=bool(lo > 0.48 and hi < 0.52),
                                   win=w, contests=len(Dd), win_pct=100 * w / max(1, len(Dd)), p_sign=binom_two_sided(w, len(Dd)),
                                   share_vw=float(np.average(D.s, weights=D.n)))

# ---------------------------------------------------------------- 3. learned rules on held-out elections
MAIN = next(iter(s3.values()))['main']
N['main_tags'] = MAIN
H = P[P['heldout']].index.tolist()
HW = P[P['split'] == 'test_ood_city'].index.tolist(); HI = P[P['split'] == 'test_id'].index.tolist()
N['n_heldout'] = len(H); N['n_heldout_warsaw'] = len(HW); N['n_heldout_id'] = len(HI)
tabs = {}; prim = {}
for kind in KINDS:
    if kind not in MAIN:
        continue
    dai = 'DAI:' + MAIN[kind]
    tags = ['DAI:' + t for t, d in DAI.items() if d['kind'] == kind and not d.get('hold')]
    rules = INC + ['Sel:' + kind, 'SelEJR:' + kind] + sorted(tags)
    T = rule_table(H, kind, rules); tabs[kind] = T
    S = summarize(T); S.insert(0, 'kind', kind); S.to_csv(os.path.join(OUT, f'T_heldout_{kind}.csv'), index=False)
    N[f'heldout_{kind}'] = S.set_index('rule').round(5).to_dict(orient='index')
    for sub, fl in [('warsaw', HW), ('id', HI)]:
        if not T.file.isin(fl).any():
            continue
        Ss = summarize(T[T.file.isin(fl)], cluster=False).set_index('rule')
        N[f'heldout_{kind}_{sub}'] = Ss.loc[[dai, 'Sel:' + kind, 'SelEJR:' + kind, MES[kind], 'GreedyAV'], ['L', 'win_rate', 'ejr_rate', 'q1_ratio_mean']].round(5).to_dict(orient='index')
    comps = {}
    for rival in [MES[kind], 'GreedyAV', 'Sel:' + kind, 'SelEJR:' + kind]:
        for col in ['L', 'q1_ratio', 'q1_excl', 'excl', 'ejr', 'ejr1']:
            comps[f'{rival}|{col}'] = paired(T, dai, rival, col)
    prim[kind] = comps
    N[f'sel_minus_selejr_{kind}'] = dict(municipality=paired(T, 'Sel:' + kind, 'SelEJR:' + kind, 'L'),
                                         edition=paired(T, 'Sel:' + kind, 'SelEJR:' + kind, 'L', cluster='edition'))
    N[f'paired_edition_{kind}'] = {f'{r}|{c}': paired(T, dai, r, c, cluster='edition') for r in [MES[kind], 'GreedyAV', 'Sel:' + kind, 'SelEJR:' + kind]
                                   for c in ['L', 'q1_ratio', 'ejr']}
    # McNemar-type exact test on EJR violations, DAI vs MES (election-level discordance)
    A = T[T.rule == dai].set_index('file'); M_ = T[T.rule == MES[kind]].set_index('file'); Sx = T[T.rule == 'Sel:' + kind].set_index('file')
    for nm, R in [('mes', M_), ('sel', Sx)]:
        for col in ['ejr', 'ejr1']:
            b10 = int(((A[col] == 1) & (R.loc[A.index, col] == 0)).sum()); b01 = int(((A[col] == 0) & (R.loc[A.index, col] == 1)).sum())
            N[f'ejr_discord_{kind}_{nm}_{col}'] = dict(dai_only=b10, rival_only=b01, p=binom_two_sided(min(b10, b01), b10 + b01))
    # seed-to-seed variability of the main learned rule
    seeds = [t for t in tags if '_league_s' in t]
    N[f'seeds_{kind}'] = {t: dict(L=float(T[T.rule == t].L.mean()), ejr=float(T[T.rule == t].ejr.mean()),
                                  q1_ratio=float(T[T.rule == t].q1_ratio.mean()), best_val=DAI[t[4:]]['best_val']) for t in seeds}
    # selector choices
    ch = collections.Counter(s3[f]['selector_choice'][kind]['sel'] for f in H)
    chE = collections.Counter(s3[f]['selector_choice'][kind]['selejr'] for f in H)
    N[f'selector_choice_{kind}'] = dict(sel=dict(ch), selejr=dict(chE))
    # voters: pooled decisive-share by mainstream quintile (mechanism decomposition)
    dec = {}
    for rival in [MES[kind], 'GreedyAV', 'SelEJR:' + kind, 'Sel:' + kind]:
        q_el = np.full((len(H), 5), np.nan); pooled = np.zeros((5, 3))
        for i, f in enumerate(H):
            dd = s3[f]['kinds'][kind]['decomp'].get(rival)
            if not dd:
                continue
            dd = np.array(dd, float); pooled += dd
            with np.errstate(invalid='ignore', divide='ignore'):
                q_el[i] = dd[:, 0] / (dd[:, 0] + dd[:, 1])
        dec[rival] = dict(pooled_prefer_dai=(pooled[:, 0] / pooled.sum(1)).tolist(), pooled_prefer_rival=(pooled[:, 1] / pooled.sum(1)).tolist(),
                          pooled_indiff=(pooled[:, 2] / pooled.sum(1)).tolist(), election_mean_decisive=np.nanmean(q_el, 0).tolist())
    N[f'decomp_{kind}'] = dec
fam = {}
for kind in prim:
    for k, v in prim[kind].items():
        if k.endswith('|L') or k.endswith('|q1_ratio') or k.endswith('|ejr'):
            fam[f'{kind}|{k}'] = v['p']
hp = holm(fam)
for kind in prim:
    for k, v in prim[kind].items():
        v['p_holm'] = hp.get(f'{kind}|{k}', np.nan)
N['paired'] = prim
pd.DataFrame([dict(kind=k, key=kk, **v) for k in prim for kk, v in prim[k].items()]).to_csv(os.path.join(OUT, 'T_paired.csv'), index=False)

# exact EJR audit (small elections, all subsets): completeness of the restricted |T|<=2 audit
rows = []
for kind in ['card', 'cost']:
    for f in P.index:
        rec = s3[f]
        if not rec['small']:
            continue
        for r in rec['rules']:
            if not (r in INC or r.startswith('Sel') or (r.startswith('DAI:') and r[4:] in MAIN.values())):
                continue
            a = pr(rec, kind, r)
            if not a.get('ejrx_complete'):
                continue
            rows.append(dict(kind=kind, file=f, rule=r, restricted=int(a['ejr_n'] > 0), exact=int(a['ejrx_n'] > 0),
                             restricted1=int(a['ejr_up1_n'] > 0), exact1=int(a['ejrx_up1_n'] > 0), heldout=f in H))
EX = pd.DataFrame(rows)
if len(EX):
    for kind in ['card', 'cost']:
        g = EX[EX.kind == kind]
        N[f'exact_audit_{kind}'] = dict(n_rule_elections=len(g), n_elections=int(g.file.nunique()),
                                        exact_viol=int(g.exact.sum()), restricted_viol=int(g.restricted.sum()),
                                        missed_by_restricted=int(((g.exact == 1) & (g.restricted == 0)).sum()),
                                        exact1_viol=int(g.exact1.sum()), missed1=int(((g.exact1 == 1) & (g.restricted1 == 0)).sum()),
                                        by_rule={r: [int(x.exact.sum()), int(x.exact1.sum()), len(x)] for r, x in g.groupby('rule')})
    N['exact_small_total_elections'] = int(EX.file.nunique())
N['n_small_elections_primary'] = int(sum(s3[f]['small'] for f in P.index))

# ---------------------------------------------------------------- 4. demographic and neighbourhood audits (held-out only)
dem = []
for kind in ['card', 'cost']:
    if kind not in MAIN:
        continue
    dai = 'DAI:' + MAIN[kind]
    for f in H:
        rec = s3[f]
        for r in [dai, MES[kind], 'GreedyAV', 'Sel:' + kind, 'SelEJR:' + kind]:
            a = pr(rec, kind, r)
            for grp in ['age', 'sex']:
                for gname, v in (a.get(grp) or {}).items():
                    dem.append(dict(kind=kind, file=f, city=inst[f]['city'], rule=r, dim=grp, group=gname, u=v[0], excl=v[1], size=v[2], mean_u=a['mean_u']))
            if a.get('nbh'):
                vals = np.array([v[0] for v in a['nbh'].values()])
                dem.append(dict(kind=kind, file=f, city=inst[f]['city'], rule=r, dim='nbh', group='worst_ratio', u=float(vals.min() / a['mean_u']) if a['mean_u'] > 0 else np.nan,
                                excl=np.nan, size=len(vals), mean_u=a['mean_u']))
DM = pd.DataFrame(dem); DM.to_csv(os.path.join(OUT, 'T_demographic_rows.csv'), index=False)
dsum = {}
if len(DM):
    for (kind, dim, grp), g in DM.groupby(['kind', 'dim', 'group']):
        g = g.assign(rel=g.u / g.mean_u if dim != 'nbh' else g.u)
        piv = g.pivot_table(index=['file', 'city'], columns='rule', values='rel').reset_index()
        dai = 'DAI:' + MAIN[kind]
        out = dict(n_elections=int(len(piv)), n_cities=int(piv.city.nunique()))
        for r in [dai, MES[kind], 'GreedyAV', 'Sel:' + kind, 'SelEJR:' + kind]:
            if r in piv:
                out[r] = float(piv[r].mean())
        for rival in [MES[kind], 'Sel:' + kind]:
            if rival in piv and dai in piv and len(piv) > 3 and piv.city.nunique() > 1:
                D = piv.assign(d=piv[dai] - piv[rival])
                e, lo, hi, _ = cboot(D, 'd', nb=1000)
                out[f'diff_vs_{rival}'] = [float(e), lo, hi]
        dsum[f'{kind}|{dim}|{grp}'] = out
N['demographics_heldout'] = dsum

# ---------------------------------------------------------------- 5. official outcomes: relaxed descriptive benchmark
for kind in ['card', 'cost']:
    fl = [f for f in P.index if 'Official' in s3[f]['rules']]
    T = rule_table(fl, kind, ['Official', 'GreedyAV', MES[kind]])
    S = summarize(T, cluster=False).set_index('rule')
    N[f'official_{kind}'] = S[['n_elections', 'L', 'win_rate', 'ejr_rate', 'q1_ratio_mean', 'spent']].round(5).to_dict(orient='index')
    N[f'official_{kind}_clean'] = summarize(T[T.file.isin(P.index[P.clean])], cluster=False).set_index('rule')[['n_elections', 'L', 'win_rate']].round(5).to_dict(orient='index')
    eqg = [f for f in fl if pr(s3[f], kind, 'Official') is pr(s3[f], kind, 'GreedyAV')]
    N[f'official_equals_greedy_{kind}'] = len(eqg)
N['official_meta_rules'] = P.loc[[f for f in P.index if 'Official' in s3[f]['rules']], 'meta_rule'].value_counts().to_dict()

# ---------------------------------------------------------------- 6. leave-one-municipality-out generalisation
lomo = []
for t, d in DAI.items():
    if not d.get('hold') or d['hold'] == 'TEMPORAL':
        continue
    kind = d['kind']; Hf = P[P.city == d['hold']].index.tolist()
    if not Hf:
        continue
    T = rule_table(Hf, kind, ['DAI:' + t, 'Sel:' + kind, 'SelEJR:' + kind, MES[kind], 'GreedyAV', 'DAI:' + MAIN[kind]])
    S = summarize(T, cluster=False).set_index('rule')
    lomo.append(dict(kind=kind, held=d['hold'], n=len(Hf), n_train=d['n_train'], L_lomo=S.loc['DAI:' + t, 'L'], win_lomo=S.loc['DAI:' + t, 'win_rate'],
                     L_sel=S.loc['Sel:' + kind, 'L'], L_selejr=S.loc['SelEJR:' + kind, 'L'], L_mes=S.loc[MES[kind], 'L'], L_gav=S.loc['GreedyAV', 'L'],
                     ejr_lomo=S.loc['DAI:' + t, 'ejr_rate'], ejr_mes=S.loc[MES[kind], 'ejr_rate'], ejr_sel=S.loc['Sel:' + kind, 'ejr_rate'],
                     q1_lomo=S.loc['DAI:' + t, 'q1_ratio_mean'], q1_mes=S.loc[MES[kind], 'q1_ratio_mean']))
LM = pd.DataFrame(lomo).sort_values(['kind', 'n'], ascending=[True, False]) if lomo else pd.DataFrame()
LM.to_csv(os.path.join(OUT, 'T_lomo.csv'), index=False)
N['lomo'] = LM.round(5).to_dict(orient='records')

# ---------------------------------------------------------------- 7. temporal split (if available)
tp = os.path.join(ROOT, 'out', 'stage3_temporal.json')
if os.path.exists(tp):
    t3 = {r['file']: r for r in json.load(open(tp)) if 'error' not in r}
    tmp = []
    for kind in ['card', 'cost']:
        tag = f'DAI:{kind}_temporal'
        fl = [f for f in P.index if f in t3 and P.loc[f, 'yr'] >= 2023 and P.loc[f, 'city'] != 'Warszawa']
        if not fl:
            continue
        save = dict(s3)
        s3.update(t3)
        T = rule_table(fl, kind, [tag, 'Sel:' + kind, 'SelEJR:' + kind, MES[kind], 'GreedyAV'])
        s3.clear(); s3.update(save)
        S = summarize(T).set_index('rule')
        N[f'temporal_{kind}'] = S[['n_elections', 'L', 'L_lo', 'L_hi', 'win_rate', 'ejr_rate', 'ejr1_rate', 'q1_ratio_mean', 'excl']].round(5).to_dict(orient='index')
        N[f'temporal_{kind}_paired'] = {r: paired(T, tag, r, 'L') for r in ['Sel:' + kind, MES[kind], 'GreedyAV']}
        tk = tag[4:]
        if tk in DAI:
            N[f'temporal_{kind}_ntrain'] = [DAI[tk]['n_train'], DAI[tk]['n_val']]

# ---------------------------------------------------------------- 8. sensitivity of the primary held-out contrasts
sens = []
for kind in ['card', 'cost']:
    if kind not in tabs:
        continue
    T = tabs[kind]; dai = 'DAI:' + MAIN[kind]
    subsets = {'primary': H, 'cap<=20%': [f for f in H if inst[f]['share_at_cap'] <= 0.2], 'clean': [f for f in H if P.loc[f, 'clean']],
               'warsaw': HW, 'id_only': HI, 'n>5000': [f for f in H if inst[f]['n'] > 5000]}
    Tdup = rule_table([f for f in E.index if E.loc[f, 'heldout']], kind, [dai, 'Sel:' + kind, MES[kind]])
    for nm, fl in subsets.items():
        g = T[T.file.isin(fl)]
        if not len(g):
            continue
        S = summarize(g, cluster=False).set_index('rule')
        sens.append(dict(kind=kind, subset=nm, n=len(fl), L_dai=S.loc[dai, 'L'], L_sel=S.loc['Sel:' + kind, 'L'], L_mes=S.loc[MES[kind], 'L'],
                         ejr_dai=S.loc[dai, 'ejr_rate'], ejr_sel=S.loc['Sel:' + kind, 'ejr_rate'], ejr_mes=S.loc[MES[kind], 'ejr_rate'],
                         q1_dai=S.loc[dai, 'q1_ratio_mean'], q1_mes=S.loc[MES[kind], 'q1_ratio_mean'], win_dai=S.loc[dai, 'win_rate']))
    S = summarize(Tdup, cluster=False).set_index('rule')
    sens.append(dict(kind=kind, subset='with_knapsack_duplicates', n=int(E['heldout'].sum()), L_dai=S.loc[dai, 'L'], L_sel=S.loc['Sel:' + kind, 'L'],
                     L_mes=S.loc[MES[kind], 'L'], ejr_dai=S.loc[dai, 'ejr_rate'], ejr_sel=S.loc['Sel:' + kind, 'ejr_rate'],
                     ejr_mes=S.loc[MES[kind], 'ejr_rate'], q1_dai=S.loc[dai, 'q1_ratio_mean'], q1_mes=S.loc[MES[kind], 'q1_ratio_mean'],
                     win_dai=S.loc[dai, 'win_rate']))
SN = pd.DataFrame(sens); SN.to_csv(os.path.join(OUT, 'T_sensitivity.csv'), index=False)
N['sensitivity'] = SN.round(5).to_dict(orient='records')
# sensitivity of the structural (incumbent) results to caps and data quality
for kind in ['card', 'cost']:
    g = CL[(CL.kind == kind) & (CL.label != 'single')]
    out = {}
    for nm, mask in [('cap<=20%', g.file.map(lambda f: inst[f]['share_at_cap'] <= 0.2)), ('clean', g.file.map(lambda f: bool(P.loc[f, 'clean'])))]:
        gg = g[mask.values]
        out[nm] = dict(n=len(gg), strict_cw_pct=100 * (gg.label == 'strict_cw').mean(), cycles=int((gg.label == 'cycle').sum()))
    N[f'elec_sens_{kind}'] = out

# ---------------------------------------------------------------- 9. solver certificates, validation, training logs
sl = J('solver_log.json')
N['solver'] = dict(n=len(sl), status1=dict(collections.Counter(str(s['status1']) for s in sl)), status2=dict(collections.Counter(str(s['status2']) for s in sl)),
                   max_gap1=float(max(abs(s['gap1'] or 0) for s in sl)), primary_kept=int(sum(bool(s['primary_kept']) for s in sl)))
vp = os.path.join(ROOT, 'out', 'validation_pabutools.json')
if os.path.exists(vp):
    VR = pd.DataFrame(json.load(open(vp)))
    N['validation'] = {r: dict(n=len(g), identical=int(g.identical.sum()), ref_subset=int(g.ref_subset_of_me.sum()),
                               extra_projects_max=int((g.n_me - g.n_ref).max()), welfare_rel_diff_max=float(g.welfare_rel_diff.abs().max()))
                       for r, g in VR.groupby('rule')}
    N['validation_n_elections'] = int(VR.file.nunique())
N['training'] = {t: dict(kind=d['kind'], rivals=d['rivals'], lam=d['lam'], hold=d.get('hold', ''), n_train=d['n_train'], n_val=d['n_val'],
                         best_gen=d['best_gen'], best_val=d['best_val'], final_val=d['log'][-1]['val_obj'], secs=d['log'][-1]['secs'])
                 for t, d in sorted(DAI.items())}


# ---------------------------------------------------------------- 10. sample flow and derived descriptive numbers
X = {}
appr = [e for e in cat.values() if str(e['vote_type']) == 'approval' and str(e['experimental']) != 'True']
X['catalogue'] = len(cat); X['approval_nonexp'] = len(appr)
X['checker_invalid'] = sum(1 for e in appr if e['checker_status'] not in ('valid', 'valid_with_warning'))
S1 = pd.DataFrame(J('stage1_instances.json')); ex = S1[~S1.file.isin(E.index)]
small = (ex.n < 10) | (ex.m < 3) | (ex.B <= 0)
X['excl_size_budget'] = int(small.sum()); X['excl_fully_fundable'] = int((~small & (ex.total_cost <= ex.B)).sum())
nv = pd.Series({f: int(str(cat[f]['num_votes_raw']).replace(' ', '')) for f in P.index})
X['ballots_recorded_primary'] = int(nv.sum()); X['empty_or_dropped_ballots_primary'] = int((nv - P['n']).sum())
X['currencies'] = dict(collections.Counter(cat[f]['currency'] for f in P.index))
X['n_capped'] = int((P['max_length'].fillna(0) > 0).sum()); X['share_at_cap_median'] = float(P['share_at_cap'].median())
X['share_at_cap_median_capped'] = float(P.loc[P['max_length'].fillna(0) > 0, 'share_at_cap'].median())
for k in KINDS:
    g = CL[(CL.kind == k) & (CL.label != 'single')].copy(); g['jac'] = P.loc[g.file, 'jaccard'].values
    g['q'] = pd.qcut(g.jac, 4, labels=False)
    X[f'cycle_by_jacq_{k}'] = [float(100 * (x.label == 'cycle').mean()) for _, x in g.groupby('q')]
if len(LM):
    LMx = LM.assign(d=LM.L_lomo - LM.L_sel)
    X['lomo_sel_wins'] = int((LMx.d < 0).sum()); X['lomo_n'] = int(len(LMx))
    for k in ['card', 'cost']:
        g = LMx[LMx.kind == k]
        X[f'lomo_d_{k}'] = [float(np.average(g.d, weights=g.n)), float(g.d.min()), float(g.d.max())]
        X[f'lomo_mes_d_{k}'] = float(np.average(g.L_lomo - g.L_mes, weights=g.n))
        X[f'lomo_ejr_{k}'] = [float(np.average(g[c], weights=g.n)) for c in ['ejr_lomo', 'ejr_mes', 'ejr_sel']]
N['extra'] = X

json.dump(N, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=float)
print('analysis done', len(N), 'keys')
