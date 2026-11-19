"""Core library (v2, revision) for model-implied rule-level elections on Pabulib data.

Changes from v1 (in response to review): greedy tie-breaking follows the stated specification for every score;
utilitarian optima are solved lexicographically (primary objective to optimality, then tie-break) with solver
status recorded; a third utility model (square-root cost) is available; EJR audits are generalised to additive
project utilities, report EJR up to one project as well as exact EJR, return witness certificates, and an exact
(all-subset) audit is provided for small elections; majority graphs are classified into strict Condorcet winner,
weak Condorcet winner only, strict cycle, and tie-only cases.
"""
import csv, io, math
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds, linprog

csv.field_size_limit(10**8)
KINDS = ('card', 'cost', 'sqrt')


# --------------------------------------------------------------------- parsing
def parse_pb(text):
    """Return dict(meta, projects, votes) from a Pabulib .pb file."""
    meta, projects, votes = {}, [], []
    section, header = None, None
    for row in csv.reader(io.StringIO(text), delimiter=';'):
        if not row:
            continue
        key = row[0].strip()
        if key in ('META', 'PROJECTS', 'VOTES') and len(row) == 1:
            section, header = key, None
            continue
        if header is None:
            header = [h.strip() for h in row]
            continue
        rec = dict(zip(header, [r.strip() for r in row]))
        if section == 'META':
            meta[row[0].strip()] = row[1].strip() if len(row) > 1 else ''
        elif section == 'PROJECTS':
            projects.append(rec)
        elif section == 'VOTES':
            votes.append(rec)
    return dict(meta=meta, projects=projects, votes=votes)


def build_instance(raw, name=''):
    """Compact numeric instance.  A is a CSR n x m 0/1 approval matrix (empty ballots dropped)."""
    meta, P, V = raw['meta'], raw['projects'], raw['votes']
    pid = [p['project_id'] for p in P]
    idx = {k: j for j, k in enumerate(pid)}
    cost = np.array([float(str(p['cost']).replace(',', '.')) for p in P])
    rows, cols, age, sex, nbh = [], [], [], [], []
    for i, v in enumerate(V):
        ballot = [b.strip() for b in v.get('vote', '').split(',') if b.strip() != '']
        js = sorted({idx[b] for b in ballot if b in idx})
        rows += [i] * len(js); cols += js
        age.append(v.get('age', '')); sex.append(v.get('sex', '')); nbh.append(v.get('neighborhood', ''))
    n, m = len(V), len(P)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, m))
    keep = np.asarray(A.sum(1)).ravel() > 0
    A = A[keep]
    age = np.array(age, dtype=object)[keep]; sex = np.array(sex, dtype=object)[keep]
    nbh = np.array(nbh, dtype=object)[keep]
    sel = np.array([p.get('selected', '') for p in P], dtype=object)
    cat = [p.get('category', '') for p in P]
    budget = float(str(meta.get('budget', 'nan')).replace(',', '.'))
    return dict(name=name, meta=meta, A=A, Ac=A.tocsc(), cost=cost, B=budget, n=A.shape[0], m=m,
                pid=pid, age=age, sex=sex, nbh=nbh, selected=sel, category=cat)


# --------------------------------------------------------------- utilities
def proj_util(inst, kind):
    """Utility a supporter derives from each funded project: 1 (count), cost, or sqrt(cost)."""
    c = inst['cost']
    if kind == 'card':
        return np.ones(inst['m'])
    if kind == 'cost':
        return c.copy()
    if kind == 'sqrt':
        return np.sqrt(c)
    raise ValueError(kind)


def utilities(inst, W, kind):
    """Per-voter additive utility of funded set W (bool m-vector)."""
    return inst['A'] @ (W.astype(float) * proj_util(inst, kind))


def ballot_norm(inst, kind):
    """Per-voter utility of the full approval set (the normaliser: utility as a share of the ballot)."""
    return inst['A'] @ proj_util(inst, kind)


def _approvals(inst):
    return np.asarray(inst['Ac'].sum(0)).ravel()


def _rank(inst):
    """Deterministic tie-breaking rank: more approvals, then cheaper, then file order."""
    app = _approvals(inst); m = inst['m']
    rank = np.empty(m, int)
    rank[np.lexsort((np.arange(m), inst['cost'], -app))] = np.arange(m)
    return rank


# ------------------------------------------------------------- incumbent rules
def greedy_by_score(inst, score):
    """Fund in decreasing score; ties by more approvals, then lower cost, then file order; skip what does not fit."""
    B, c = inst['B'], inst['cost']
    order = np.lexsort((np.arange(inst['m']), c, -_approvals(inst), -score))
    W = np.zeros(inst['m'], bool); spent = 0.0
    for j in order:
        if spent + c[j] <= B + 1e-9:
            W[j] = True; spent += c[j]
    return W


def rule_greedy_av(inst):
    return greedy_by_score(inst, _approvals(inst).astype(float))


def rule_greedy_ratio(inst):
    return greedy_by_score(inst, _approvals(inst) / np.maximum(inst['cost'], 1e-9))


def _complete_greedy(inst, W):
    W = W.copy(); c, B = inst['cost'], inst['B']
    order = np.argsort(_rank(inst))
    spent = c[W].sum()
    for j in order:
        if not W[j] and spent + c[j] <= B + 1e-9:
            W[j] = True; spent += c[j]
    return W


SOLVER_LOG = []


def rule_max_util(inst, kind):
    """Exact utilitarian optimum, solved lexicographically: (1) maximise sum_i u_i(W) to optimality;
    (2) among optima, maximise total approvals (then greedy completion with zero-value projects only)."""
    app = _approvals(inst).astype(float)
    c, B, m = inst['cost'], inst['B'], inst['m']
    val = app * proj_util(inst, kind)
    ub = (c <= B + 1e-9).astype(float)
    cons = [LinearConstraint(c[None, :], -np.inf, B)]
    r1 = milp(c=-val, constraints=cons, integrality=np.ones(m), bounds=Bounds(np.zeros(m), ub),
              options=dict(time_limit=600, mip_rel_gap=0.0))
    if r1.x is None:
        SOLVER_LOG.append(dict(name=inst['name'], kind=kind, stage=1, status=int(r1.status), msg=str(r1.message)))
        return rule_greedy_av(inst)
    V = float(val @ np.round(r1.x))
    integral = np.allclose(val, np.round(val))
    tol = 0.5 if integral else 1e-9 * max(abs(V), 1.0)
    cons2 = cons + [LinearConstraint(val[None, :], V - tol, np.inf)]
    r2 = milp(c=-app, constraints=cons2, integrality=np.ones(m), bounds=Bounds(np.zeros(m), ub),
              options=dict(time_limit=600, mip_rel_gap=0.0))
    x = r2.x if r2.x is not None else r1.x
    W = np.round(x) > 0.5
    V2 = float(val[W].sum())
    SOLVER_LOG.append(dict(name=inst['name'], kind=kind, status1=int(r1.status), gap1=float(getattr(r1, 'mip_gap', np.nan) or 0.0),
                           status2=int(r2.status) if r2.x is not None else -1, V1=V, V2=V2, primary_kept=bool(V2 >= V - tol)))
    return _complete_greedy(inst, W)


def _mes_core(inst, kind, endow):
    """Method of Equal Shares with per-voter endowment `endow` (no completion)."""
    Ac = inst['Ac']; c = inst['cost']; m, n = inst['m'], inst['n']
    u = proj_util(inst, kind)
    b = np.full(n, float(endow))
    W = np.zeros(m, bool)
    rho_lb = np.zeros(m)                     # rho can only increase as budgets shrink
    indptr, indices = Ac.indptr, Ac.indices
    alive = np.diff(indptr) > 0
    rank = _rank(inst)
    while True:
        best, best_rho = -1, np.inf
        cand = np.where(alive & ~W)[0]
        cand = cand[np.lexsort((rank[cand], rho_lb[cand]))]
        for j in cand:
            if rho_lb[j] > best_rho:
                break
            bs = b[indices[indptr[j]:indptr[j + 1]]]
            if bs.sum() < c[j] - 1e-9 * max(c[j], 1):
                alive[j] = False; continue
            s = np.sort(bs); k = len(s)
            csum = np.concatenate(([0.0], np.cumsum(s)))
            r = np.arange(k)
            alpha = (c[j] - csum[:-1]) / (k - r)
            hit = np.where(alpha <= s + 1e-12)[0]
            rho = alpha[hit[0]] / u[j]
            rho_lb[j] = rho
            if rho < best_rho or (rho == best_rho and rank[j] < rank[best]):
                best, best_rho = j, rho
        if best < 0:
            break
        W[best] = True
        sup = indices[indptr[best]:indptr[best + 1]]
        b[sup] = np.maximum(b[sup] - best_rho * u[best], 0.0)
    return W


def rule_mes(inst, kind, completion=True, step_frac=0.01):
    """Method of Equal Shares completed as in pabutools' exhaustion_by_budget_increase (the scheme used in practice):
    the virtual budget is raised in additive steps of 1% of B; the procedure stops at the first outcome that is
    infeasible for the real budget (returning the previous outcome) or exhaustive (returning it); the result is
    then completed by utilitarian greedy (approvals)."""
    c, B, n = inst['cost'], inst['B'], inst['n']
    W = _mes_core(inst, kind, B / n)
    if not completion:
        return W
    prev = W; k = 0
    def exhaustive(W_):
        rem = B - c[W_].sum()
        return not ((~W_) & (c <= rem + 1e-9)).any()
    while True:
        if c[W].sum() > B + 1e-9:
            W = prev; break
        if exhaustive(W):
            break
        k += 1
        if k > 100 * (n + 1):
            break
        prev = W
        W = _mes_core(inst, kind, (B + k * step_frac * B) / n)
    return _complete_greedy(inst, W)


def rule_phragmen(inst):
    """Budgeted sequential Phragmen, skipping projects that no longer fit."""
    Ac = inst['Ac']; c = inst['cost']; B = inst['B']; m, n = inst['m'], inst['n']
    load = np.zeros(n); W = np.zeros(m, bool); spent = 0.0
    app = _approvals(inst); rank = _rank(inst)
    while True:
        ok = (~W) & (app > 0) & (spent + c <= B + 1e-9)
        if not ok.any():
            break
        t = (c + Ac.T @ load) / np.maximum(app, 1)
        t[~ok] = np.inf
        j = np.lexsort((rank, t))[0]
        sup = Ac.indices[Ac.indptr[j]:Ac.indptr[j + 1]]
        load[sup] = t[j]; W[j] = True; spent += c[j]
    return W


def rule_greedy_nash(inst):
    """Greedy Nash welfare, cost utilities: maximise marginal sum_i log(1+u_i/kappa) per unit cost, kappa=B/n."""
    Ac = inst['Ac']; c = inst['cost']; B = inst['B']; m, n = inst['m'], inst['n']
    kappa = B / n
    u = np.zeros(n); W = np.zeros(m, bool); spent = 0.0
    app = _approvals(inst); rank = _rank(inst)
    indptr, indices = Ac.indptr, Ac.indices
    while True:
        ok = np.where((~W) & (app > 0) & (spent + c <= B + 1e-9))[0]
        if len(ok) == 0:
            break
        gain = np.full(m, -np.inf)
        base = np.log1p(u / kappa)
        for j in ok:
            s = indices[indptr[j]:indptr[j + 1]]
            gain[j] = (np.log1p((u[s] + c[j]) / kappa) - base[s]).sum() / c[j]
        j = np.lexsort((rank, -gain))[0]
        s = indices[indptr[j]:indptr[j + 1]]
        u[s] += c[j]; W[j] = True; spent += c[j]
    return W


INCUMBENTS = ['GreedyAV', 'GreedyRatio', 'MaxCard', 'MaxCost', 'MES-card', 'MES-cost', 'Phragmen', 'GreedyNash']


def run_incumbents(inst):
    return {'GreedyAV': rule_greedy_av(inst), 'GreedyRatio': rule_greedy_ratio(inst),
            'MaxCard': rule_max_util(inst, 'card'), 'MaxCost': rule_max_util(inst, 'cost'),
            'MES-card': rule_mes(inst, 'card'), 'MES-cost': rule_mes(inst, 'cost'),
            'Phragmen': rule_phragmen(inst), 'GreedyNash': rule_greedy_nash(inst)}


def official_outcome(inst):
    s = inst['selected']
    if len(s) == 0 or any(x == '' for x in s):
        return None
    try:
        W = np.array([int(float(x)) > 0 for x in s])
    except ValueError:
        return None
    if inst['cost'][W].sum() > inst['B'] * 1.0001:
        return None
    return W


# ------------------------------------------------------------- learned rule
NFEAT = 9
HID = 8


def policy_features(inst, sat_card, sat_cost, b, spent, app, cand):
    """State-dependent project features: one round of voter->project message passing."""
    Ac = inst['Ac']; c = inst['cost']; B = inst['B']; n = inst['n']
    kappa = B / n
    AT = Ac.T
    f_zero = (AT @ (sat_card == 0).astype(float))[cand] / n     # uncovered supporters
    f_harm = (AT @ (1.0 / (1.0 + sat_card)))[cand] / n          # PAV-type weight
    f_harmc = (AT @ (1.0 / (1.0 + sat_cost / kappa)))[cand] / n  # cost-utility analogue
    f_bud = (AT @ b)[cand] / np.maximum(c[cand], 1e-9)           # equal-share purchasing power
    a = app[cand] / n
    cs = c[cand] / B
    rem = (B - spent) / B
    return np.stack([a, np.log(cs + 1e-6) / 10.0, np.log(a / (cs + 1e-6) + 1e-6) / 10.0,
                     f_zero, f_harm, f_harmc, np.minimum(f_bud, 3.0) / 3.0,
                     np.minimum(cs / max(rem, 1e-6), 1.0), np.full(len(cand), rem)], axis=1)


def n_params():
    return NFEAT * HID + HID + HID + NFEAT


def unpack(theta):
    k = 0
    W1 = theta[k:k + NFEAT * HID].reshape(NFEAT, HID); k += NFEAT * HID
    b1 = theta[k:k + HID]; k += HID
    w2 = theta[k:k + HID]; k += HID
    v = theta[k:k + NFEAT]
    return W1, b1, w2, v


def rule_learned(inst, theta):
    """Sequential neural rule: repeatedly fund the affordable project with the highest learned score."""
    W1, b1, w2, v = unpack(theta)
    Ac = inst['Ac']; c = inst['cost']; B = inst['B']; m, n = inst['m'], inst['n']
    app = _approvals(inst); rank = _rank(inst)
    W = np.zeros(m, bool); spent = 0.0
    sat_card = np.zeros(n); sat_cost = np.zeros(n); b = np.full(n, B / n)
    indptr, indices = Ac.indptr, Ac.indices
    while True:
        cand = np.where((~W) & (app > 0) & (spent + c <= B + 1e-9))[0]
        if len(cand) == 0:
            break
        F = policy_features(inst, sat_card, sat_cost, b, spent, app, cand)
        s = np.tanh(F @ W1 + b1) @ w2 + F @ v
        j = cand[np.lexsort((rank[cand], -s))[0]]
        sup = indices[indptr[j]:indptr[j + 1]]
        W[j] = True; spent += c[j]
        sat_card[sup] += 1; sat_cost[sup] += c[j]
        b[sup] = np.maximum(b[sup] - c[j] / len(sup), 0.0)
    return W


# ------------------------------------------------------------- elections
def pairwise_counts(U):
    """U: K x n utilities.  P[a,b] = number of voters strictly preferring outcome a to b."""
    K = U.shape[0]
    P = np.zeros((K, K))
    for a in range(K):
        P[a] = (U[a][None, :] > U).sum(1)
    return P


def soft_pairwise(U, norm, s):
    """Logistic (Koster et al. 2022-style) vote model: P_i(a>b) = sigma(s*(u_a-u_b)/norm_i); returns expected counts."""
    K = U.shape[0]
    P = np.zeros((K, K))
    Un = U / np.maximum(norm, 1e-12)[None, :]
    for a in range(K):
        d = Un[a][None, :] - Un
        P[a] = (1.0 / (1.0 + np.exp(-s * d))).sum(1)
    np.fill_diagonal(P, 0)
    return P


def outcome_classes(names, outs):
    keys, order = {}, []
    for r in names:
        k = outs[r].tobytes()
        if k not in keys:
            keys[k] = []; order.append(k)
        keys[k].append(r)
    return [keys[k] for k in order]


def condorcet(P):
    """Strict Condorcet winner (index or -1), weak winners, and the top cycle (Smith set)."""
    K = P.shape[0]
    M = P - P.T
    strict = [a for a in range(K) if all(M[a, b] > 0 for b in range(K) if b != a)]
    weak = [a for a in range(K) if all(M[a, b] >= 0 for b in range(K) if b != a)]
    cop = (M > 0).sum(1) + 0.5 * ((M == 0).sum(1) - 1)
    order = np.argsort(-cop, kind='stable')
    smith = list(range(K))
    for k in range(1, K + 1):
        S = set(order[:k].tolist())
        if all(M[a, b] > 0 for a in S for b in range(K) if b not in S):
            smith = sorted(S); break
    return (strict[0] if strict else -1), weak, smith


def maximal_lottery(P):
    """Maximal lottery (Fishburn 1984): optimal mixed strategy of the symmetric zero-sum game M = P - P^T."""
    M = P - P.T
    K = M.shape[0]
    if K == 1:
        return np.ones(1)
    Mn = M / max(np.abs(M).max(), 1.0)
    cvec = np.zeros(K + 1); cvec[-1] = -1
    A_ub = np.hstack([-Mn.T, np.ones((K, 1))])
    res = linprog(cvec, A_ub=A_ub, b_ub=np.zeros(K), A_eq=np.hstack([np.ones((1, K)), np.zeros((1, 1))]),
                  b_eq=[1], bounds=[(0, None)] * K + [(None, None)], method='highs')
    p = np.clip(res.x[:K], 0, None)
    return p / p.sum()




def has_strict_cycle(M):
    """True if the strict-majority digraph (edge a->b iff M[a,b] > 0) contains a directed cycle."""
    K = M.shape[0]
    adj = [[b for b in range(K) if M[a, b] > 0] for a in range(K)]
    color = [0] * K
    def dfs(u):
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1 or (color[v] == 0 and dfs(v)):
                return True
        color[u] = 2
        return False
    return any(color[u] == 0 and dfs(u) for u in range(K))


def classify(P):
    """'single' | 'strict_cw' | 'weak_cw' (no strict winner, some candidate never loses) | 'cycle' | 'tie_only'."""
    K = P.shape[0]
    if K == 1:
        return 'single'
    M = P - P.T
    if any(all(M[a, b] > 0 for b in range(K) if b != a) for a in range(K)):
        return 'strict_cw'
    if any(all(M[a, b] >= 0 for b in range(K) if b != a) for a in range(K)):
        return 'weak_cw'
    return 'cycle' if has_strict_cycle(M) else 'tie_only'


# ------------------------------------------------------------- audits
def _ejr_T(uW, sup, uT, cT, B, n, uW_plus=None):
    """Count unsatisfied supporters of T and test affordability. Returns (k_exact, viol_exact, k_up1, viol_up1)."""
    k = int((uW[sup] < uT - 1e-9).sum())
    ve = k * B / n >= cT - 1e-9
    if uW_plus is None:
        return k, ve, k, ve
    k1 = int((uW[sup] + uW_plus <= uT + 1e-9).sum())
    return k, ve, k1, k1 * B / n >= cT - 1e-9


def ejr_restricted(inst, W, kind, keep_cert=3):
    """EJR audit within cohesive groups defined by |T| <= 2 projects (exact within this family), for additive
    project utilities u_p = proj_util(kind). Exact EJR: some supporter has u_i(W) >= u(T). EJR up to one project
    (Peters et al. 2021): some supporter has u_i(W) >= u(T) or u_i(W) + max_{a in T minus W} u_a > u(T).
    Returns counts and up to keep_cert witness certificates (T, unsatisfied group size)."""
    Ac = inst['Ac']; c = inst['cost']; B = inst['B']; n, m = inst['n'], inst['m']
    up = proj_util(inst, kind)
    uW = utilities(inst, W, kind)
    ind, ptr = Ac.indptr, Ac.indices
    ind, ptr = Ac.indices, Ac.indptr
    app = np.diff(ptr)
    out = dict(n_exact=0, n_up1=0, worst_exact=0.0, worst_up1=0.0, cert=[])
    def rec(T, k, ve, k1, v1):
        if ve:
            out['n_exact'] += 1; out['worst_exact'] = max(out['worst_exact'], k / n)
            if len(out['cert']) < keep_cert:
                out['cert'].append(dict(T=[inst['pid'][t] for t in T], group=k, type='exact'))
        if v1:
            out['n_up1'] += 1; out['worst_up1'] = max(out['worst_up1'], k1 / n)
    for j in np.where(~W)[0]:
        if app[j] * B / n < c[j] - 1e-9:
            continue
        sup = ind[ptr[j]:ptr[j + 1]]
        rec((j,), *_ejr_T(uW, sup, up[j], c[j], B, n, up[j]))
    for j in range(m):
        if app[j] * B / n < c[j] - 1e-9:
            continue
        sj = ind[ptr[j]:ptr[j + 1]]
        for k2 in range(j + 1, m):
            if W[j] and W[k2]:
                continue
            cT = c[j] + c[k2]
            if min(app[j], app[k2]) * B / n < cT - 1e-9:
                continue
            both = np.intersect1d(sj, ind[ptr[k2]:ptr[k2 + 1]], assume_unique=True)
            if len(both) * B / n < cT - 1e-9:
                continue
            mx = max(up[t] for t in (j, k2) if not W[t])
            rec((j, k2), *_ejr_T(uW, both, up[j] + up[k2], cT, B, n, mx))
    return out


def ejr_exact_small(inst, W, kind, max_nodes=3_000_000):
    """Exact EJR and EJR-up-to-one audit over ALL project sets T by depth-first search with affordability pruning
    (if the supporters of T cannot afford T, no superset is affordable). Voter sets are Python-int bitsets.
    Returns dict(n_exact, n_up1, complete) where complete=False if the node budget was exhausted."""
    A = inst['A'].tocsr(); Ac = inst['Ac']; c = inst['cost']; B = inst['B']; n, m = inst['n'], inst['m']
    up = proj_util(inst, kind)
    uW = utilities(inst, W, kind)
    order = np.argsort(uW, kind='stable')               # voters sorted by utility -> thresholds are prefixes
    pos = np.empty(n, int); pos[order] = np.arange(n)
    us = uW[order]
    bits = []
    for j in range(m):
        s = 0
        for v in pos[Ac.indices[Ac.indptr[j]:Ac.indptr[j + 1]]]:
            s |= (1 << int(v))
        bits.append(s)
    def prefix_lt(x):
        return (1 << int(np.searchsorted(us, x - 1e-9, side='left'))) - 1
    def prefix_le(x):
        return (1 << int(np.searchsorted(us, x + 1e-9, side='right'))) - 1
    res = dict(n_exact=0, n_up1=0, complete=True)
    nodes = [0]
    def dfs(start, S, cT, uT, mxu, allin):
        for j in range(start, m):
            nodes[0] += 1
            if nodes[0] > max_nodes:
                res['complete'] = False; return
            S2 = S & bits[j] if S is not None else bits[j]
            c2 = cT + c[j]
            cnt = S2.bit_count()
            if cnt * B / n < c2 - 1e-9:
                continue                                  # unaffordable: prune whole subtree
            u2 = uT + up[j]; all2 = allin and W[j]
            mx2 = mxu if W[j] else max(mxu, up[j])
            if not all2:
                k = (S2 & prefix_lt(u2)).bit_count()
                if k * B / n >= c2 - 1e-9:
                    res['n_exact'] += 1
                k1 = (S2 & prefix_le(u2 - mx2)).bit_count()
                if k1 * B / n >= c2 - 1e-9:
                    res['n_up1'] += 1
            dfs(j + 1, S2, c2, u2, mx2, all2)
            if not res['complete']:
                return
    dfs(0, None, 0.0, 0.0, 0.0, True)
    res['nodes'] = nodes[0]
    return res


def ballot_cap_info(inst):
    """Maximum ballot length from META (if any) and the share of voters at that cap."""
    meta = inst['meta']
    blen = np.asarray(inst['A'].sum(1)).ravel()
    mx = None
    for key in ('max_length', 'max_sum_points'):
        try:
            mx = int(float(meta.get(key, '')))
            break
        except ValueError:
            continue
    at_cap = float((blen >= mx).mean()) if mx else 0.0
    return dict(max_length=mx, share_at_cap=at_cap, mean_ballot=float(blen.mean()))


def _gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    k = len(x)
    return (2 * np.arange(1, k + 1) - k - 1).dot(x) / (k * x.sum())


def instance_covariates(inst):
    A = inst['A']; Ac = inst['Ac']; c = inst['cost']; B = inst['B']; n, m = inst['n'], inst['m']
    app = _approvals(inst)
    blen = np.asarray(A.sum(1)).ravel()
    C = (Ac.T @ Ac).toarray().astype(float)
    d = np.sqrt(np.maximum(np.diag(C), 1e-12))
    Cn = C / d[:, None] / d[None, :]
    ev = np.sort(np.linalg.eigvalsh(Cn))[::-1]
    lam2 = ev[1] / ev[0] if m > 1 else 0.0
    rng = np.random.default_rng(0)
    k = min(n, 2000)
    S = A[rng.choice(n, size=k, replace=False)]
    sz = np.asarray(S.sum(1)).ravel()
    inter = (S @ S.T).toarray()
    J = inter / np.maximum(sz[:, None] + sz[None, :] - inter, 1)
    jac = J[np.triu_indices(k, 1)].mean() if k > 1 else 1.0
    return dict(n=n, m=m, budget_share=B / c.sum(), mean_ballot=float(blen.mean()), lam2=float(lam2),
                jaccard=float(jac), gini_app=float(_gini(app)), frac_affordable=float((c <= B).mean()))


# ------------------------------------------------------------- process pool helper
class SerialPool:
    """Drop-in replacement for multiprocessing.Pool when NPROC <= 1 (debugging)."""
    def __init__(self, n=1, initializer=None, initargs=()):
        if initializer:
            initializer(*initargs)
    def map(self, f, it, chunksize=1):
        return [f(x) for x in it]
    def imap_unordered(self, f, it, chunksize=1):
        return (f(x) for x in it)
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def make_pool(n, initializer=None, initargs=()):
    if n <= 1:
        return SerialPool(n, initializer, initargs)
    from multiprocessing import Pool
    return Pool(n, initializer=initializer, initargs=initargs)
