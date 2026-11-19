"""Cross-check pbcore (v2) against pabutools 1.2.3, including the COMPLETED Method of Equal Shares
(exhaustion by 1% budget increase, then utilitarian greedy completion), on a random sample of 150 elections."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, _HERE); _sys.path.insert(0, 'pylib')
import json, zipfile, random, collections
import numpy as np
import pbcore
from pabutools.election.pabulib import parse_pabulib_from_string
from pabutools.election.satisfaction.additivesatisfaction import Cost_Sat, Cardinality_Sat
from pabutools.rules import (greedy_utilitarian_welfare, method_of_equal_shares, sequential_phragmen,
                             completion_by_rule_combination, exhaustion_by_budget_increase)

zf = zipfile.ZipFile('approval.zip')
recs = json.load(open('out/stage1_instances.json'))
pool = [r['file'] for r in recs if r.get('split') and r['n'] <= 3000 and r['m'] <= 40]
random.Random(11).shuffle(pool)
rows = []
for fn in pool[:150]:
    text = zf.read(fn).decode('utf-8')
    inst = pbcore.build_instance(pbcore.parse_pb(text), fn)
    I, prof = parse_pabulib_from_string(text)

    def toW(S):
        s = {p.name for p in S}
        return np.array([pid in s for pid in inst['pid']])

    def mes_completed(sat):
        return completion_by_rule_combination(I, prof, [exhaustion_by_budget_increase, greedy_utilitarian_welfare],
                                              [{'rule': method_of_equal_shares, 'rule_params': {'sat_class': sat}}, {'sat_class': Cost_Sat}])
    pairs = {
        'GreedyAV': (toW(greedy_utilitarian_welfare(I, prof, sat_class=Cost_Sat)), pbcore.rule_greedy_av(inst)),
        'GreedyRatio': (toW(greedy_utilitarian_welfare(I, prof, sat_class=Cardinality_Sat)), pbcore.rule_greedy_ratio(inst)),
        'MES-cost (core)': (toW(method_of_equal_shares(I, prof, sat_class=Cost_Sat)), pbcore._mes_core(inst, 'cost', inst['B'] / inst['n'])),
        'MES-card (core)': (toW(method_of_equal_shares(I, prof, sat_class=Cardinality_Sat)), pbcore._mes_core(inst, 'card', inst['B'] / inst['n'])),
        'MES-cost (completed)': (toW(mes_completed(Cost_Sat)), pbcore.rule_mes(inst, 'cost')),
        'MES-card (completed)': (toW(mes_completed(Cardinality_Sat)), pbcore.rule_mes(inst, 'card')),
        'Phragmen (skip vs stop)': (toW(sequential_phragmen(I, prof)), pbcore.rule_phragmen(inst)),
    }
    for rule, (Wref, Wme) in pairs.items():
        uref = pbcore.utilities(inst, Wref, 'cost').sum(); ume = pbcore.utilities(inst, Wme, 'cost').sum()
        rows.append(dict(file=fn, rule=rule, identical=bool((Wref == Wme).all()), welfare_rel_diff=float((ume - uref) / max(uref, 1)),
                         n_ref=int(Wref.sum()), n_me=int(Wme.sum()), ref_subset_of_me=bool((Wref <= Wme).all())))
json.dump(rows, open('out/validation_pabutools.json', 'w'))
agg = collections.defaultdict(list)
for r in rows:
    agg[r['rule']].append(r['identical'])
print({k: (sum(v), len(v)) for k, v in agg.items()}, flush=True)
