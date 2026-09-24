"""Independent small-grid test oracle extracted from eval_v222.core.

Used only by tests. Exponential in the number of unknown nodes.
"""
import itertools
import numpy as np


def exhaustive(pattern,k=3):
    p=np.asarray(pattern);n=len(p);unknown=np.flatnonzero(p<0)
    marginal=np.zeros((5,n+1),bool);joint=set()
    for fill in itertools.product((0,1),repeat=len(unknown)):
        b=p.copy();b[unknown]=fill
        # Independent oracle uses explicit maximal runs, not production scan/DP.
        runs=[];start=None
        for j in range(n+1):
            bit=b[j] if j<n else 0
            if bit and start is None:start=j
            if not bit and start is not None:
                if j-start>=k:runs.append((start,j-1))
                start=None
        total=sum(b-a for a,b in runs);count=len(runs)
        vals=(int(count>0),runs[0][0] if count else n,runs[-1][1] if count else n,count,total)
        for e,value in enumerate(vals):marginal[e,value]=True
        joint.add((count,total))
    return marginal,joint
