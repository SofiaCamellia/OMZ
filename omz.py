"""OMZ v3 minimal algorithm release, extracted/adapted from research code.

Historical source modules: dev_v29.geometry, eval_v222.core/engine,
eval_omz_refine_20260922.run_refine. See README.md for scope and conventions.
All algorithms use a regular, finite grid. H is in grid steps unless the caller
explicitly converts it to physical units before scoring.
"""
import numpy as np

ENDPOINTS = ('event', 'first', 'last', 'count', 'total')
Q = (.50, .80, .90, .95)
TOL = 1e-10


def _checked_states(canlow, canhigh, k, minimum_k=1):
    """Release-only validation; DP transitions below are the source algorithms."""
    low = np.asarray(canlow, bool)
    high = np.asarray(canhigh, bool)
    if (low.ndim != 1 or len(low) < 2 or high.shape != low.shape
            or not np.all(low | high)):
        raise ValueError('at least two aligned nodes, each with a permitted state')
    if (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or k < minimum_k):
        raise ValueError(f'integer k >= {minimum_k} required')
    return low, high, len(low)



def _first(canlow,canhigh,k):
    n=len(canlow);result=np.zeros(n+1,bool);noqual={0}
    for a in range(n):
        if 0 in noqual and a+k<=n and canlow[a:a+k].all():result[a]=True
        new=set()
        if canhigh[a] and noqual:new.add(0)
        if canlow[a]:new.update(r+1 for r in noqual if r+1<k)
        noqual=new
    result[n]=bool(noqual)
    return result


def _attainable(canlow,canhigh,k):
    # Each active run length maps to bitsets of CLOSED-layer span/count totals.
    totals={0:1};counts={0:1}
    for lo,hi in zip(canlow,canhigh):
        nt={};nc={}
        for r,t in totals.items():
            c=counts[r]
            if hi:
                nt[0]=nt.get(0,0)|(t<<(r-1 if r>=k else 0))
                nc[0]=nc.get(0,0)|(c<<int(r>=k))
            if lo:nt[r+1]=t;nc[r+1]=c
        totals,counts=nt,nc
    total=count=0
    for r,t in totals.items():
        total|=t<<(r-1 if r>=k else 0);count|=counts[r]<<int(r>=k)
    n=len(canlow)
    return np.array([bool((total>>i)&1) for i in range(n)]),np.array([bool((count>>i)&1) for i in range(n+1)])


def allowed(y, valid, epsilon=0., threshold=60.):
    y=np.asarray(y,float);v=np.asarray(valid)
    if y.ndim!=1 or v.shape!=y.shape or v.dtype!=np.bool_ or not np.isfinite(y[v]).all():
        raise ValueError('finite observed values and aligned explicit bool support')
    if not np.isfinite(epsilon) or epsilon<0: raise ValueError('nonnegative epsilon')
    if not np.isfinite(threshold):raise ValueError('finite threshold')
    lo=y-epsilon;hi=y+epsilon
    forced_low=v&(hi<threshold);forced_high=v&(lo>=threshold)
    return ~forced_high,~forced_low,v&~(forced_low|forced_high)


def feasible(canlow, canhigh, k=3):
    low,high,n=_checked_states(canlow,canhigh,k)
    first=_first(low,high,k);rev=_first(low[::-1],high[::-1],k)
    total,count=_attainable(low,high,k)
    out=np.zeros((5,n+1),bool)
    out[1]=first;out[2]=np.r_[rev[:-1][::-1],rev[-1]]
    out[3]=count;out[4,:n]=total
    out[0,0]=first[-1];out[0,1]=first[:-1].any()
    if not out.any(1).all():raise AssertionError('empty endpoint set')
    return out


def envelope(t):
    """Discrete grid hull; ABSENT is preserved separately only for first/last."""
    t=np.asarray(t,bool);out=np.zeros_like(t);n=t.shape[-1]-1
    for e,row in enumerate(t):
        stop=n if e in (1,2) else n+1
        ix=np.flatnonzero(row[:stop])
        if len(ix):out[e,ix[0]:ix[-1]+1]=True
        if e in (1,2):out[e,n]=row[n]
    if np.any(t&~out):raise AssertionError('not a superset')
    return out


def sample_endpoints(x,k=3,threshold=60.):
    """Independent complete-curve scan; return grid indices (H in grid steps)."""
    x=np.asarray(x)
    if x.ndim!=2 or not all(x.shape) or not np.isfinite(x).all():raise ValueError('nonempty complete samples')
    _checked_states(np.ones(x.shape[1],bool),np.ones(x.shape[1],bool),k)
    if not np.isfinite(threshold):raise ValueError('finite threshold')
    m,n=x.shape;out=np.zeros((m,5),np.int64)
    out[:,1:3]=n;r=np.zeros(m,int)
    for j in range(n+1):
        bit=x[:,j]<threshold if j<n else np.zeros(m,bool)
        close=(~bit)&(r>=k);ix=np.flatnonzero(close)
        out[ix,0]=1;out[ix,1]=np.minimum(out[ix,1],j-r[ix]);out[ix,2]=j-1
        out[ix,3]+=1;out[ix,4]+=r[ix]-1
        r=np.where(bit,r+1,0)
    return out


def empirical_mass(values,categories):
    if isinstance(categories,(bool,np.bool_)) or not isinstance(categories,(int,np.integer)) or categories<1:
        raise ValueError('positive integer categories')
    a=np.asarray(values)
    if a.ndim!=2 or not all(a.shape) or not np.isfinite(a).all() or not np.all(a==np.round(a)) or np.any((a<0)|(a>=categories)):
        raise ValueError('integer [profile,sample] endpoint values')
    n,m=a.shape
    p=np.bincount((np.arange(n)[:,None]*categories+a.astype(int)).ravel(),minlength=n*categories).reshape(n,categories)/m
    np.testing.assert_allclose(p.sum(1),1.,rtol=0,atol=1e-12)
    return p


def categorical_scores(p):
    """Multiclass Brier, not binary Brier/2 and not a boundary distance."""
    p=np.asarray(p,float)
    if np.any(p<0) or not np.isfinite(p).all() or not np.allclose(p.sum(-1),1,rtol=0,atol=1e-12):raise ValueError('probability mass')
    return 1.+np.sum(p*p,axis=-1,keepdims=True)-2*p


def mass_sets(p,levels=Q):
    p=np.asarray(p,float)
    categorical_scores(p)  # release validation of the supplied probability mass
    order=np.argsort(-p,axis=-1,kind='stable')
    sorted_p=np.take_along_axis(p,order,axis=-1);cumulative=np.cumsum(sorted_p,axis=-1)
    out=[]
    for q in levels:
        if not 0<q<=1:raise ValueError('mass level')
        keep=(cumulative-sorted_p)<q
        keep &= sorted_p>0
        b=np.zeros_like(p,bool);np.put_along_axis(b,order,keep,axis=-1);out.append(b)
    return np.stack(out)


def empirical_crps(samples,grid):
    x=np.asarray(samples,float);grid=np.asarray(grid,float)
    if x.ndim!=2 or not all(x.shape) or grid.ndim!=1 or not len(grid) or not np.isfinite(grid).all():
        raise ValueError('nonempty [profile,sample] values and finite 1-D grid')
    x=np.sort(x,axis=1);m=x.shape[1]
    if not np.isfinite(x).all():raise ValueError('finite sample endpoints')
    # Half mean pair distance = dot(sorted sample, 2*i-M-1)/M^2.
    correction=(x*(2*np.arange(1,m+1)-m-1)).sum(1)/(m*m)
    ans=np.empty((len(x),len(grid)))
    for start in range(0,len(x),64):
        ans[start:start+64]=np.abs(x[start:start+64,:,None]-np.asarray(grid)[None,None,:]).mean(1)-correction[start:start+64,None]
    return ans


def joint_kh(canlow,canhigh,k=3):
    """Exact joint (K,Hsteps): active r -> K -> bitset of closed H."""
    low,high,n=_checked_states(canlow,canhigh,k,minimum_k=2)
    states={0:{0:1}}
    def merge(dst,source,r):
        dk=int(r>=k);dh=r-1 if dk else 0
        for count,bit in source.items():dst[count+dk]=dst.get(count+dk,0)|(bit<<dh)
    for lo,hi in zip(low,high):
        nxt={}
        for r,table in states.items():
            if hi:merge(nxt.setdefault(0,{}),table,r)
            if lo:nxt[r+1]=table.copy()
        states=nxt
    result={}
    for r,table in states.items():merge(result,table,r)
    Kmax=(n+1)//(k+1);out=np.zeros((Kmax+1,n),bool)
    for count,bit in result.items():out[count]=[bool((bit>>h)&1) for h in range(n)]
    return out


def global_feasible_pairs_steps(nodes: int, minimum: int):
    """All globally possible (count, span_steps), independent of observations."""
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
           or x < 1 for x in (nodes, minimum)):
        raise ValueError('positive integer J/k required')
    out = {(0, 0)}
    for count in range(1, (nodes+1)//(minimum+1)+1):
        out.update((count, span) for span in range((minimum-1)*count, nodes-2*count+2))
    return frozenset(out)


def classify(possible, prediction):
    possible=np.asarray(possible,bool);prediction=np.asarray(prediction,bool)
    if not possible.any(-1).all(): raise ValueError('empty truth set is not a certificate')
    inside = (possible & prediction).any(-1)
    outside = (possible & ~prediction).any(-1)
    return np.where(~outside, 0, np.where(~inside, 1, 2)).astype(np.uint8)


def score_bounds(score, possible):
    score=np.asarray(score,float);possible=np.asarray(possible,bool)
    if not np.isfinite(score).all():raise ValueError('finite category scores')
    if not possible.any(-1).all(): raise ValueError('empty truth set')
    return np.stack((np.where(possible, score, np.inf).min(-1),
                     np.where(possible, score, -np.inf).max(-1)), -1)


def rank(bounds, gamma=0.):
    b = np.asarray(bounds)
    if not np.isfinite(gamma) or gamma < 0 or not np.isfinite(b).all() or np.any(b[..., 0] > b[..., 1]+TOL):
        raise ValueError('finite ordered bounds and nonnegative gamma required')
    return np.where(b[..., 1] < -gamma-TOL, -1,
                    np.where(b[..., 0] > gamma+TOL, 1, 0)).astype(np.int8)


def kh_candidates(marginal, k=3):
    """Cartesian and global-filtered K/H baselines; arrays indexed by [K,Hsteps].

    Adapted from the inline baseline construction in run_refine.main.
    The marginal input must have been computed with this same k.
    """
    marginal = np.asarray(marginal, bool)
    if marginal.ndim != 2 or marginal.shape[0] != 5 or marginal.shape[1] < 3:
        raise ValueError('five endpoint rows on a grid with at least two nodes')
    n = marginal.shape[1] - 1
    pairs = global_feasible_pairs_steps(n, k)
    kmax = (n + 1) // (k + 1)
    product = marginal[3, :kmax+1, None] & marginal[4, None, :n]
    global_mask = np.zeros_like(product)
    for count, span in pairs:
        global_mask[count, span] = True
    return product, product & global_mask


def compare_scores(score_a, score_b, possible):
    """Bounds for score A minus score B: return (common_truth, separate).

    Adapted from eval_v222.engine.evaluate, removing scenario/method axes.
    Every score array and possible set shares its final category axis.
    Lower score is better; rank +1 means B is uniformly better than A.
    """
    score_a, score_b = np.asarray(score_a, float), np.asarray(score_b, float)
    common = score_bounds(score_a - score_b, possible)
    a, b = score_bounds(score_a, possible), score_bounds(score_b, possible)
    separate = np.stack((a[..., 0] - b[..., 1], a[..., 1] - b[..., 0]), -1)
    return common, separate

