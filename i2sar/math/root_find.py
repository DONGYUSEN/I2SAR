from __future__ import annotations

import numpy as np


def find_zero_brent(
    a: float,
    b: float,
    f,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> tuple[int, float]:
    """
    Brent's root-finding algorithm.
    
    Find a zero of the function f in the interval [a, b].
    
    Parameters
    ----------
    a : float
        Left endpoint of the interval
    b : float
        Right endpoint of the interval
    f : callable
        Function to find root of
    tol : float, optional
        Tolerance for convergence, defaults to 1e-7
    max_iter : int, optional
        Maximum number of iterations, defaults to 100
    
    Returns
    -------
    tuple[int, float]
        (convergence_code, root)
        convergence_code: 0 = success, 1 = max iterations reached, 2 = no bracket
    """
    fa = f(a)
    fb = f(b)
    
    if np.sign(fa) == np.sign(fb):
        return (2, 0.0)
    
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    c = a
    fc = fa
    d = b - a
    e = d
    
    for _ in range(max_iter):
        if abs(fc) < abs(fb):
            a, b = b, c
            c = a
            fa, fb = fb, fc
            fc = fa
        
        tol1 = 2.0 * np.finfo(float).eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        
        if abs(xm) <= tol1 or fb == 0:
            return (0, b)
        
        if abs(e) < tol1 or abs(fa) <= abs(fb):
            d = xm
            e = d
        else:
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            
            if p > 0:
                q = -q
            p = abs(p)
            
            min_val = min(3.0 * xm * q - abs(tol1 * q), abs(e * q))
            if 2.0 * p < min_val:
                e = d
                d = p / q
            else:
                d = xm
                e = d
        
        a = b
        fa = fb
        
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm > 0 else -tol1
        
        fb = f(b)
        
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c = a
            fc = fa
            d = b - a
            e = d
    
    return (1, b)
