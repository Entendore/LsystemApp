#!/usr/bin/env python3
"""
Fauna — a production-grade academic L-system laboratory.

Engine (Qt-free, testable): D0L/IL systems, parametric & conditional
productions, context sensitivity, weighted stochastic rules, 3D turtle
with tropism, filled polygons ({ . }), sphere/disc elements (@O/@o),
growth-matrix analysis, camera/lighting primitives, line-numbered
grammar diagnostics.

Renderer: shaded-tube mesh pipeline (world-space key light + ambient,
fog, height/depth colouring, polygons, elements, ground grid/shadow)
with perspective or orthographic camera, 2D roll, wire fallback,
asynchronous mesh building for large scenes, PNG/SVG/turntable export.

Application: threaded non-blocking computation, async render prep,
session restore (including unsaved edits), QSettings persistence,
recent files, keyboard navigation, clipboard image copy, drag & drop,
headless CLI export, logging, crash dialog, self-test suite.

Usage:
    python fauna_lsystem.py                       # GUI
    python fauna_lsystem.py --preset "Berry Bush (3D elements)"
    python fauna_lsystem.py --load sys.json
    python fauna_lsystem.py --preset "Whorl Bush (3D)" --fov 55 --fog 30 \
        --export bush.png --size 3200             # headless render
    python fauna_lsystem.py --list-presets
    python fauna_lsystem.py --selftest            # engine test suite

Dependencies:  pip install PySide6
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import traceback
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field

__version__ = "1.4.0"
APP_NAME = "Fauna"
ORG_NAME = "FaunaLabs"

MAX_SYMBOLS = 400_000
DEBOUNCE_MS = 220

MESH_LIVE_MAX = 45_000        # live-view shaded-tube cap
MESH_TURNTABLE_MAX = 6_000    # synchronous (turntable) mesh budget
MESH_EXPORT_CAP = 300_000     # shaded export cap (else wire export)
MESH_SHADES = 8               # lambert tone levels
LIGHT_AZ_DEFAULT = 140        # key-light azimuth (deg, world)
LIGHT_EL_DEFAULT = 50         # key-light elevation (deg, world)
AMBIENT_DEFAULT = 18          # ambient floor (percent of full tone)
FILL_OPACITY_DEFAULT = 82     # polygon/element fill opacity (percent)

LOG = logging.getLogger("fauna")


def _log_path() -> str:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Logs")
    else:
        base = os.environ.get("XDG_STATE_HOME",
                              os.path.expanduser("~/.local/state"))
    d = os.path.join(base, "fauna")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "fauna.log")


def setup_logging(level: int, console: bool = False) -> None:
    LOG.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        fh = logging.FileHandler(_log_path(), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    except OSError:
        pass
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        LOG.addHandler(ch)


# ===========================================================================
# ENGINE — pure Python, no Qt. Deterministic given (text, seed).
# ===========================================================================


class LSystemError(ValueError):
    """User-facing error in an L-system definition."""


class OperationCancelled(Exception):
    """Internal: raised when a computation is superseded."""


# ---- expression language (parameters, conditions, #define) ----------------

_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>\d+\.\d*|\.\d+|\d+)
  | (?P<id>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>\*\*|<=|>=|==|!=|&&|\|\||[-+*/%^()<>!,&|])
""", re.VERBOSE)


def _tokenize(src: str):
    toks, i, n = [], 0, len(src)
    while i < n:
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise LSystemError(f"bad character {src[i]!r} in expression {src!r}")
        i = m.end()
        if m.lastgroup != "ws":
            toks.append((m.lastgroup, m.group()))
    return toks


class _Expr:
    """Recursive-descent parser → AST tuples.  ||  <  &&  <  cmp  <  +-  <
    */%  <  unary -!  <  ** ^ (right-assoc)."""

    __slots__ = ("toks", "i")

    def __init__(self, toks):
        self.toks, self.i = toks, 0

    def _peek(self, *ops):
        if self.i < len(self.toks):
            k, v = self.toks[self.i]
            if k == "op" and v in ops:
                return v
        return None

    def _take(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def parse(self):
        node = self._or()
        if self.i != len(self.toks):
            raise LSystemError(f"unexpected {self.toks[self.i][1]!r} in expression")
        return node

    def _or(self):
        n = self._and()
        while self._peek("||"):
            self._take()
            n = ("or", n, self._and())
        return n

    def _and(self):
        n = self._cmp()
        while self._peek("&&"):
            self._take()
            n = ("and", n, self._cmp())
        return n

    def _cmp(self):
        n = self._add()
        while True:
            op = self._peek("==", "!=", "<", "<=", ">", ">=")
            if not op:
                return n
            self._take()
            n = ("cmp", op, n, self._add())

    def _add(self):
        n = self._mul()
        while True:
            op = self._peek("+", "-")
            if not op:
                return n
            self._take()
            n = ("bin", op, n, self._mul())

    def _mul(self):
        n = self._unary()
        while True:
            op = self._peek("*", "/", "%")
            if not op:
                return n
            self._take()
            n = ("bin", op, n, self._unary())

    def _unary(self):
        if self._peek("-"):
            self._take()
            return ("neg", self._unary())
        if self._peek("!"):
            self._take()
            return ("not", self._unary())
        if self._peek("+"):
            self._take()
            return self._unary()
        return self._pow()

    def _pow(self):
        n = self._atom()
        if self._peek("**", "^"):
            self._take()
            n = ("bin", "**", n, self._unary())
        return n

    def _atom(self):
        if self.i >= len(self.toks):
            raise LSystemError("unexpected end of expression")
        k, v = self._take()
        if k == "num":
            return ("num", float(v))
        if k == "id":
            if self._peek("("):
                self._take()
                args = []
                if not self._peek(")"):
                    args.append(self._or())
                    while self._peek(","):
                        self._take()
                        args.append(self._or())
                if not self._peek(")"):
                    raise LSystemError("missing ')' in call")
                self._take()
                return ("call", v, args)
            return ("var", v)
        if k == "op" and v == "(":
            node = self._or()
            if not self._peek(")"):
                raise LSystemError("missing ')'")
            self._take()
            return node
        raise LSystemError(f"unexpected {v!r} in expression")


def parse_expr(src: str):
    toks = _tokenize(src)
    if not toks:
        raise LSystemError("empty expression")
    return _Expr(toks).parse()


def _truthy(v) -> bool:
    return v != 0


_MATH_FN = {n: getattr(math, n) for n in
            ("sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sqrt",
             "exp", "log", "log10", "floor", "ceil", "fabs", "fmod", "pow")}
_MATH_FN.update({"abs": abs, "min": min, "max": max, "round": round,
                 "ln": math.log})


def eval_expr(node, env: dict, rng: random.Random) -> float:
    k = node[0]
    if k == "num":
        return node[1]
    if k == "var":
        name = node[1]
        if name in env:
            return env[name]
        if name == "pi":
            return math.pi
        if name == "e":
            return math.e
        raise LSystemError(f"unknown identifier '{name}'")
    if k == "neg":
        return -eval_expr(node[1], env, rng)
    if k == "not":
        return 0.0 if _truthy(eval_expr(node[1], env, rng)) else 1.0
    if k == "or":
        if _truthy(eval_expr(node[1], env, rng)):
            return 1.0
        return 1.0 if _truthy(eval_expr(node[2], env, rng)) else 0.0
    if k == "and":
        if not _truthy(eval_expr(node[1], env, rng)):
            return 0.0
        return 1.0 if _truthy(eval_expr(node[2], env, rng)) else 0.0
    if k == "cmp":
        op = node[1]
        a = eval_expr(node[2], env, rng)
        b = eval_expr(node[3], env, rng)
        ok = {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b,
              ">": a > b, ">=": a >= b}[op]
        return 1.0 if ok else 0.0
    if k == "bin":
        op = node[1]
        a = eval_expr(node[2], env, rng)
        b = eval_expr(node[3], env, rng)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                raise ZeroDivisionError("division by zero in expression")
            return a / b
        if op == "%":
            return math.fmod(a, b)
        return a ** b
    name, args = node[1], [eval_expr(a, env, rng) for a in node[2]]
    if name == "rand":
        if len(args) != 2:
            raise LSystemError("rand(a, b) takes two arguments")
        return rng.uniform(args[0], args[1])
    if name == "randint":
        return float(rng.randint(int(args[0]), int(args[1])))
    fn = _MATH_FN.get(name)
    if fn is None:
        raise LSystemError(f"unknown function '{name}'")
    return fn(*args)


# ---- modules, productions, classification ---------------------------------


def _split_top(s: str, sep: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == sep and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def parse_modules(text: str) -> list[tuple[str, tuple]]:
    """Parse a word into modules. '@O(r)'/'@o(r)' become two-char symbols."""
    mods, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace() or c == ",":
            i += 1
            continue
        sym = c
        i += 1
        if sym == "@" and i < n and text[i] in "Oo":
            sym += text[i]
            i += 1
        params: tuple = ()
        if i < n and text[i] == "(":
            depth, j = 1, i + 1
            while j < n and depth:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            if depth:
                raise LSystemError(f"unbalanced '(' after '{sym}'")
            inner = text[i + 1:j - 1]
            i = j
            if inner.strip():
                params = tuple(parse_expr(p) for p in _split_top(inner, ","))
        mods.append((sym, params))
    return mods


def _ctx_names(mods):
    out = []
    for s, ps in mods:
        names = []
        for e in ps:
            if e[0] != "var":
                raise LSystemError("context parameters must be plain names")
            names.append(e[1])
        out.append((s, names))
    return out


class Rule:
    __slots__ = ("sym", "params", "lctx", "rctx", "cond", "variants", "spec")

    def __init__(self, sym, params, lctx, rctx, cond, variants):
        self.sym, self.params = sym, params
        self.lctx, self.rctx, self.cond = lctx, rctx, cond
        self.variants = variants
        self.spec = len(lctx) + len(rctx)


def _parse_rule(head: str) -> Rule:
    lhs_txt, rhs_txt = head.split("->", 1)
    cond = None
    if ":" in lhs_txt:
        lhs_txt, cond_txt = lhs_txt.split(":", 1)
        cond = parse_expr(cond_txt)
    left = right = ""
    if "<" in lhs_txt:
        left, lhs_txt = lhs_txt.split("<", 1)
    if ">" in lhs_txt:
        lhs_txt, right = lhs_txt.split(">", 1)
    lctx = _ctx_names(parse_modules(left)) if left.strip() else []
    rctx = _ctx_names(parse_modules(right)) if right.strip() else []
    core = parse_modules(lhs_txt)
    if len(core) != 1:
        raise LSystemError(f"exactly one symbol expected on the left: {head!r}")
    (sym, ps), = core
    pnames = []
    for e in ps:
        if e[0] != "var":
            raise LSystemError("production parameters must be plain names")
        pnames.append(e[1])
    variants = []
    for chunk in _split_top(rhs_txt, ";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        weight = None
        m = re.match(r"^(\d+(?:\.\d*)?|\.\d+)\s*(.+)$", chunk)
        if m:
            weight = float(m.group(1))
            chunk = m.group(2)
        variants.append((weight, parse_modules(chunk)))
    if not variants:
        variants = [(None, [])]
    return Rule(sym, pnames if ps else None, lctx, rctx, cond, variants)


def parse_program(text: str) -> tuple[list[Rule], dict[str, float]]:
    """Parse rules-editor text → (productions, #define constants).
    Errors are reported with 1-based line numbers."""
    defines: dict[str, float] = {}
    rules: list[Rule] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        try:
            parts = raw.split("#", 1)
            head = parts[0].strip()
            tail = parts[1].strip() if len(parts) > 1 else ""
            if tail.startswith("define"):
                bits = tail.split(None, 2)
                if len(bits) >= 3:
                    defines[bits[1]] = eval_expr(parse_expr(bits[2]),
                                                 dict(defines),
                                                 random.Random(0))
                continue
            if not head or "->" not in head:
                continue
            rules.append(_parse_rule(head))
        except LSystemError as e:
            raise LSystemError(f"line {lineno}: {e}") from None
        except ZeroDivisionError as e:
            raise LSystemError(f"line {lineno}: {e}") from None
        except OverflowError as e:
            raise LSystemError(f"line {lineno}: {e}") from None
    return rules, defines


def rules_by_symbol(rules) -> dict[str, list[Rule]]:
    """Group productions per symbol, longest context first."""
    out: dict[str, list[Rule]] = {}
    for r in rules:
        out.setdefault(r.sym, []).append(r)
    for lst in out.values():
        lst.sort(key=lambda r: (-r.spec, 0 if r.params else 1))
    return out


def classify(rules) -> str:
    if not rules:
        return "D0L · no productions"
    both = any(r.lctx and r.rctx for r in rules)
    ctx = any(r.lctx or r.rctx for r in rules)
    base = "2L" if both else ("1L" if ctx else "D0L")
    tags = []
    if any(len(r.variants) > 1 or any(w is not None for w, _ in r.variants)
           for r in rules):
        tags.append("stochastic")
    if any(r.params or any(ps for _, mods in r.variants for _, ps in mods)
           for r in rules):
        tags.append("parametric")
    if any(r.cond is not None for r in rules):
        tags.append("conditional")
    return " · ".join([base] + tags)


def collect_alphabet(rules, axiom_mods) -> set[str]:
    syms = {s for s, _ in axiom_mods}
    for r in rules:
        syms.add(r.sym)
        for _, mods in r.variants:
            syms.update(s for s, _ in mods)
    return syms


# ---- derivation ------------------------------------------------------------


def is_simple(rules, axiom_mods) -> bool:
    for r in rules:
        if r.lctx or r.rctx or r.cond is not None or r.params:
            return False
        for _, mods in r.variants:
            for _, ps in mods:
                if ps:
                    return False
    return all(not ps for _, ps in axiom_mods)


def simple_tables(rules):
    return {r.sym: [(w, "".join(s for s, _ in mods)) for w, mods in r.variants]
            for r in rules}


def _pick(variants, rng):
    if len(variants) == 1:
        return variants[0][1]
    tot = sum(w if w is not None else 1.0 for w, _ in variants)
    x, acc = rng.random() * tot, 0.0
    for w, rhs in variants:
        acc += w if w is not None else 1.0
        if x <= acc:
            return rhs
    return variants[-1][1]


def expand_simple(axiom: str, table, iters: int, rng: random.Random, cap: int,
                  cancel=None) -> tuple[str, bool]:
    det = all(len(v) == 1 and v[0][0] is None for v in table.values())
    s = axiom
    if det:
        t = str.maketrans({ch: v[0][1] for ch, v in table.items()})
        for _ in range(iters):
            if cancel is not None and cancel():
                raise OperationCancelled()
            s2 = s.translate(t)
            if len(s2) > cap:
                return s, True
            s = s2
        return s, False
    get = table.get
    for _ in range(iters):
        if cancel is not None and cancel():
            raise OperationCancelled()
        out = []
        for ch in s:
            v = get(ch)
            if v is None:
                out.append(ch)
            elif len(v) == 1:
                out.append(v[0][1])
            else:
                out.append(_pick(v, rng))
        s2 = "".join(out)
        if len(s2) > cap:
            return s, True
        s = s2
    return s, False


def _step_complex(mods, by_sym, defines, rng, cap, cancel=None):
    out = []
    ap = out.append
    n = len(mods)
    for i in range(n):
        if cancel is not None and not (i & 0x7F) and cancel():
            raise OperationCancelled()
        sym, vals = mods[i]
        cands = by_sym.get(sym)
        if not cands:
            ap(mods[i])
            continue
        for r in cands:
            if r.params is not None and len(vals) != len(r.params):
                continue
            env = None
            ok = True
            if r.lctx:
                k = len(r.lctx)
                if i - k < 0:
                    continue
                env = dict(defines)
                for j in range(k):
                    cs, cnames = r.lctx[j]
                    cm = mods[i - k + j]
                    if cm[0] != cs or len(cm[1]) < len(cnames):
                        ok = False
                        break
                    for pj, pn in enumerate(cnames):
                        env[pn] = cm[1][pj]
                if not ok:
                    continue
            if r.rctx:
                k = len(r.rctx)
                if i + 1 + k > n:
                    continue
                if env is None:
                    env = dict(defines)
                for j in range(k):
                    cs, cnames = r.rctx[j]
                    cm = mods[i + 1 + j]
                    if cm[0] != cs or len(cm[1]) < len(cnames):
                        ok = False
                        break
                    for pj, pn in enumerate(cnames):
                        env[pn] = cm[1][pj]
                if not ok:
                    continue
            if r.params is not None:
                if env is None:
                    env = dict(defines)
                for pn, pv in zip(r.params, vals):
                    env[pn] = pv
            if r.cond is not None and not _truthy(
                    eval_expr(r.cond, env if env is not None else dict(defines),
                              rng)):
                continue
            rhs = _pick(r.variants, rng)
            env2 = env if env is not None else dict(defines)
            for s2, exprs in rhs:
                if not exprs:
                    ap((s2, ()))
                else:
                    ap((s2, tuple(eval_expr(e, env2, rng) for e in exprs)))
            break
        else:
            ap(mods[i])
    if len(out) > cap:
        return mods, True
    return out, False


def expand_complex(axiom_mods, by_sym, defines, rng, iters, cap, cancel=None):
    mods = axiom_mods
    for _ in range(iters):
        mods, tr = _step_complex(mods, by_sym, defines, rng, cap, cancel)
        if tr:
            return mods, True
    return mods, False


# ---- turtle ----------------------------------------------------------------


def _rotv(vx, vy, vz, kx, ky, kz, c, s):
    d = vx * kx + vy * ky + vz * kz
    rx = ky * vz - kz * vy
    ry = kz * vx - kx * vz
    rz = kx * vy - ky * vx
    wx = vx * c + rx * s + kx * d * (1.0 - c)
    wy = vy * c + ry * s + ky * d * (1.0 - c)
    wz = vz * c + rz * s + kz * d * (1.0 - c)
    n = math.sqrt(wx * wx + wy * wy + wz * wz)
    if n > 0.0:
        wx /= n
        wy /= n
        wz /= n
    return wx, wy, wz


def interpret(mods, angle_deg: float, jitter: float, seed: int, draw_set,
              trop=(0.0, 0.0, 0.0), elast: float = 0.0, cancel=None):
    """Turtle-plot a word.

    Returns (groups, polys, elems, elem_orders, total, bbox3, max_depth,
    width_explicit). Polygon alphabet: { . }; elements: @O/@o.
    """
    rng = random.Random((seed * 2654435761 & 0xFFFFFFFF) ^ 0x9E3779B9)
    rad0 = math.radians(angle_deg)
    uniform = rng.uniform
    groups: dict = {}
    polys: list = []
    elems: list = []
    elem_orders: list = []
    total = 0
    x = y = z = 0.0
    hx, hy, hz = 0.0, 1.0, 0.0
    lx, ly, lz = 1.0, 0.0, 0.0
    ux, uy, uz = 0.0, 0.0, -1.0
    w = 1.0
    depth = 0
    stack: list = []
    poly = None
    skip = False
    skip_depth = 0
    width_explicit = False
    minx = miny = minz = maxx = maxy = maxz = 0.0
    gx, gy, gz = trop
    use_trop = elast > 0.0 and (gx or gy or gz)

    def bare():
        return rad0 * (1.0 + uniform(-jitter, jitter)) if jitter else rad0

    def turn(ang):
        nonlocal hx, hy, hz, lx, ly, lz
        c, s = math.cos(ang), math.sin(ang)
        hx, hy, hz = _rotv(hx, hy, hz, ux, uy, uz, c, s)
        lx, ly, lz = _rotv(lx, ly, lz, ux, uy, uz, c, s)

    def pitch(ang):
        nonlocal hx, hy, hz, ux, uy, uz
        c, s = math.cos(ang), math.sin(ang)
        hx, hy, hz = _rotv(hx, hy, hz, lx, ly, lz, c, s)
        ux, uy, uz = _rotv(ux, uy, uz, lx, ly, lz, c, s)

    def roll(ang):
        nonlocal lx, ly, lz, ux, uy, uz
        c, s = math.cos(ang), math.sin(ang)
        lx, ly, lz = _rotv(lx, ly, lz, hx, hy, hz, c, s)
        ux, uy, uz = _rotv(ux, uy, uz, hx, hy, hz, c, s)

    def bend():
        nonlocal hx, hy, hz, lx, ly, lz, ux, uy, uz
        hx += elast * gx
        hy += elast * gy
        hz += elast * gz
        n = math.sqrt(hx * hx + hy * hy + hz * hz) or 1.0
        hx /= n
        hy /= n
        hz /= n
        d = lx * hx + ly * hy + lz * hz
        lx -= d * hx
        ly -= d * hy
        lz -= d * hz
        n = math.sqrt(lx * lx + ly * ly + lz * lz) or 1.0
        lx /= n
        ly /= n
        lz /= n
        ux = hy * lz - hz * ly
        uy = hz * lx - hx * lz
        uz = hx * ly - hy * lx

    def mark(px, py, pz):
        nonlocal minx, miny, minz, maxx, maxy, maxz
        if px < minx: minx = px
        elif px > maxx: maxx = px
        if py < miny: miny = py
        elif py > maxy: maxy = py
        if pz < minz: minz = pz
        elif pz > maxz: maxz = pz

    idx = 0
    for item in mods:
        if cancel is not None and not (idx & 0xFFF) and cancel():
            raise OperationCancelled()
        idx += 1
        if item.__class__ is str:
            sym, vals = item, ()
        else:
            sym, vals = item
        if skip:
            if sym == "[":
                skip_depth += 1
            elif sym == "]":
                if skip_depth == 0:
                    skip = False
                    if stack:
                        (x, y, z, hx, hy, hz, lx, ly, lz,
                         ux, uy, uz, w, depth, poly) = stack.pop()
                else:
                    skip_depth -= 1
            continue
        if sym in draw_set:
            ln = vals[0] if vals else 1.0
            if use_trop:
                bend()
            ex, ey, ez = x + hx * ln, y + hy * ln, z + hz * ln
            key = (depth, round(w * 8) / 8.0)
            g = groups.get(key)
            if g is None:
                g = groups[key] = ([], [])
            g[0].append((x, y, z, ex, ey, ez))
            g[1].append(total)
            total += 1
            x, y, z = ex, ey, ez
            mark(x, y, z)
        elif sym == "f" or sym == "g":
            ln = vals[0] if vals else 1.0
            x += hx * ln
            y += hy * ln
            z += hz * ln
            mark(x, y, z)
        elif sym == "+":
            turn(vals[0] if vals else bare())
        elif sym == "-":
            turn(-(vals[0] if vals else bare()))
        elif sym == "&":
            pitch(vals[0] if vals else bare())
        elif sym == "^":
            pitch(-(vals[0] if vals else bare()))
        elif sym == "/":
            roll(vals[0] if vals else bare())
        elif sym == "\\":
            roll(-(vals[0] if vals else bare()))
        elif sym == "|":
            turn(math.pi)
        elif sym == "!":
            width_explicit = True
            w = vals[0] if vals else w * 0.9
        elif sym == "{":
            poly = []
        elif sym == ".":
            if poly is not None:
                poly.append((x, y, z))
                mark(x, y, z)
        elif sym == "}":
            if poly is not None and len(poly) >= 3:
                polys.append((depth, poly))
            poly = None
        elif sym == "@O" or sym == "@o" or sym == "@":
            r = vals[0] if vals else 1.0
            kind = 0 if sym == "@o" else 1
            elems.append((kind, depth, r, x, y, z))
            elem_orders.append(total)
            mark(x, y, z)
        elif sym == "[":
            stack.append((x, y, z, hx, hy, hz, lx, ly, lz, ux, uy, uz,
                          w, depth, poly))
            poly = None
            depth += 1
        elif sym == "]":
            if stack:
                (x, y, z, hx, hy, hz, lx, ly, lz,
                 ux, uy, uz, w, depth, poly) = stack.pop()
        elif sym == "%":
            skip = True
            skip_depth = 0
            poly = None
    max_depth = max((k[0] for k in groups), default=0)
    return groups, polys, elems, elem_orders, total, \
        (minx, miny, minz, maxx, maxy, maxz), max_depth, width_explicit


# ---- camera & lighting primitives (pure math) -------------------------------


def cam_basis(yaw_deg: float, pitch_deg: float):
    yr, pr = math.radians(yaw_deg), math.radians(pitch_deg)
    return math.cos(yr), math.sin(yr), math.cos(pr), math.sin(pr)


def project_point(x, y, z, cy, sy, cp, sp, persp: float | None = None):
    """Yaw about Y, pitch about X. Returns (screen_x, screen_y_up, depth);
    larger depth = closer to viewer. With persp=D, divide pf = D/(D−t)."""
    ax = x * cy + z * sy
    az1 = -x * sy + z * cy
    vy = y * cp - az1 * sp
    t = y * sp + az1 * cp
    if persp is not None and persp > 0.0:
        pf = persp / max(persp - t, persp * 0.05)
        return ax * pf, vy * pf, t
    return ax, vy, t


def light_dir(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    ar, er = math.radians(az_deg), math.radians(el_deg)
    ce = math.cos(er)
    return (ce * math.sin(ar), math.sin(er), ce * math.cos(ar))


def persp_distance(bbox3, fov_deg: float) -> float | None:
    if fov_deg <= 0.0:
        return None
    minx, miny, minz, maxx, maxy, maxz = bbox3
    r = 0.5 * max(maxx - minx, maxy - miny, maxz - minz, 1e-6)
    return r / math.tan(math.radians(fov_deg) * 0.5) + r


# ---- growth analysis (D0L matrix method) -----------------------------------


def growth_analysis(rules, axiom_mods, steps=16):
    """Exact vₙ₊₁ = M·vₙ for context-free, condition-free systems."""
    if any(r.lctx or r.rctx or r.cond is not None for r in rules):
        return None
    letters, idx = [], {}

    def add(s):
        if s not in idx:
            idx[s] = len(letters)
            letters.append(s)

    for s, _ in axiom_mods:
        add(s)
    for r in rules:
        add(r.sym)
        for _, mods in r.variants:
            for s, _ in mods:
                add(s)
    n = len(letters)
    if n == 0:
        return None
    M = [[0.0] * n for _ in range(n)]
    for r in rules:
        j = idx[r.sym]
        ws = [w if w is not None else 1.0 for w, _ in r.variants]
        tot = sum(ws) or 1.0
        for (w, mods), wi in zip(r.variants, ws):
            cnt: dict[str, int] = {}
            for s, _ in mods:
                cnt[s] = cnt.get(s, 0) + 1
            for s, c in cnt.items():
                M[idx[s]][j] += wi / tot * c
    v = [0.0] * n
    for s, _ in axiom_mods:
        v[idx[s]] += 1
    rows = [(0, sum(v), None)]
    prev = sum(v)
    for k in range(1, steps + 1):
        v = [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]
        t = sum(v)
        rows.append((k, t, (t / prev) if prev else None))
        prev = t
        if t > 1e15:
            break
    u = [0.0] * n
    for s, _ in axiom_mods:
        u[idx[s]] += 1
    lam, prev_m = 1.0, None
    for _ in range(200):
        u2 = [sum(M[i][j] * u[j] for j in range(n)) for i in range(n)]
        m = max(abs(x) for x in u2)
        if m == 0:
            break
        if prev_m:
            lam = m / prev_m
        u = [x / m for x in u2]
        prev_m = m
    tot = sum(abs(x) for x in u) or 1.0
    freqs = sorted(((letters[i], abs(u[i]) / tot) for i in range(n)),
                   key=lambda t: -t[1])
    return rows, lam, [f for f in freqs if f[1] > 1e-4]


def _fmt_num(v) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4g}"


def _fmt_module(m) -> str:
    s, vals = m
    return s if not vals else s + "(" + ",".join(_fmt_num(v) for v in vals) + ")"


def derivation_text(axiom, rules, defines, iters, simple, seed,
                    max_steps=10, line_cap=300) -> str:
    rng = random.Random(seed)
    lines = []
    shown = min(iters, max_steps)
    if simple:
        table = simple_tables(rules)
        s = axiom
        for k in range(shown + 1):
            disp = s if len(s) <= line_cap else \
                s[:line_cap] + f" … (+{len(s) - line_cap} symbols)"
            lines.append(f"ω{k}   ({len(s):,} symbols)\n  {disp}")
            s, _ = expand_simple(s, table, 1, rng, MAX_SYMBOLS)
    else:
        by_sym = rules_by_symbol(rules)
        mods = parse_modules(axiom)
        for k in range(shown + 1):
            txt = " ".join(_fmt_module(m) for m in mods)
            disp = txt if len(txt) <= line_cap else txt[:line_cap] + " …"
            lines.append(f"ω{k}   ({len(mods):,} symbols)\n  {disp}")
            mods, tr = _step_complex(mods, by_sym, defines, rng, 200_000)
            if tr:
                lines.append("(symbol cap reached)")
                break
    if iters > shown:
        lines.append(f"… {iters - shown} further step(s) not shown")
    return "\n\n".join(lines)


# ---- presets ----------------------------------------------------------------

_HETERO_AXIOM = "-(90) H r " + "a " * 7 + "l H"

PRESETS: list[tuple[str, dict]] = [
    ("Staghorn Coral", dict(
        axiom="F", iters=5, angle=22.5, jitter=0.0, draw="FG",
        rules="F -> FF-[-F+F+F]+[+F-F-F]",
        trunk_w=2.6, taper=0.70, trunk="#8a4a34", tip="#ffb37a", bg="#0d151b")),
    ("Sea Fan", dict(
        axiom="F", iters=6, angle=25.7, jitter=0.15, draw="FG",
        rules="F -> F[+F]F[-F][F]",
        trunk_w=2.2, taper=0.78, trunk="#7a5560", tip="#ffd9a0", bg="#101820")),
    ("Sea Feather", dict(
        axiom="X", iters=6, angle=22.5, jitter=0.0, draw="FG",
        rules="X -> F-[[X]+X]+F[+FX]-X\nF -> FF",
        trunk_w=2.0, taper=0.72, trunk="#2f6b45", tip="#c9f5a8", bg="#0b1410")),
    ("Antler", dict(
        axiom="F", iters=7, angle=30.0, jitter=0.25, draw="FG",
        rules="F -> F[+F]F[-F]F",
        trunk_w=3.0, taper=0.78, trunk="#cbb79a", tip="#fff1dc", bg="#14131a")),
    ("Neuron", dict(
        axiom="F", iters=6, angle=37.0, jitter=0.50, draw="FG",
        rules="F -> F[+F]F[-F][F]",
        trunk_w=2.2, taper=0.80, trunk="#405fae", tip="#b9f0ff", bg="#0a0d16")),
    ("Coral Atoll", dict(
        axiom="F+F+F+F", iters=3, angle=90.0, jitter=0.0, draw="FG",
        rules="F -> F+F-F-FF+F+F-F",
        trunk_w=1.8, taper=0.94, trunk="#d99a55", tip="#ffe6b8", bg="#0a1a24")),
    ("Sea Dragon", dict(
        axiom="FX", iters=13, angle=90.0, jitter=0.0, draw="FG",
        rules="X -> X+YF+\nY -> -FX-Y",
        trunk_w=1.6, taper=0.97, trunk="#4fa3c7", tip="#f0a1ff", bg="#0c0c1a")),
    ("Pink Sponge", dict(
        axiom="A", iters=8, angle=60.0, jitter=0.0, draw="FGAB",
        rules="A -> B-A-B\nB -> A+B+A",
        trunk_w=1.8, taper=0.90, trunk="#d95f76", tip="#ffc9d4", bg="#140f14")),
    ("Reef Maze", dict(
        axiom="A", iters=6, angle=90.0, jitter=0.0, draw="F",
        rules="A -> +BF-AFA-FB+\nB -> -AF+BFB+FA-",
        trunk_w=1.6, taper=1.00, trunk="#58c9b5", tip="#eafff7", bg="#0a0f14")),
    ("Star Coral", dict(
        axiom="F-F-F-F-F", iters=4, angle=72.0, jitter=0.0, draw="FG",
        rules="F -> F-F++F+F-F-F",
        trunk_w=1.7, taper=0.95, trunk="#ff8a50", tip="#ffe3b3", bg="#131118")),
    ("Ice Coral", dict(
        axiom="F++F++F", iters=4, angle=60.0, jitter=0.0, draw="FG",
        rules="F -> F-F++F-F",
        trunk_w=1.8, taper=0.90, trunk="#7fb8d9", tip="#eaf9ff", bg="#0c1118")),
    ("Veinwork", dict(
        axiom="F", iters=4, angle=30.0, jitter=0.35, draw="FG",
        rules="F -> FF+[+F-F-F]-[-F+F+F]",
        trunk_w=2.8, taper=0.68, trunk="#93312f", tip="#ff8f85", bg="#150d0d")),
    ("Daisy (2D fills)", dict(
        axiom="A(0)", iters=5, angle=90.0, jitter=0.0, draw="F",
        rules="A(t) : t < 5 -> F(0.9) [ +(30) L ] [ -(30) L ] A(t+1)\n"
              "A(t) : t >= 5 -> "
              "[ +(26) P(0) ] [ -(26) P(0) ] [ P(0) ] @O(0.24)\n"
              "L -> [ { . f(0.55) . + (95) f(0.3) . + (170) f(0.55) . } ]\n"
              "P(k) : k < 6 -> "
              "[ +(26) { . f(0.5) . + (145) f(0.14) . } ] P(k+1)\n"
              "P(k) : k >= 6 -> f(0.12)",
        trunk_w=2.2, taper=0.92, trunk="#3d7a3c", tip="#ffd23f",
        bg="#0c1410")),
    ("Sapling (2D fills)", dict(
        axiom="A(0)", iters=7, angle=90.0, jitter=0.0, draw="F",
        rules="A(t) : t < 7 -> F(1) [ +(24) L ] [ -(24) L ] A(t+1)\n"
              "A(t) : t >= 7 -> "
              "[ { . f(0.45) . + (90) f(0.45) . + (90) f(0.45) . "
              "+ (90) f(0.45) . } ]\n"
              "L -> [ { . f(0.5) . + (95) f(0.28) . + (170) f(0.5) . } ]",
        trunk_w=2.4, taper=0.90, trunk="#5a4a32", tip="#8fd05a",
        bg="#0d120c")),
    ("Anabaena (Fibonacci, λ=φ)", dict(
        axiom="a", iters=10, angle=90.0, jitter=0.0, draw="ab",
        rules="a -> a b\nb -> a",
        trunk_w=2.4, taper=1.00, trunk="#2f6b45", tip="#c9f5a8", bg="#0b1410")),
    ("Hypha · tip growth (linear)", dict(
        axiom="-(90) A(0.6)", iters=12, angle=90.0, jitter=0.0, draw="F",
        rules="A(s) : s < 3 -> F(s) A(s*1.5)\n"
              "A(s) : s >= 3 -> F(s) A(0.6)",
        trunk_w=2.0, taper=1.00, trunk="#8a6a3c", tip="#f2e2c4", bg="#101014")),
    ("Parametric Bush (λ=√2)", dict(
        axiom="X(0)", iters=14, angle=18.0, jitter=0.0, draw="F",
        rules="X(t) : t < 1 -> F(0.7)X(t+1)\n"
              "X(t) : t >= 1 -> F(1)[+(25)X(0)][-(25)X(0)]",
        trunk_w=2.2, taper=0.78, trunk="#4c7d3f", tip="#d7f0a0", bg="#0b120c")),
    ("Heterocyst signals (2L)", dict(
        axiom=_HETERO_AXIOM, iters=5, angle=90.0, jitter=0.0, draw="F",
        rules="r < a > l -> !(4.5)F(1)\n"
              "r < a -> r\n"
              "a > l -> l\n"
              "r -> !(1.8)F(1)\n"
              "l -> !(1.8)F(1)\n"
              "a -> !(1.8)F(1)\n"
              "H -> !(4.5)F(1)",
        trunk_w=2.0, taper=1.00, trunk="#3f7d8f", tip="#9fe8ff", bg="#0a1216")),
    ("Whorl Bush (3D)", dict(
        axiom="A(0)", iters=6, angle=26.0, jitter=0.0, draw="F",
        rules="A(t) : t < 5 -> "
              "!(0.9-0.13*t)F(1)[&(26)A(t+1)][/(120)&(26)A(t+1)][/(240)&(26)A(t+1)]\n"
              "A(t) : t >= 5 -> !(0.25)F(1)",
        trunk_w=2.0, taper=1.00, trunk="#7a5a3a", tip="#ffe9b0", bg="#121009",
        trop=(0.0, -1.0, 0.0), elast=0.22)),
    ("Berry Bush (3D elements)", dict(
        axiom="A(0)", iters=6, angle=27.0, jitter=0.0, draw="F",
        rules="A(t) : t < 5 -> "
              "!(0.85-0.12*t)F(1)[&(27)A(t+1)][/(120)&(27)A(t+1)]"
              "[/(240)&(27)A(t+1)]\n"
              "A(t) : t >= 5 -> "
              "!(0.28)F(0.3)[+(30)f(0.16)@O(0.11)][-(30)f(0.16)@O(0.11)]"
              "f(0.12)@O(0.13)",
        trunk_w=2.0, taper=1.00, trunk="#4a3b2a", tip="#d94f3d",
        bg="#101014", trop=(0.0, -1.0, 0.0), elast=0.18)),
    ("Phyllotaxis (3D)", dict(
        axiom="A(0)", iters=11, angle=25.0, jitter=0.0, draw="F",
        rules="A(t) : t < 8 -> !(0.7-0.06*t)F(0.9)/(137.5)[&(38)B(0)]A(t+1)\n"
              "A(t) : t >= 8 -> !(0.2)F(1.1)\n"
              "B(s) : s < 3 -> !(0.22)F(0.5)[&(42)B(s+1)][/(180)&(42)B(s+1)]\n"
              "B(s) : s >= 3 -> !(0.15)F(0.4)",
        trunk_w=2.0, taper=1.00, trunk="#3e5d33", tip="#d9f29a", bg="#0b100c",
        trop=(0.0, -1.0, 0.0), elast=0.12)),
]


# ---- self-test suite (engine only, no Qt) -----------------------------------


def run_selftest() -> int:
    failures: list[str] = []
    passed = 0

    def check(name, fn):
        nonlocal passed
        try:
            fn()
            passed += 1
            print(f"  ok   {name}")
        except AssertionError as e:
            failures.append(f"{name}: {e or 'assertion failed'}")
            print(f"  FAIL {name}: {e or 'assertion failed'}")
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"  FAIL {name}: {type(e).__name__}: {e}")

    def t_fib():
        rules, _ = parse_program("a -> a b\nb -> a")
        rng = random.Random(0)
        s = "a"
        lens = [len(s)]
        for _ in range(4):
            s, _t = expand_simple(s, simple_tables(rules), 1, rng, MAX_SYMBOLS)
            lens.append(len(s))
        assert lens == [1, 2, 3, 5, 8], lens

    def t_lambda_phi():
        rules, _ = parse_program("a -> a b\nb -> a")
        res = growth_analysis(rules, parse_modules("a"), steps=8)
        assert res is not None
        _rows, lam, _f = res
        assert abs(lam - (1 + math.sqrt(5)) / 2) < 1e-3, lam

    def t_lctx():
        rules, _ = parse_program("a < b -> c")
        mods, _t = expand_complex(parse_modules("a b b"),
                                  rules_by_symbol(rules), {}, random.Random(0),
                                  1, MAX_SYMBOLS)
        assert "".join(s for s, _ in mods) == "acb", mods

    def t_rctx():
        rules, _ = parse_program("b > a -> d")
        mods, _t = expand_complex(parse_modules("b a b"),
                                  rules_by_symbol(rules), {}, random.Random(0),
                                  1, MAX_SYMBOLS)
        assert "".join(s for s, _ in mods) == "dab", mods

    def t_2l_beads():
        text = ("r < a > l -> !(4.5)F(1)\nr < a -> r\na > l -> l\n"
                "r -> !(1.8)F(1)\nl -> !(1.8)F(1)\na -> !(1.8)F(1)\n"
                "H -> !(4.5)F(1)")
        rules, _ = parse_program(text)
        axiom = parse_modules("H r " + "a " * 7 + "l H")
        by = rules_by_symbol(rules)

        def beads(n):
            mods, _t = expand_complex(axiom, by, {}, random.Random(0), n,
                                      MAX_SYMBOLS)
            return sum(1 for s, ps in mods
                       if s == "!" and ps and abs(ps[0] - 4.5) < 1e-9)
        assert beads(3) == 2, beads(3)
        assert beads(4) == 3, beads(4)

    def t_parametric_hypha():
        rules, _ = parse_program("A(s) : s < 3 -> F(s) A(s*1.5)\n"
                                 "A(s) : s >= 3 -> F(s) A(0.6)")
        mods, _t = expand_complex(parse_modules("A(0.6)"),
                                  rules_by_symbol(rules), {}, random.Random(0),
                                  3, MAX_SYMBOLS)
        fs = [m for m in mods if m[0] == "F"]
        assert len(fs) == 3, mods
        assert mods[-1][0] == "A" and abs(mods[-1][1][0] - 2.025) < 1e-9, mods[-1]

    def t_define():
        rules, defines = parse_program("#define k 3\nA -> F(k*2)")
        mods, _t = expand_complex(parse_modules("A"), rules_by_symbol(rules),
                                  defines, random.Random(0), 1, MAX_SYMBOLS)
        assert mods == [("F", (6.0,))], mods

    def t_stochastic():
        rules, _ = parse_program("a -> 0.5 F ; 0.5 G")
        table = simple_tables(rules)
        s1, _ = expand_simple("a", table, 6, random.Random(7), MAX_SYMBOLS)
        s2, _ = expand_simple("a", table, 6, random.Random(7), MAX_SYMBOLS)
        assert len(s1) == 64 and set(s1) <= set("FG")
        assert s1 == s2
        assert 18 <= s1.count("F") <= 46, s1.count("F")

    def t_cap():
        rules, _ = parse_program("a -> aa")
        _s, trunc = expand_simple("a", simple_tables(rules), 3,
                                  random.Random(0), 10)
        assert trunc

    def t_turtle_square():
        g, polys, el, eo, total, bbox, _md, _w = interpret(
            "F+F+F+F", 90.0, 0.0, 0, frozenset("F"))
        assert total == 4 and not polys and not el
        assert abs((bbox[3] - bbox[0]) - 1) < 1e-9
        assert abs((bbox[4] - bbox[1]) - 1) < 1e-9

    def t_polygon():
        _g, polys, el, _eo, total, bbox, _md, _w = interpret(
            "{.f(1).+(90)f(1).+(90)f(1).}", 90.0, 0.0, 0, frozenset("F"))
        assert total == 0 and len(polys) == 1 and not el, (total, polys)
        verts = polys[0][1]
        assert len(verts) == 4, verts
        assert abs((max(v[0] for v in verts) - min(v[0] for v in verts)) - 1) \
            < 1e-9
        assert abs(bbox[3] - bbox[0] - 1) < 1e-9

    def t_elements():
        _g, polys, el, eo, total, _bbox, _md, _w = interpret(
            "F(1)@O(0.5)f(1)@o(0.25)", 90.0, 0.0, 0, frozenset("F"))
        assert total == 1 and not polys, (total, polys)
        assert len(el) == 2 and eo == [1, 1], (el, eo)
        kind0, depth0, r0, x0, y0, z0 = el[0]
        assert kind0 == 1 and depth0 == 0 and abs(r0 - 0.5) < 1e-9
        assert abs(x0) < 1e-9 and abs(y0 - 1.0) < 1e-9 and abs(z0) < 1e-9
        kind1, _d1, r1, _x1, y1, _z1 = el[1]
        assert kind1 == 0 and abs(r1 - 0.25) < 1e-9 and abs(y1 - 2.0) < 1e-9

    def t_perspective():
        cy, sy, cp, sp = cam_basis(0.0, 0.0)
        ax_near, _v, _t = project_point(1, 0, 5, cy, sy, cp, sp, persp=10.0)
        ax_far, _v, _t = project_point(1, 0, -5, cy, sy, cp, sp, persp=10.0)
        assert ax_near > 1.0, ax_near
        assert 0.0 < ax_far < 1.0, ax_far

    def t_expr():
        v = eval_expr(parse_expr("2*(3+4)^2 - 10/4"), {}, random.Random(0))
        assert abs(v - 95.5) < 1e-9, v

    def t_grow_expected():
        rules, _ = parse_program("a -> 0.5 a a ; 0.5 a")
        res = growth_analysis(rules, parse_modules("a"), steps=3)
        assert res is not None and res[0][2][1] == 1.25, res

    def t_depth_model():
        cy, sy, cp, sp = cam_basis(0.0, 0.0)
        assert project_point(0, 0, 1, cy, sy, cp, sp)[2] > \
            project_point(0, 0, -1, cy, sy, cp, sp)[2]
        cy, sy, cp, sp = cam_basis(0.0, 90.0)
        assert project_point(0, 1, 0, cy, sy, cp, sp)[2] > \
            project_point(0, -1, 0, cy, sy, cp, sp)[2]

    def t_line_numbers():
        try:
            parse_program("A -> F(2)\nB -> G(\n")
        except LSystemError as e:
            assert "line 2" in str(e), str(e)
            return
        raise AssertionError("expected LSystemError")

    def t_word_freq():
        req = ComputeRequest(axiom="aab", rules_text="", step=0, angle=0.0,
                             jitter=0.0, seed=0, draw="", trop=(0.0, 0.0, 0.0),
                             elast=0.0)
        res = compute_system(req)
        assert res.ok and res.word_freq[:2] == [("a", 2), ("b", 1)], \
            res.word_freq

    print(f"{APP_NAME} {__version__} — engine self-test")
    check("fibonacci word lengths [1,2,3,5,8]", t_fib)
    check("Perron–Frobenius λ = φ", t_lambda_phi)
    check("left context (a<b -> c)", t_lctx)
    check("right context (b>a -> d)", t_rctx)
    check("2L signal crossing → 3 beads", t_2l_beads)
    check("parametric conditional (hypha)", t_parametric_hypha)
    check("#define constant", t_define)
    check("stochastic determinism", t_stochastic)
    check("symbol cap truncation", t_cap)
    check("turtle: closed unit square", t_turtle_square)
    check("polygon capture {. . . .}", t_polygon)
    check("elements @O/@o capture", t_elements)
    check("perspective projection divide", t_perspective)
    check("expression evaluator", t_expr)
    check("stochastic expected growth", t_grow_expected)
    check("camera depth model (painter order)", t_depth_model)
    check("grammar errors report line numbers", t_line_numbers)
    check("compute_system word frequency", t_word_freq)
    total = passed + len(failures)
    print(f"{passed}/{total} checks passed")
    return 1 if failures else 0


# ---- compute pipeline (shared by GUI, CLI, tests) ---------------------------


@dataclass(frozen=True, slots=True)
class ComputeRequest:
    axiom: str
    rules_text: str
    step: int
    angle: float
    jitter: float
    seed: int
    draw: str
    trop: tuple[float, float, float]
    elast: float


@dataclass(slots=True)
class ComputeResult:
    gen: int = 0
    step: int = 0
    ok: bool = False
    error: str = ""
    detail: str = ""
    groups: dict = field(default_factory=dict)
    polys: list = field(default_factory=list)
    elems: tuple = ([], [])
    total: int = 0
    bbox3: tuple = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    max_depth: int = 0
    width_explicit: bool = False
    symbol_count: int = 0
    truncated: bool = False
    is3d: bool = False
    class_text: str = ""
    alphabet: int = 0
    growth: tuple | None = None
    word_freq: list = field(default_factory=list)   # [(sym, count)] top-12
    ctx: tuple | None = None


def compute_system(req: ComputeRequest, cancel=None) -> ComputeResult:
    """Pure compute pipeline: parse → derive → plot → analyse."""
    res = ComputeResult(step=req.step)
    try:
        rules, defines = parse_program(req.rules_text)
        axiom_mods = parse_modules(req.axiom)
    except LSystemError as e:
        raise LSystemError(str(e)) from None
    if not axiom_mods:
        raise LSystemError("axiom is empty")
    by_sym = rules_by_symbol(rules)
    simple = is_simple(rules, axiom_mods)
    rng = random.Random(req.seed)
    if simple:
        table = simple_tables(rules)
        mods, trunc = expand_simple(req.axiom, table, req.step, rng,
                                    MAX_SYMBOLS, cancel)
    else:
        mods, trunc = expand_complex(axiom_mods, by_sym, defines, rng,
                                     req.step, MAX_SYMBOLS, cancel)
    if cancel is not None and cancel():
        raise OperationCancelled()
    groups, polys, elems, elem_orders, total, bbox3, maxd, wex = interpret(
        mods, req.angle, req.jitter, req.seed, frozenset(req.draw),
        req.trop, req.elast, cancel)
    growth = growth_analysis(rules, axiom_mods, steps=min(req.step + 6, 24))
    stoch = any(len(r.variants) > 1 or
                any(w is not None for w, _ in r.variants) for r in rules)
    wf = Counter(s for s, _ in mods if len(s) == 1 and s.isalpha())
    res.ok = True
    res.groups = groups
    res.polys = polys
    res.elems = (elems, elem_orders)
    res.total = total
    res.bbox3 = bbox3
    res.max_depth = maxd
    res.width_explicit = wex
    res.symbol_count = len(mods)
    res.truncated = trunc
    res.is3d = (bbox3[5] - bbox3[2]) > 1e-9
    res.class_text = classify(rules)
    res.alphabet = len(collect_alphabet(rules, axiom_mods))
    res.growth = (growth[0], growth[1], growth[2], stoch) if growth else None
    res.word_freq = wf.most_common(12)
    res.ctx = (req.axiom, rules, defines, simple)
    return res


# ===========================================================================
# GUI (Qt) — everything below requires PySide6
# ===========================================================================

if "--selftest" in sys.argv:
    setup_logging(logging.WARNING)
    sys.exit(run_selftest())

try:
    from PySide6.QtCore import (QRect, QSize, Qt, QPointF, QRectF, QTimer,
                                QObject, QRunnable, QSettings, Signal,
                                QThreadPool)
    from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QFontDatabase,
                               QIcon, QImage, QKeySequence, QLineF, QPainter,
                               QPen, QPixmap, QRadialGradient, QShortcut,
                               QSyntaxHighlighter, QTextCharFormat, QPolygonF)
    from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog,
                                   QComboBox, QDialog, QDialogButtonBox,
                                   QDoubleSpinBox, QFileDialog, QFormLayout,
                                   QGroupBox, QHBoxLayout, QInputDialog,
                                   QLabel, QLineEdit, QMainWindow, QMenu,
                                   QMessageBox, QPlainTextEdit, QPushButton,
                                   QSizePolicy, QSlider, QSpinBox, QSplitter,
                                   QTabWidget, QTextEdit, QToolBar,
                                   QVBoxLayout, QWidget)
except ImportError:
    sys.stderr.write(
        f"{APP_NAME} requires PySide6 — install it with:\n"
        "    pip install PySide6\n"
        "(the engine self-test works without Qt:  python "
        f"{os.path.basename(sys.argv[0])} --selftest)\n")
    sys.exit(1)


# ---- syntax highlighting ----------------------------------------------------


class RuleHighlighter(QSyntaxHighlighter):
    def __init__(self, doc):
        super().__init__(doc)

        def fmt(color, bold=False, italic=False):
            f = QTextCharFormat()
            f.setForeground(QBrush(QColor(color)))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            if italic:
                f.setFontItalic(True)
            return f

        self.f_comment = fmt("#5c6a7a", italic=True)
        self.f_keyword = fmt("#4d9fe8", bold=True)
        self.f_symbol = fmt("#e8b04d", bold=True)
        self.f_number = fmt("#7fd6a4")
        self.f_ctx = fmt("#d98cd9")
        self.f_bracket = fmt("#7a8798")
        self.re_lhs = re.compile(r"^\s*([A-Za-z@])(?=\s*(?:\(|<|>|:|->))")
        self.re_arrow = re.compile(r"->")
        self.re_num = re.compile(r"\d+(?:\.\d+)?")
        self.re_ctx = re.compile(r"[<>]")
        self.re_br = re.compile(r"[\[\](){}.]")
        self.re_define = re.compile(r"^\s*(#define)\s+([A-Za-z_]\w*)")

    def highlightBlock(self, text: str):
        if text.lstrip().startswith("#define"):
            m = self.re_define.match(text)
            if m:
                self.setFormat(m.start(1), 7, self.f_keyword)
                self.setFormat(m.start(2), m.end(2) - m.start(2), self.f_symbol)
                rest = text[m.end(2):]
                for nm in self.re_num.finditer(rest):
                    self.setFormat(m.end(2) + nm.start(),
                                   nm.end() - nm.start(), self.f_number)
            return
        ci = text.find("#")
        code = text if ci < 0 else text[:ci]
        for m in self.re_lhs.finditer(code):
            self.setFormat(m.start(1), 1, self.f_symbol)
        for m in self.re_arrow.finditer(code):
            self.setFormat(m.start(), 2, self.f_keyword)
        for m in self.re_num.finditer(code):
            self.setFormat(m.start(), m.end() - m.start(), self.f_number)
        for m in self.re_ctx.finditer(code):
            self.setFormat(m.start(), 1, self.f_ctx)
        for m in self.re_br.finditer(code):
            self.setFormat(m.start(), 1, self.f_bracket)
        if ci >= 0:
            self.setFormat(ci, len(text) - ci, self.f_comment)


# ---- mesh renderer -----------------------------------------------------------

def _project(x, y, z, cy, sy, cp, sp, persp=None):
    ax, ay, _t = project_point(x, y, z, cy, sy, cp, sp, persp)
    return ax, ay


def projected_bbox(bbox3, yaw_deg: float, pitch_deg: float, persp=None):
    cy, sy, cp, sp = cam_basis(yaw_deg, pitch_deg)
    minx, miny, minz, maxx, maxy, maxz = bbox3
    pts = [_project(X, Y, Z, cy, sy, cp, sp, persp)
           for X in (minx, maxx) for Y in (miny, maxy) for Z in (minz, maxz)]
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    return (min(xs), min(ys), max(xs), max(ys)), (cy, sy, cp, sp)


class MeshDraw:
    __slots__ = ("passes", "glow")


def build_mesh(groups, polys=None, elems=None, *, trunk: QColor, tip: QColor,
               bg: QColor, trunk_w: float, taper: float, width_explicit: bool,
               max_depth: int, yaw: float, pitch: float, rscale: float,
               glow: bool = False, cap: int = MESH_LIVE_MAX, buckets: int = 64,
               shades: int = MESH_SHADES, hi_min_px: float = 2.4,
               light_w=(0.413, 0.766, -0.492), persp: float | None = None,
               color_by: str = "depth", bbox3=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
               fog: float = 0.0, ambient: float = 0.18,
               fill_alpha: float = 0.82):
    """Shaded-tube mesh with polygons and elements (thread-safe)."""
    nseg = sum(len(g[0]) for g in groups.values())
    elist, eorders = elems if elems else ([], [])
    nelem = len(elist)
    if (not nseg and not polys and not nelem) or nseg + nelem > cap:
        return None
    cy, sy, cp, sp = cam_basis(yaw, pitch)
    lw = light_w
    ln = math.sqrt(lw[0] ** 2 + lw[1] ** 2 + lw[2] ** 2) or 1.0
    lwx, lwy, lwz = lw[0] / ln, lw[1] / ln, lw[2] / ln
    s2x = lwx * cy + lwz * sy
    az1 = -lwx * sy + lwz * cy
    s2y = lwy * cp - az1 * sp
    sl = math.hypot(s2x, s2y)
    l2x, l2y = (s2x / sl, s2y / sl) if sl > 1e-9 else (-0.6, 0.8)

    hmin, hmax = bbox3[1], bbox3[4]
    use_h = color_by == "height" and (hmax - hmin) > 1e-9
    hspan = (hmax - hmin) if use_h else 1.0

    def grad_t(depth: int, ymid: float) -> float:
        if use_h:
            return min(max((ymid - hmin) / hspan, 0.0), 1.0)
        return depth / max_depth if max_depth else 0.0

    def base_col(t: float) -> tuple[int, int, int]:
        return (round(trunk.red() + (tip.red() - trunk.red()) * t),
                round(trunk.green() + (tip.green() - trunk.green()) * t),
                round(trunk.blue() + (tip.blue() - trunk.blue()) * t))

    br = (bg.red(), bg.green(), bg.blue())
    fog_on_use = fog > 0.001 and (bbox3[5] - bbox3[2]) > 1e-9

    def fog_mix(rgb, f):
        if f <= 0.0:
            return rgb
        return (round(rgb[0] * (1 - f) + br[0] * f),
                round(rgb[1] * (1 - f) + br[1] * f),
                round(rgb[2] * (1 - f) + br[2] * f))

    pen_cache: dict = {}

    def tube_pen(ck) -> QPen:
        pen = pen_cache.get(ck)
        if pen is None:
            cr, cg, cb, wq = ck
            pen = QPen(QColor(cr, cg, cb), max(wq, 1e-4))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen_cache[ck] = pen
        return pen

    keys = list(groups.keys())
    gradii = []
    for depth, wq in keys:
        wpx = wq if width_explicit else trunk_w * (taper ** depth)
        gradii.append(max(wpx * 0.5 * rscale, 1e-6))

    recs = []
    zmin = zmax = None
    glow_segs: dict[int, list] = {}
    for gidx, key in enumerate(keys):
        depth, _wq = key
        r = gradii[gidx]
        can_hi = (2.0 * r / rscale) >= hi_min_px
        if glow:
            glow_segs[gidx] = (gl, base_col(grad_t(depth, 0.0)))
        for (x1, y1, z1, x2, y2, z2) in groups[key][0]:
            dxw, dyw, dzw = x2 - x1, y2 - y1, z2 - z1
            dwl = math.sqrt(dxw * dxw + dyw * dyw + dzw * dzw)
            lam = abs((dxw * lwx + dyw * lwy + dzw * lwz) / dwl) \
                if dwl > 1e-12 else 1.0
            shade = int(lam * (shades - 1) + 0.5)
            a = project_point(x1, y1, z1, cy, sy, cp, sp, persp)
            b = project_point(x2, y2, z2, cy, sy, cp, sp, persp)
            zv = 0.5 * (a[2] + b[2])
            if zmin is None or zv < zmin: zmin = zv
            if zmax is None or zv > zmax: zmax = zv
            recs.append((zv, gidx, shade, a[0], a[1], b[0], b[1], can_hi, r,
                         0.5 * (y1 + y2)))
            if glow:
                gl.append((a[0], a[1], b[0], b[1]))

    poly_recs = []
    for (pdepth, verts) in (polys or []):
        ymid = sum(v[1] for v in verts) / len(verts)
        (ux_, uy_, uz_), (vx_, vy_, vz_), (wx_, wy_, wz_) = \
            verts[0], verts[1], verts[2]
        e1 = (vx_ - ux_, vy_ - uy_, vz_ - uz_)
        e2 = (wx_ - ux_, wy_ - uy_, wz_ - uz_)
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        lam = abs((nx * lwx + ny * lwy + nz * lwz) / nl) if nl > 1e-12 else 0.35
        pts2 = []
        zsum = 0.0
        for (px, py, pz) in verts:
            ax, ay, zv = project_point(px, py, pz, cy, sy, cp, sp, persp)
            pts2.append((ax, ay))
            zsum += zv
        zv = zsum / len(pts2)
        poly_recs.append((zv, grad_t(pdepth, ymid), lam, pts2))
        if zmin is None or zv < zmin: zmin = zv
        if zmax is None or zv > zmax: zmax = zv

    elem_recs = []
    for (kind, edepth, er, ex, ey, ez) in elist:
        ax, ay, zv = project_point(ex, ey, ez, cy, sy, cp, sp, persp)
        elem_recs.append((zv, kind, grad_t(edepth, ey), er, ax, ay))
        if zmin is None or zv < zmin: zmin = zv
        if zmax is None or zv > zmax: zmax = zv

    if zmin is None:
        return None
    span = (zmax - zmin) or 1.0
    use_b = (zmax - zmin) > 1e-9

    def bucket(zv) -> int:
        return int((zv - zmin) / span * (buckets - 1)) if use_b else 0

    def pf_at(b: int) -> float:
        if not persp:
            return 1.0
        zc = zmin + (b + 0.5) / buckets * span
        return persp / max(persp - zc, persp * 0.05)

    def fog_at(zv) -> float:
        if not fog_on_use:
            return 0.0
        return fog * (zv - zmin) / span

    tube_by_b: dict = {}
    hi_by_b: dict = {}
    for (zv, gidx, shade, ax, ay, bx, by, can_hi, r, ymid) in recs:
        depth = keys[gidx][0]
        b = bucket(zv)
        rgb = base_col(grad_t(depth, ymid))
        L = shade / (shades - 1)
        tone = ambient + (1.0 - ambient) * L
        f = fog_at(zv)
        cr, cg, cb = fog_mix((round(rgb[0] * tone), round(rgb[1] * tone),
                              round(rgb[2] * tone)), f)
        pf = pf_at(b)
        w_world = 2.0 * r * pf
        ck = (cr, cg, cb, round(w_world * 8) / 8.0)
        tube_by_b.setdefault(b, {}).setdefault(ck, []).append(
            QLineF(ax, ay, bx, by))
        if can_hi:
            dx, dy = bx - ax, by - ay
            dln = math.hypot(dx, dy)
            if dln > 1e-9:
                pxn, pyn = -dy / dln, dx / dln
                d = pxn * l2x + pyn * l2y
                if abs(d) > 0.12:
                    off = r * 0.42 * pf * (1.0 if d > 0 else -1.0)
                    sc = QColor(cr, cg, cb).lighter(175)
                    hck = (sc.red(), sc.green(), sc.blue(),
                           round(r * 1.05 * pf * 8) / 8.0)
                    hi_by_b.setdefault(b, {}).setdefault(hck, []).append(
                        QLineF(ax + pxn * off, ay + pyn * off,
                               bx + pxn * off, by + pyn * off))

    poly_by_b: dict = {}
    for (zv, t, lam, pts2) in poly_recs:
        b = bucket(zv)
        rgb = base_col(t)
        tone = ambient + (1.0 - ambient) * (0.35 + 0.65 * lam)
        cr, cg, cb = fog_mix((round(rgb[0] * tone), round(rgb[1] * tone),
                              round(rgb[2] * tone)), fog_at(zv))
        fc = QColor(cr, cg, cb, round(255 * fill_alpha))
        op = QPen(QColor(cr, cg, cb).darker(150), 0)
        poly = QPolygonF([QPointF(ax, ay) for (ax, ay) in pts2])
        poly_by_b.setdefault(b, []).append((QBrush(fc), op, poly))

    ball_by_b: dict = {}
    ring_by_b: dict = {}
    for (zv, kind, t, er, ax, ay) in elem_recs:
        b = bucket(zv)
        rgb = base_col(t)
        cr, cg, cb = fog_mix(rgb, fog_at(zv))
        if kind:
            ck = (cr, cg, cb, round(er * 8) / 8.0)
            ball_by_b.setdefault(b, {}).setdefault(ck, []).append((ax, ay))
        else:
            w_world = max(0.18 * er, 0.02) * pf_at(b)
            ck = (cr, cg, cb, round(w_world * 8) / 8.0)
            ring_by_b.setdefault(b, {}).setdefault(ck, []).append(
                ((ax, ay), er * pf_at(b)))

    def materialize_balls(b):
        out = []
        for (cr, cg, cb, rq), centers in ball_by_b.get(b, {}).items():
            r = max(rq, 1e-4)
            g = QRadialGradient(QPointF(l2x * 0.45 * r, l2y * 0.45 * r),
                                1.35 * r)
            g.setColorAt(0.0, QColor(cr, cg, cb).lighter(165))
            g.setColorAt(0.55, QColor(cr, cg, cb))
            g.setColorAt(1.0, QColor(round(cr * 0.35), round(cg * 0.35),
                                     round(cb * 0.35)))
            out.append((g, r, centers))
        return out

    def materialize_rings(b):
        out = []
        for (cr, cg, cb, wq), items in ring_by_b.get(b, {}).items():
            pen = QPen(QColor(cr, cg, cb), max(wq, 1e-4))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            out.append((pen, items))
        return out

    mesh = MeshDraw()
    all_b = sorted(set(tube_by_b) | set(poly_by_b) | set(ball_by_b) |
                   set(ring_by_b))
    passes = []
    for b in all_b:
        tubes = [(tube_pen(ck), lines)
                 for ck, lines in tube_by_b.get(b, {}).items()]
        stripes = [(tube_pen(ck), lines)
                   for ck, lines in hi_by_b.get(b, {}).items()]
        passes.append((tubes, stripes, poly_by_b.get(b, []),
                       materialize_balls(b), materialize_rings(b)))
    mesh.passes = passes
    mesh.glow = []
    if glow:
        for gidx in sorted(glow_segs):
            gl, rgb = glow_segs[gidx]
            _depth, wq = keys[gidx]
            wpx = wq if width_explicit else trunk_w * (taper ** _depth)
            gpen = QPen(QColor(rgb[0], rgb[1], rgb[2], 42),
                        max((wpx * 3.2 + 2.2) * rscale, 1e-4))
            gpen.setCapStyle(Qt.PenCapStyle.RoundCap)
            mesh.glow.append((gpen, [QLineF(ax, ay, bx, by)
                                     for (ax, ay, bx, by) in gl]))
    return mesh


def draw_mesh(p: QPainter, mesh: MeshDraw) -> None:
    for pen, lines in mesh.glow:
        p.setPen(pen)
        p.drawLines(lines)
    for tubes, stripes, poly_triples, balls, rings in mesh.passes:
        for pen, lines in tubes:
            p.setPen(pen)
            p.drawLines(lines)
        for pen, lines in stripes:
            p.setPen(pen)
            p.drawLines(lines)
        for brush, open_pen, polyf in poly_triples:
            p.setBrush(brush)
            p.setPen(open_pen)
            p.drawPolygon(polyf)
        p.setPen(Qt.PenStyle.NoPen)
        for grad, r, centers in balls:
            p.setBrush(QBrush(grad))
            for (cx, cyc) in centers:
                p.save()
                p.translate(cx, cyc)
                p.drawEllipse(QPointF(0, 0), r, r)
                p.restore()
        for pen, items in rings:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(pen)
            for (cx, cyc), r in items:
                p.drawEllipse(QPointF(cx, cyc), r, r)


def paint_backdrop(p: QPainter, rect, bg: QColor) -> None:
    g = QRadialGradient(QPointF(rect.center().x(), rect.center().y()),
                        max(rect.width(), rect.height()) * 0.78)
    g.setColorAt(0.0, bg.lighter(112))
    g.setColorAt(1.0, bg.darker(105))
    p.fillRect(rect, QBrush(g))


def draw_ground_shadow(p: QPainter, bbox3, yaw: float, pitch: float,
                       persp=None) -> None:
    minx, miny, minz, maxx, maxy, maxz = bbox3
    if (maxz - minz) < 1e-9 or pitch <= 8.0:
        return
    cy, sy, cp, sp = cam_basis(yaw, pitch)
    pts = [_project(X, miny, Z, cy, sy, cp, sp, persp)
           for X in (minx, maxx) for Z in (minz, maxz)]
    pxs = [q[0] for q in pts]
    pys = [q[1] for q in pts]
    bcx, bcy = (min(pxs) + max(pxs)) / 2.0, (min(pys) + max(pys)) / 2.0
    rad = max(max(pxs) - min(pxs), max(pys) - min(pys)) * 0.62
    if rad <= 0.0:
        return
    g = QRadialGradient(QPointF(bcx, bcy), rad * 1.15)
    g.setColorAt(0.0, QColor(0, 0, 0, 95))
    g.setColorAt(0.55, QColor(0, 0, 0, 48))
    g.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(g))
    p.drawEllipse(QPointF(bcx, bcy), rad, rad * 0.42)
    p.restore()


def _wire_elem_paint(p: QPainter, elems, count, trunk: QColor, tip: QColor,
                     max_depth: int, fill_alpha: float, zoom: float,
                     cy, sy, cp, sp, persp) -> None:
    """Wire-mode elements. `count=None` draws all; otherwise an int stroke
    budget clips by plot order (elements carry their own order list)."""
    elist, eorders = elems
    if not elist:
        return
    if count is None:
        k = len(elist)
    elif eorders:
        k = bisect_right(eorders, count - 1)
    else:
        k = 0
    for (kind, edepth, er, ex, ey, ez) in elist[:k]:
        ax, ay, t = project_point(ex, ey, ez, cy, sy, cp, sp, persp)
        pf = persp / max(persp - t, persp * 0.05) if persp else 1.0
        tt = edepth / max_depth if max_depth else 0.0
        c = QColor(round(trunk.red() + (tip.red() - trunk.red()) * tt),
                   round(trunk.green() + (tip.green() - trunk.green()) * tt),
                   round(trunk.blue() + (tip.blue() - trunk.blue()) * tt))
        r = er * pf
        if kind:
            fc = QColor(c.red(), c.green(), c.blue(),
                        round(255 * fill_alpha))
            p.setBrush(QBrush(fc))
            p.setPen(QPen(QColor(c).darker(150), 0))
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(c, max(0.8, 0.22 * r * zoom))
            pen.setCosmetic(True)
            p.setPen(pen)
        p.drawEllipse(QPointF(ax, ay), r, r)


def render_image(groups, polys, elems, total, bbox3, *, trunk: QColor,
                 tip: QColor, bg: QColor, trunk_w: float, taper: float,
                 width_explicit: bool, glow: bool, max_depth: int,
                 target: int = 2048, transparent: bool = False,
                 yaw: float = 0.0, pitch: float = 0.0, quality: str = "auto",
                 persp: float | None = None,
                 light_w=(0.413, 0.766, -0.492), color_by: str = "depth",
                 fog: float = 0.0, ambient: float = AMBIENT_DEFAULT / 100.0,
                 fill_alpha: float = FILL_OPACITY_DEFAULT / 100.0) -> QImage:
    """Auto-fitted offscreen render (thread-safe). quality: auto/mesh/wire."""
    if not total and not polys and not (elems and elems[0]):
        return QImage()
    (minx, miny, maxx, maxy), (cy, sy, cp, sp) = \
        projected_bbox(bbox3, yaw, pitch, persp)
    bw = max(maxx - minx, 1e-9)
    bh = max(maxy - miny, 1e-9)
    m = target * 0.025
    if bw >= bh:
        w, h = target, max(2, round(target * bh / bw))
    else:
        h, w = target, max(2, round(target * bw / bh))
    img = QImage(w, h, QImage.Format.Format_ARGB32 if transparent
                 else QImage.Format.Format_RGB32)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    if not transparent:
        paint_backdrop(p, QRect(0, 0, w, h), bg)
    z = min((w - 2 * m) / bw, (h - 2 * m) / bh)
    p.translate(w / 2, h / 2)
    p.scale(z, -z)
    p.translate(-(minx + maxx) / 2, -(miny + maxy) / 2)
    has3d = (bbox3[5] - bbox3[2]) > 1e-9
    mesh = None
    if quality in ("mesh", "auto"):
        mesh = build_mesh(groups, polys, elems, trunk=trunk, tip=tip, bg=bg,
                          trunk_w=trunk_w, taper=taper,
                          width_explicit=width_explicit, max_depth=max_depth,
                          yaw=yaw, pitch=pitch, rscale=1.0 / z, glow=glow,
                          cap=MESH_EXPORT_CAP, buckets=96, light_w=light_w,
                          persp=persp, color_by=color_by, bbox3=bbox3,
                          fog=fog, ambient=ambient, fill_alpha=fill_alpha)
    if mesh is not None:
        if has3d and pitch > 8.0 and not transparent:
            draw_ground_shadow(p, bbox3, yaw, pitch, persp)
        draw_mesh(p, mesh)
    else:
        def pen_for(key, glow_pass):
            depth, wq = key
            t = depth / max_depth if max_depth else 0.0
            c = QColor(round(trunk.red() + (tip.red() - trunk.red()) * t),
                       round(trunk.green() + (tip.green() - trunk.green()) * t),
                       round(trunk.blue() + (tip.blue() - trunk.blue()) * t))
            wpx = wq if width_explicit else trunk_w * (taper ** depth)
            if glow_pass:
                c.setAlpha(50)
                wpx = wpx * 3.5 + 2.5
            pen = QPen(c, max(0.7, wpx))
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            return pen

        keys = sorted(groups.keys())
        for glow_pass in ((True, False) if glow else (False,)):
            for key in keys:
                segs, _order = groups[key]
                p.setPen(pen_for(key, glow_pass))
                p.drawLines([QLineF(*_project(x1, y1, z1, cy, sy, cp, sp, persp),
                                    *_project(x2, y2, z2, cy, sy, cp, sp, persp))
                             for (x1, y1, z1, x2, y2, z2) in segs])
        for (pdepth, verts) in (polys or []):
            t = pdepth / max_depth if max_depth else 0.0
            c = QColor(round(trunk.red() + (tip.red() - trunk.red()) * t),
                       round(trunk.green() + (tip.green() - trunk.green()) * t),
                       round(trunk.blue() + (tip.blue() - trunk.blue()) * t),
                       round(255 * fill_alpha))
            p.setBrush(QBrush(c))
            p.setPen(QPen(QColor(c).darker(150), 0))
            p.drawPolygon(QPolygonF(
                [QPointF(*_project(px, py, pz, cy, sy, cp, sp, persp))
                 for (px, py, pz) in verts]))
        _wire_elem_paint(p, elems, None, trunk, tip, max_depth, fill_alpha,
                         z, cy, sy, cp, sp, persp)
    p.end()
    return img


def render_preset_qimage(cfg: dict, target: int, glow: bool = False) -> QImage:
    rules, defines = parse_program(cfg["rules"])
    axiom_mods = parse_modules(cfg["axiom"])
    by_sym = rules_by_symbol(rules)
    rng = random.Random(cfg.get("seed", 1))
    iters = cfg["iters"]
    simple = is_simple(rules, axiom_mods)
    table = simple_tables(rules) if simple else None
    for _ in range(6):
        if simple:
            mods, _t = expand_simple(cfg["axiom"], table, iters, rng, 30000)
        else:
            mods, _t = expand_complex(axiom_mods, by_sym, defines, rng, iters,
                                      30000)
        if len(mods) <= 16000 or iters <= 1:
            break
        iters = max(1, iters - max(1, iters // 3))
    groups, polys, elems, _eo, total, bbox3, maxd, wex = interpret(
        mods, cfg["angle"], cfg["jitter"], cfg.get("seed", 1),
        frozenset(cfg["draw"]), tuple(cfg.get("trop", (0.0, 0.0, 0.0))),
        cfg.get("elast", 0.0))
    has3d = (bbox3[5] - bbox3[2]) > 1e-9
    return render_image(groups, polys, (elems, _eo), total, bbox3,
                        trunk=QColor(cfg["trunk"]), tip=QColor(cfg["tip"]),
                        bg=QColor(cfg["bg"]), trunk_w=cfg["trunk_w"],
                        taper=cfg["taper"], width_explicit=wex, glow=glow,
                        max_depth=maxd, target=target, quality="mesh",
                        yaw=30.0 if has3d else 0.0,
                        pitch=18.0 if has3d else 0.0)


# ---- background workers ------------------------------------------------------


class _WorkerSignals(QObject):
    done = Signal(object)
    thumbs = Signal(object)


class ComputeWorker(QRunnable):
    def __init__(self, req: ComputeRequest, gen: int, current_gen):
        super().__init__()
        self.req, self.gen = req, gen
        self._current_gen = current_gen
        self.signals = _WorkerSignals()
        self.setAutoDelete(True)

    def _cancelled(self) -> bool:
        return self.gen != self._current_gen()

    def run(self):
        res = ComputeResult(gen=self.gen, step=self.req.step)
        try:
            res = compute_system(self.req, self._cancelled)
            res.gen = self.gen
        except OperationCancelled:
            return
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            res.error = str(e) or type(e).__name__
        except Exception:
            LOG.exception("computation failed")
            res.error = "unexpected error during computation"
            res.detail = traceback.format_exc()
        self.signals.done.emit(res)


class ThumbsWorker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = _WorkerSignals()
        self.setAutoDelete(True)

    def run(self):
        out = []
        try:
            for i, (_name, cfg) in enumerate(PRESETS):
                out.append((i, render_preset_qimage(cfg, 96)))
        except Exception:
            LOG.exception("thumbnail rendering failed")
        try:
            icon = render_preset_qimage(PRESETS[0][1], 256)
        except Exception:
            icon = QImage()
        self.signals.thumbs.emit((icon, out))


class MeshWorker(QRunnable):
    """Builds a MeshDraw off the GUI thread; token guards staleness."""

    def __init__(self, params: dict, token: int):
        super().__init__()
        self.params, self.token = params, token
        self.signals = _WorkerSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            mesh = build_mesh(**self.params)
        except Exception:
            LOG.exception("mesh build failed")
            mesh = None
        self.signals.done.emit((self.token, mesh))


# ---- widgets ------------------------------------------------------------------


class ColorButton(QPushButton):
    def __init__(self, title: str, color: str, parent=None):
        super().__init__(title, parent)
        self._color = QColor(color)
        self.clicked.connect(self._pick)
        self._sync()

    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color) -> None:
        self._color = QColor(color)
        self._sync()

    def _sync(self) -> None:
        c = self._color
        lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        text = "#10151a" if lum > 140 else "#f2f6fa"
        self.setStyleSheet(
            f"QPushButton {{ background-color: {c.name()}; color: {text};"
            f" border: 1px solid #0b0f14; border-radius: 6px;"
            f" padding: 5px 8px; font-weight: 600; }}")

    def _pick(self) -> None:
        c = QColorDialog.getColor(self._color, self, "Choose colour")
        if c.isValid():
            self._color = c
            self._sync()


class GrowthChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(126)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip("Symbol count |ωₙ| per derivation step (log scale).\n"
                        "Amber bar = the step currently rendered.")
        self._rows = None
        self._lam = None
        self._cur = 0
        self._stoch = False

    def set_data(self, rows, lam, cur, stoch=False):
        self._rows, self._lam, self._cur, self._stoch = rows, lam, cur, stoch
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#0e131a"))
        if not self._rows:
            p.setPen(QColor("#546274"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "growth function unavailable for this system class")
            return
        rows = self._rows
        nmax = max(k for k, _, _ in rows)
        lg = [math.log10(max(c, 1e-12)) for _, c, _ in rows]
        ymax = max(max(lg), 1.0)
        ml, mr, mt, mb = 42, 10, 16, 16
        cw, ch = w - ml - mr, h - mt - mb

        def X(k):
            return ml + (k / max(nmax, 1)) * cw

        def Y(v):
            return h - mb - (v / ymax) * ch

        pf = p.font()
        pf.setPointSize(7)
        p.setFont(pf)
        kstep = max(1, math.ceil(ymax / 4))
        k = 0
        while k <= ymax + 1e-9:
            yv = Y(k)
            p.setPen(QColor(255, 255, 255, 16))
            p.drawLine(QLineF(ml, yv, w - mr, yv))
            p.setPen(QColor("#546274"))
            p.drawText(QRectF(0, yv - 6, ml - 6, 12),
                       Qt.AlignmentFlag.AlignRight |
                       Qt.AlignmentFlag.AlignVCenter, f"1e{k}")
            k += kstep
        bw = min(18.0, max(2.0, cw / max(nmax, 1) * 0.5))
        p.setPen(Qt.PenStyle.NoPen)
        for i, (kk, _c, _r) in enumerate(rows):
            xv, yv = X(kk), Y(lg[i])
            p.setBrush(QBrush(QColor("#e8b04d" if kk == self._cur
                                     else "#4d9fe8")))
            p.drawRect(QRectF(xv - bw / 2, yv, bw, (h - mb) - yv))
        p.setPen(QColor(255, 255, 255, 30))
        p.drawLine(QLineF(ml, h - mb, w - mr, h - mb))
        p.setPen(QColor("#546274"))
        p.drawText(QRectF(ml - 8, h - mb + 2, 20, 12),
                   Qt.AlignmentFlag.AlignLeft, "0")
        if nmax > 0:
            p.drawText(QRectF(X(nmax) - 30, h - mb + 2, 30, 12),
                       Qt.AlignmentFlag.AlignRight, str(nmax))
        if self._lam is not None:
            kind = ("exponential" if self._lam > 1.001 else
                    "decay" if self._lam < 0.999 else "sub-exponential")
            p.setPen(QColor("#4d9fe8"))
            p.drawText(QRectF(ml + 4, 2, w - ml - 8, 12),
                       Qt.AlignmentFlag.AlignLeft,
                       f"λ ≈ {self._lam:.4f} · {kind} · "
                       + ("E|ωₙ|" if self._stoch else "|ωₙ|"))


class LSystemCanvas(QWidget):
    """Interactive viewport with async mesh building, keyboard navigation,
    clipboard image copy and a context menu."""

    exportRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._bg = QColor("#0d151b")
        self._trunk = QColor("#8a4a34")
        self._tip = QColor("#ffb37a")
        self._trunk_w = 2.6
        self._taper = 0.70
        self._glow = True
        self._groups: dict = {}
        self._polys: list = []
        self._elems: tuple = ([], [])
        self._total = 0
        self._bbox3 = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        self._max_depth = 0
        self._width_explicit = False
        self._has3d = False
        self._yaw = 0.0
        self._pitch = 0.0
        self._roll = 0.0
        self._fov = 0
        self._persp_d: float | None = None
        self._light_az = LIGHT_AZ_DEFAULT
        self._light_el = LIGHT_EL_DEFAULT
        self._light_w = light_dir(self._light_az, self._light_el)
        self._grid = False
        self._color_by = "depth"
        self._fog = 0.0
        self._ambient = AMBIENT_DEFAULT / 100.0
        self._fill_alpha = FILL_OPACITY_DEFAULT / 100.0
        self._proj: dict = {}
        self._view_dirty = True
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._cx = self._cy = 0.0
        self._user_navigated = False
        self._pen_cache: dict = {}
        self._reveal = 1.0
        self._anim_enabled = True
        self._speed = 22
        self._anim = QTimer(self, interval=16, timeout=self._tick)
        self._orbit_timer = QTimer(self, interval=16, timeout=self._orbit_tick)
        self._drag_last = None
        self._mode = "pan"
        self._mode_setting = "auto"
        self._mesh = None
        self._mesh_dirty = True
        self._mesh_suspended = False
        self._mesh_rscale = 1.0
        self._mesh_note = ""
        self._mesh_token = 0
        self._mesh_pending = False
        self._install_shortcuts()

    # ---------- data / style ----------

    def set_data(self, groups, polys, elems, total, bbox3, max_depth,
                 width_explicit):
        self._groups = groups
        self._polys = polys
        self._elems = elems
        self._total = total
        self._bbox3 = bbox3
        self._max_depth = max_depth
        self._width_explicit = width_explicit
        self._has3d = (bbox3[5] - bbox3[2]) > 1e-9
        self._persp_d = persp_distance(bbox3, self._fov)
        self._pen_cache.clear()
        self._invalidate_mesh()
        self._mesh_suspended = False
        self._view_dirty = True
        if total or polys or (elems and elems[0]):
            self.fit_view()
        if self._anim_enabled and total:
            self._reveal = 0.0
            self._anim.start()
        else:
            self._reveal = 1.0
            self._anim.stop()
        self.update()

    def set_style(self, trunk_w, taper, trunk, tip, bg, glow):
        self._trunk_w = trunk_w
        self._taper = taper
        self._trunk = QColor(trunk)
        self._tip = QColor(tip)
        self._bg = QColor(bg)
        self._glow = glow
        self._pen_cache.clear()
        self._invalidate_mesh()
        self.update()

    def set_render_mode(self, mode: str) -> None:
        if mode not in ("auto", "mesh", "wire"):
            mode = "auto"
        self._mode_setting = mode
        self._invalidate_mesh()
        self.update()

    def set_render_opts(self, color_by: str, fog: float, ambient: float,
                        fill_alpha: float) -> None:
        self._color_by = color_by if color_by in ("depth", "height") else "depth"
        self._fog = min(max(fog, 0.0), 0.95)
        self._ambient = min(max(ambient, 0.0), 0.9)
        self._fill_alpha = min(max(fill_alpha, 0.0), 1.0)
        self._invalidate_mesh()
        self.update()

    def set_fov(self, fov_deg: int) -> None:
        self._fov = max(0, int(fov_deg))
        self._persp_d = persp_distance(self._bbox3, self._fov)
        self._invalidate_mesh()
        self._view_dirty = True
        self.fit_view()
        self.update()

    def set_light(self, az_deg: float, el_deg: float) -> None:
        self._light_az, self._light_el = az_deg, el_deg
        self._light_w = light_dir(az_deg, el_deg)
        self._invalidate_mesh()
        self.update()

    def set_grid(self, on: bool) -> None:
        self._grid = bool(on)
        self.update()

    def set_roll(self, deg: float) -> None:
        self._roll = deg % 360.0
        self.update()

    def set_camera(self, yaw: float, pitch: float) -> None:
        self._yaw = yaw % 360.0
        self._pitch = max(-89.0, min(89.0, pitch))
        self._view_dirty = True
        self._invalidate_mesh()
        self.fit_view()
        self.update()

    def get_camera(self) -> dict:
        return dict(yaw=self._yaw, pitch=self._pitch, roll=self._roll,
                    zoom=self._zoom)

    def apply_camera(self, cam: dict) -> None:
        self._yaw = float(cam.get("yaw", 0.0)) % 360.0
        self._pitch = max(-89.0, min(89.0, float(cam.get("pitch", 0.0))))
        self._roll = float(cam.get("roll", 0.0)) % 360.0
        self._zoom = min(max(float(cam.get("zoom", 1.0)), 1e-4), 5000.0)
        self._user_navigated = True
        self._view_dirty = True
        self._invalidate_mesh()
        self.update()

    def set_anim(self, enabled: bool, speed: int) -> None:
        self._anim_enabled = enabled
        self._speed = max(1, int(speed))
        if not enabled:
            self._anim.stop()
            if self._reveal < 1.0:
                self._reveal = 1.0
                self.update()

    def replay(self) -> None:
        if self._total:
            self._reveal = 0.0
            self._anim.start()

    def set_auto_orbit(self, on: bool) -> None:
        if on:
            self._orbit_timer.start()
        else:
            self._orbit_timer.stop()
            self._mesh_suspended = False
            self._invalidate_mesh()
            self.update()

    # ---------- mesh management ----------

    def _invalidate_mesh(self) -> None:
        self._mesh_dirty = True
        self._mesh_token += 1          # stale async results are dropped
        self._mesh_pending = False

    def _mesh_capable(self) -> bool:
        nelem = len(self._elems[0])
        return bool(self._total or nelem) and self._mode_setting != "wire" and \
            (self._total + nelem) <= MESH_LIVE_MAX

    def _mesh_active(self) -> bool:
        self._mesh_note = ""
        nelem = len(self._elems[0])
        if (not self._total and not nelem) or self._mode_setting == "wire":
            return False
        if self._mesh_suspended:
            self._mesh_note = "wire · interactive"
            return False
        if (self._total + nelem) > MESH_LIVE_MAX:
            self._mesh_note = (f"wire · {self._total + nelem:,} primitives "
                               f"(mesh cap {MESH_LIVE_MAX:,})")
            return False
        return True

    def _mesh_params(self) -> dict:
        return dict(groups=self._groups, polys=self._polys, elems=self._elems,
                    trunk=self._trunk, tip=self._tip, bg=self._bg,
                    trunk_w=self._trunk_w, taper=self._taper,
                    width_explicit=self._width_explicit,
                    max_depth=self._max_depth, yaw=self._yaw,
                    pitch=self._pitch, rscale=self._mesh_rscale,
                    glow=self._glow, light_w=self._light_w,
                    persp=self._persp_d, color_by=self._color_by,
                    bbox3=self._bbox3, fog=self._fog, ambient=self._ambient,
                    fill_alpha=self._fill_alpha)

    def _dispatch_mesh_async(self) -> None:
        self._mesh_pending = True
        self._mesh_token += 1
        token = self._mesh_token
        worker = MeshWorker(self._mesh_params(), token)
        worker.signals.done.connect(self._mesh_ready)
        QThreadPool.globalInstance().start(worker)

    def _mesh_ready(self, payload) -> None:
        token, mesh = payload
        self._mesh_pending = False
        if token != self._mesh_token:
            return                     # superseded — a rebuild is coming
        self._mesh = mesh
        self._mesh_dirty = False
        self.update()

    # ---------- keyboard navigation ----------

    def _install_shortcuts(self):
        def sc(key, fn):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(Qt.ShortcutContext.WidgetShortcut)
            s.activated.connect(fn)

        sc("Left", lambda: self._key_nav(-4, 0))
        sc("Right", lambda: self._key_nav(4, 0))
        sc("Up", lambda: self._key_nav(0, 4))
        sc("Down", lambda: self._key_nav(0, -4))
        sc("Plus", lambda: self._zoom_step(1.25))
        sc("Minus", lambda: self._zoom_step(0.8))
        sc("F", self._fit_full)
        sc("Space", self.replay)

    def _key_nav(self, dyaw: float, dpitch: float) -> None:
        if not (self._total or self._polys or self._elems[0]):
            return
        if self._has3d:
            self._yaw = (self._yaw + dyaw) % 360.0
            self._pitch = max(-89.0, min(89.0, self._pitch + dpitch))
            self._view_dirty = True
            self._invalidate_mesh()
        else:
            self._pan += QPointF(-dyaw * 8.0, dpitch * 8.0)
        self._user_navigated = True
        self.update()

    def _zoom_step(self, factor: float) -> None:
        if not (self._total or self._polys or self._elems[0]):
            return
        center = QPointF(self.width() / 2, self.height() / 2)
        self._zoom_at(self._unrolled_pos(center), factor)

    def _fit_full(self) -> None:
        self.fit_view()
        self.update()

    # ---------- projection / navigation ----------

    def fit_view(self) -> None:
        self._user_navigated = False
        if not self._total and not self._polys and not self._elems[0]:
            return
        (minx, miny, maxx, maxy), _ = projected_bbox(
            self._bbox3, self._yaw, self._pitch, self._persp_d)
        bw = max(maxx - minx, 1e-9)
        bh = max(maxy - miny, 1e-9)
        self._cx, self._cy = (minx + maxx) / 2, (miny + maxy) / 2
        m = 36
        z = min((self.width() - 2 * m) / bw, (self.height() - 2 * m) / bh)
        self._zoom = min(max(z, 1e-4), 5000.0)
        self._pan = QPointF(0, 0)
        self._roll = 0.0
        self._mesh_rscale = 1.0 / self._zoom
        self._mesh_dirty = True

    def _rebuild_proj(self):
        cy, sy, cp, sp = cam_basis(self._yaw, self._pitch)
        persp = self._persp_d
        out = {}
        for key, (segs, order) in self._groups.items():
            qs = []
            ap = qs.append
            for (x1, y1, z1, x2, y2, z2) in segs:
                a = _project(x1, y1, z1, cy, sy, cp, sp, persp)
                b = _project(x2, y2, z2, cy, sy, cp, sp, persp)
                ap(QLineF(a[0], a[1], b[0], b[1]))
            out[key] = (qs, order)
        self._proj = out
        self._view_dirty = False

    # ---------- painting ----------

    def _pen(self, key, glow: bool) -> QPen:
        ck = (key, glow)
        pen = self._pen_cache.get(ck)
        if pen is None:
            depth, wq = key
            t = depth / self._max_depth if self._max_depth else 0.0
            c = QColor(
                round(self._trunk.red() + (self._tip.red() - self._trunk.red()) * t),
                round(self._trunk.green() + (self._tip.green() - self._trunk.green()) * t),
                round(self._trunk.blue() + (self._tip.blue() - self._trunk.blue()) * t))
            wpx = wq if self._width_explicit else \
                self._trunk_w * (self._taper ** depth)
            if glow:
                c.setAlpha(50)
                wpx = wpx * 3.5 + 2.5
            pen = QPen(c, max(0.7, wpx))
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._pen_cache[ck] = pen
        return pen

    def _slice(self, key, count: int):
        ql, order = self._proj[key]
        if order[0] >= count:
            return None
        if count >= self._total:
            return ql
        cnt = bisect_right(order, count - 1)
        return ql if cnt == len(ql) else ql[:cnt]

    def _draw_all(self, p: QPainter, count: int) -> None:
        keys = list(self._proj.keys())
        if self._glow:
            for key in keys:
                sub = self._slice(key, count)
                if sub:
                    p.setPen(self._pen(key, True))
                    p.drawLines(sub)
        for key in keys:
            sub = self._slice(key, count)
            if sub:
                p.setPen(self._pen(key, False))
                p.drawLines(sub)

    def _draw_wire_polys(self, p: QPainter) -> None:
        cy, sy, cp, sp = cam_basis(self._yaw, self._pitch)
        persp = self._persp_d
        for (pdepth, verts) in self._polys:
            t = pdepth / self._max_depth if self._max_depth else 0.0
            c = QColor(
                round(self._trunk.red() + (self._tip.red() - self._trunk.red()) * t),
                round(self._trunk.green() + (self._tip.green() - self._trunk.green()) * t),
                round(self._trunk.blue() + (self._tip.blue() - self._trunk.blue()) * t),
                round(255 * self._fill_alpha))
            p.setBrush(QBrush(c))
            p.setPen(QPen(QColor(c).darker(150), 0))
            p.drawPolygon(QPolygonF(
                [QPointF(*_project(px, py, pz, cy, sy, cp, sp, persp))
                 for (px, py, pz) in verts]))

    def _draw_grid(self, p: QPainter) -> None:
        minx, miny, minz, maxx, maxy, maxz = self._bbox3
        ex = (maxx - minx) or 1.0
        ez = (maxz - minz) or 1.0
        x0, x1 = minx - ex * 0.15, maxx + ex * 0.15
        z0, z1 = minz - ez * 0.15, maxz + ez * 0.15
        n = 8
        gc = QColor(self._bg).lighter(150)
        gc.setAlpha(120)
        pen = QPen(gc, 0)
        pen.setCosmetic(True)
        p.setPen(pen)
        cy, sy, cp, sp = cam_basis(self._yaw, self._pitch)
        persp = self._persp_d
        lines = []
        for i in range(n + 1):
            f = i / n
            xa = x0 + (x1 - x0) * f
            za = z0 + (z1 - z0) * f
            a1 = _project(xa, miny, z0, cy, sy, cp, sp, persp)
            b1 = _project(xa, miny, z1, cy, sy, cp, sp, persp)
            a2 = _project(x0, miny, za, cy, sy, cp, sp, persp)
            b2 = _project(x1, miny, za, cy, sy, cp, sp, persp)
            lines.append(QLineF(a1[0], a1[1], b1[0], b1[1]))
            lines.append(QLineF(a2[0], a2[1], b2[0], b2[1]))
        p.drawLines(lines)

    def _vignette(self, p: QPainter) -> None:
        g = QRadialGradient(QPointF(self.width() / 2, self.height() / 2),
                            max(self.width(), self.height()) * 0.72)
        g.setColorAt(0.0, QColor(0, 0, 0, 0))
        g.setColorAt(1.0, QColor(0, 0, 0, 80))
        p.fillRect(self.rect(), QBrush(g))

    def paintEvent(self, ev):
        p = QPainter(self)
        paint_backdrop(p, self.rect(), self._bg)
        mesh_mode = self._mesh_active()
        nelem = len(self._elems[0])
        if not self._total and not self._polys and not nelem:
            p.setPen(QColor(150, 162, 178))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Nothing to draw — check the axiom and rules.")
            self._vignette(p)
            return
        if mesh_mode and self._mesh_dirty:
            if (self._total + nelem) <= MESH_TURNTABLE_MAX:
                self._mesh = build_mesh(**self._mesh_params())   # small: sync
                self._mesh_dirty = False
            elif not self._mesh_pending:
                self._dispatch_mesh_async()                      # large: async
        solid = mesh_mode and self._mesh is not None and \
            not self._mesh_dirty and self._reveal >= 1.0
        if not solid and self._view_dirty:
            self._rebuild_proj()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.save()
        p.translate(self.width() / 2 + self._pan.x(),
                    self.height() / 2 + self._pan.y())
        if self._roll:
            p.rotate(self._roll)
        p.scale(self._zoom, -self._zoom)
        p.translate(-self._cx, -self._cy)
        if solid:
            if self._grid and self._has3d:
                self._draw_grid(p)
            if self._has3d and self._pitch > 8.0:
                draw_ground_shadow(p, self._bbox3, self._yaw, self._pitch,
                                   self._persp_d)
            draw_mesh(p, self._mesh)
        else:
            if self._grid and self._has3d:
                self._draw_grid(p)
            count = self._total if self._reveal >= 1 else \
                max(1, int(self._reveal * self._total))
            self._draw_all(p, count)
            if self._polys and self._reveal >= 1.0:
                self._draw_wire_polys(p)
            if nelem:
                cy, sy, cp, sp = cam_basis(self._yaw, self._pitch)
                _wire_elem_paint(p, self._elems,
                                 count if self._reveal < 1.0 else None,
                                 self._trunk, self._tip, self._max_depth,
                                 self._fill_alpha, self._zoom,
                                 cy, sy, cp, sp, self._persp_d)
        p.restore()
        self._vignette(p)
        p.setPen(QColor(255, 255, 255, 55))
        hud = f"zoom ×{self._zoom:.3g}"
        if self._roll:
            hud += f"  roll {self._roll:.0f}°"
        if self._has3d:
            hud += f"   yaw {self._yaw:.0f}°  pitch {self._pitch:.0f}°"
            if self._persp_d:
                hud += f"  fov {self._fov}°"
        p.drawText(10, 18, hud)
        p.setPen(QColor(255, 255, 255, 70))
        if self._mesh_pending and mesh_mode:
            mode_txt = "wire · building mesh…"
        else:
            mode_txt = self._mesh_note or \
                ("mesh · shaded tubes" if solid else "wire · fast")
        p.drawText(10, self.height() - 10, mode_txt)
        hint = ("drag orbit · shift-drag roll · middle-drag pan · "
                "wheel zoom · double-click fit"
                if self._has3d else
                "drag pan · shift-drag roll · wheel zoom · double-click fit")
        p.setPen(QColor(255, 255, 255, 46))
        p.drawText(self.rect().adjusted(0, 0, -12, -8),
                   Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
                   hint)

    # ---------- exports ----------

    def export_image(self, path: str, target: int = 2400,
                     transparent: bool = False, quality: str = "auto") -> bool:
        img = render_image(
            self._groups, self._polys, self._elems, self._total, self._bbox3,
            trunk=self._trunk, tip=self._tip, bg=self._bg,
            trunk_w=self._trunk_w, taper=self._taper,
            width_explicit=self._width_explicit, glow=self._glow,
            max_depth=self._max_depth, target=target, transparent=transparent,
            yaw=self._yaw, pitch=self._pitch, quality=quality,
            persp=self._persp_d, light_w=self._light_w,
            color_by=self._color_by, fog=self._fog, ambient=self._ambient,
            fill_alpha=self._fill_alpha)
        return not img.isNull() and img.save(path)

    def copy_image_to_clipboard(self, scale: int = 2) -> bool:
        if not (self._total or self._polys or self._elems[0]):
            return False
        target = max(64, max(self.width(), self.height()) * scale)
        img = render_image(
            self._groups, self._polys, self._elems, self._total, self._bbox3,
            trunk=self._trunk, tip=self._tip, bg=self._bg,
            trunk_w=self._trunk_w, taper=self._taper,
            width_explicit=self._width_explicit, glow=self._glow,
            max_depth=self._max_depth, target=target, transparent=False,
            yaw=self._yaw, pitch=self._pitch, quality="auto",
            persp=self._persp_d, light_w=self._light_w,
            color_by=self._color_by, fog=self._fog, ambient=self._ambient,
            fill_alpha=self._fill_alpha)
        if img.isNull():
            return False
        QApplication.clipboard().setImage(img)
        return True

    def export_turntable(self, directory: str, frames: int = 24,
                         size: int = 1400) -> int:
        written = 0
        for i in range(frames):
            yaw = (self._yaw + i * 360.0 / frames) % 360.0
            img = render_image(
                self._groups, self._polys, self._elems, self._total,
                self._bbox3,
                trunk=self._trunk, tip=self._tip, bg=self._bg,
                trunk_w=self._trunk_w, taper=self._taper,
                width_explicit=self._width_explicit, glow=self._glow,
                max_depth=self._max_depth, target=size, transparent=False,
                yaw=yaw, pitch=self._pitch, quality="mesh",
                persp=self._persp_d, light_w=self._light_w,
                color_by=self._color_by, fog=self._fog,
                ambient=self._ambient, fill_alpha=self._fill_alpha)
            path = os.path.join(directory, f"frame_{i:03d}.png")
            if img.save(path):
                written += 1
        return written

    def export_svg(self, path: str) -> bool:
        if not self._total and not self._polys and not self._elems[0]:
            return False
        W, H = max(1, self.width()), max(1, self.height())
        zoom = self._zoom
        cy, sy, cp, sp = cam_basis(self._yaw, self._pitch)
        persp = self._persp_d
        cx, cyc = self._cx, self._cy

        def TX(vx):
            return (vx - cx) * zoom + W / 2

        def TY(vy):
            return H / 2 - (vy - cyc) * zoom

        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
               f'height="{H}" viewBox="0 0 {W} {H}">',
               f'<rect width="100%" height="100%" fill="{self._bg.name()}"/>']
        if self._roll:
            out.append(f'<g transform="rotate({self._roll:.2f}, {W / 2:.1f}, '
                       f'{H / 2:.1f})">')

        def path_for(pen, lines):
            d = []
            for ln in lines:
                d.append(f"M{TX(ln.x1()):.2f} {TY(ln.y1()):.2f}"
                         f"L{TX(ln.x2()):.2f} {TY(ln.y2()):.2f}")
            c = pen.color()
            wpx = pen.widthF() * zoom
            op = f' stroke-opacity="{c.alphaF():.3f}"' if c.alpha() < 255 else ""
            return (f'<path d="{"".join(d)}" fill="none" '
                    f'stroke="{c.name()}" stroke-width="{wpx:.3f}"'
                    f'{op} stroke-linecap="round"/>')

        def poly_for(brush, pen, polyf):
            pts = " ".join(f"{TX(polyf[i].x()):.2f},{TY(polyf[i].y()):.2f}"
                           for i in range(polyf.count()))
            c = brush.color()
            return (f'<polygon points="{pts}" fill="{c.name()}" '
                    f'fill-opacity="{c.alphaF():.3f}" '
                    f'stroke="{pen.color().name()}" stroke-width="1"/>')

        mesh = build_mesh(self._groups, self._polys, self._elems,
                          trunk=self._trunk, tip=self._tip, bg=self._bg,
                          trunk_w=self._trunk_w, taper=self._taper,
                          width_explicit=self._width_explicit,
                          max_depth=self._max_depth, yaw=self._yaw,
                          pitch=self._pitch, rscale=1.0 / zoom,
                          glow=self._glow, cap=MESH_EXPORT_CAP, buckets=96,
                          light_w=self._light_w, persp=persp,
                          color_by=self._color_by, bbox3=self._bbox3,
                          fog=self._fog, ambient=self._ambient,
                          fill_alpha=self._fill_alpha)
        if mesh is not None:
            for pen, lines in mesh.glow:
                if lines:
                    out.append(path_for(pen, lines))
            for tubes, stripes, poly_triples, balls, rings in mesh.passes:
                for pen, lines in tubes + stripes:
                    if lines:
                        out.append(path_for(pen, lines))
                for brush, open_pen, polyf in poly_triples:
                    out.append(poly_for(brush, open_pen, polyf))
                for grad, r, centers in balls:
                    c = grad.colorAt(0.55)
                    for (cxp, cyp) in centers:
                        out.append(f'<circle cx="{TX(cxp):.2f}" '
                                   f'cy="{TY(cyp):.2f}" r="{r * zoom:.3f}" '
                                   f'fill="{c.name()}" '
                                   f'fill-opacity="{self._fill_alpha:.3f}"/>')
                for pen, items in rings:
                    c = pen.color()
                    for (cxp, cyp), r in items:
                        out.append(f'<circle cx="{TX(cxp):.2f}" '
                                   f'cy="{TY(cyp):.2f}" r="{r * zoom:.3f}" '
                                   f'fill="none" stroke="{c.name()}" '
                                   f'stroke-width="{pen.widthF() * zoom:.3f}"/>')
        else:
            keys = sorted(self._groups.keys())
            for key in keys:
                segs, _order = self._groups[key]
                lines = []
                for (x1, y1, z1, x2, y2, z2) in segs:
                    a = _project(x1, y1, z1, cy, sy, cp, sp, persp)
                    b = _project(x2, y2, z2, cy, sy, cp, sp, persp)
                    lines.append(QLineF(a[0], a[1], b[0], b[1]))
                out.append(path_for(self._pen(key, False), lines))
            for (pdepth, verts) in self._polys:
                polyf = QPolygonF(
                    [QPointF(*_project(px, py, pz, cy, sy, cp, sp, persp))
                     for (px, py, pz) in verts])
                t = pdepth / self._max_depth if self._max_depth else 0.0
                c = QColor(
                    round(self._trunk.red() + (self._tip.red() - self._trunk.red()) * t),
                    round(self._trunk.green() + (self._tip.green() - self._trunk.green()) * t),
                    round(self._trunk.blue() + (self._tip.blue() - self._trunk.blue()) * t),
                    round(255 * self._fill_alpha))
                out.append(poly_for(QBrush(c), QPen(QColor(c).darker(150), 0),
                                    polyf))
        if self._roll:
            out.append("</g>")
        out.append("</svg>")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(out))
            return True
        except OSError:
            LOG.exception("svg export failed")
            return False

    # ---------- interaction ----------

    def _tick(self):
        self._reveal += self._speed / 1000.0
        if self._reveal >= 1.0:
            self._reveal = 1.0
            self._anim.stop()
        self.update()

    def _orbit_tick(self):
        self._yaw = (self._yaw + 0.3) % 360.0
        self._view_dirty = True
        nelem = len(self._elems[0])
        if self._mesh_capable() and (self._total + nelem) <= MESH_TURNTABLE_MAX:
            self._mesh_suspended = False
            self._mesh_dirty = True          # shaded turntable (sync, small)
        else:
            self._mesh_suspended = True      # fast wire while spinning
        self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta and (self._total or self._polys or self._elems[0]):
            self._zoom_at(self._unrolled_pos(ev.position()),
                          1.25 if delta > 0 else 0.8)

    def _unrolled_pos(self, pos: QPointF) -> QPointF:
        if not self._roll:
            return pos
        cx = self.width() / 2 + self._pan.x()
        cyc = self.height() / 2 + self._pan.y()
        a = math.radians(-self._roll)
        dx, dy = pos.x() - cx, pos.y() - cyc
        ca, sa = math.cos(a), math.sin(a)
        return QPointF(cx + dx * ca - dy * sa, cyc + dx * sa + dy * ca)

    def _zoom_at(self, pos: QPointF, factor: float):
        old = self._zoom
        new = min(max(old * factor, 1e-4), 5000.0)
        w, h = self.width(), self.height()
        wx = (pos.x() - w / 2 - self._pan.x()) / old + self._cx
        wy = (h / 2 + self._pan.y() - pos.y()) / old + self._cy
        self._zoom = new
        self._pan = QPointF(pos.x() - w / 2 - (wx - self._cx) * new,
                            pos.y() - h / 2 + (wy - self._cy) * new)
        self._user_navigated = True
        self.update()

    def mousePressEvent(self, ev):
        has = bool(self._total or self._polys or self._elems[0])
        if ev.button() == Qt.MouseButton.MiddleButton and has:
            self._mode = "pan"
            self._drag_last = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif ev.button() == Qt.MouseButton.LeftButton and has:
            if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._mode = "roll"
            else:
                self._mode = "orbit" if self._has3d else "pan"
            self._drag_last = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        if self._drag_last is None:
            return
        d = ev.position() - self._drag_last
        if self._mode == "orbit":
            self._yaw = (self._yaw + d.x() * 0.4) % 360.0
            self._pitch = max(-89.0, min(89.0, self._pitch + d.y() * 0.4))
            self._view_dirty = True
            nelem = len(self._elems[0])
            if self._mesh_capable() and \
                    (self._total + nelem) <= MESH_TURNTABLE_MAX:
                self._mesh_suspended = False
                self._mesh_dirty = True
            else:
                self._mesh_suspended = True
        elif self._mode == "roll":
            self._roll = (self._roll + d.x() * 0.4) % 360.0
        else:
            self._pan += QPointF(d.x(), d.y())
        self._drag_last = ev.position()
        self._user_navigated = True
        self.update()

    def mouseReleaseEvent(self, ev):
        self._drag_last = None
        if self._mesh_suspended and not self._orbit_timer.isActive():
            self._mesh_suspended = False
            self._invalidate_mesh()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mouseDoubleClickEvent(self, ev):
        self._fit_full()

    def contextMenuEvent(self, ev):
        m = QMenu(self)
        m.addAction("Fit view", self._fit_full)
        m.addAction("Replay growth", self.replay)
        m.addSeparator()
        m.addAction("Copy image", self._copy_clipboard)
        m.addAction("Export PNG…", self.exportRequested)
        m.exec(ev.globalPos())

    def _copy_clipboard(self) -> None:
        ok = self.copy_image_to_clipboard()
        if ok:
            self._toast("image copied to clipboard")
        else:
            self._toast("nothing to copy")

    def _toast(self, text: str) -> None:
        self._toast_text = text
        QTimer.singleShot(2000, self._clear_toast)
        self.update()

    def _clear_toast(self) -> None:
        self._toast_text = ""
        self.update()

    _toast_text = ""

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if (self._total or self._polys or self._elems[0]) and \
                not self._user_navigated:
            self.fit_view()
            self.update()


# ---- dialogs ------------------------------------------------------------------


class TextDialog(QDialog):
    def __init__(self, title, text, html=False, parent=None, save_name=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(780, 560)
        v = QVBoxLayout(self)
        view = QTextEdit()
        view.setReadOnly(True)
        if html:
            view.setHtml(text)
        else:
            view.setFont(QFontDatabase.systemFont(
                QFontDatabase.SystemFont.FixedFont))
            view.setPlainText(text)
        v.addWidget(view, 1)
        row = QHBoxLayout()
        if save_name:
            btn = QPushButton("Save as text…")
            btn.clicked.connect(
                lambda: self._save(view.toPlainText(), save_name))
            row.addWidget(btn)
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(view.toPlainText()))
        row.addWidget(copy_btn)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)

    def _save(self, text, name):
        path, _ = QFileDialog.getSaveFileName(self, "Save text", name,
                                              "Text (*.txt)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            except OSError as e:
                QMessageBox.warning(self, "Save failed", str(e))


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export PNG")
        self.setMinimumWidth(320)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.size = QComboBox()
        for s in (1080, 1600, 2048, 2400, 3200, 4096):
            self.size.addItem(f"{s} px (long edge)", s)
        self.size.setCurrentIndex(3)
        form.addRow("Resolution", self.size)
        self.quality = QComboBox()
        self.quality.addItem("Auto — mesh, falls back if too dense", "auto")
        self.quality.addItem("Mesh — shaded, depth-sorted tubes", "mesh")
        self.quality.addItem("Wire — fast lines", "wire")
        form.addRow("Quality", self.quality)
        self.trans = QCheckBox("Transparent background")
        form.addRow(self.trans)
        v.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)


HELP_HTML = """
<h3>Productions</h3>
<pre>X -&gt; FF-[-F+F+F]            deterministic (D0L)
F -&gt; 0.3 F[+F] ; 0.7 F[-F]  weighted stochastic variants (normalised)
A(t) : t &gt; 2 -&gt; F(1)A(t-1)   parametric module + condition
AB &lt; C &gt; DE -&gt; ...           left / right context (longest match wins)
#define g 9.8                constants, usable in params &amp; conditions
# comment    (errors report the line number)</pre>
<h3>Turtle alphabet</h3>
<pre>F G …   any letter listed in "Draw": forward by x (default 1)
f g      move without drawing
+ -      turn ±δ about Up        &amp; ^   pitch ±δ about Left
/ \\      roll ±δ about Heading   |     turn 180°
!(w)     set pen width to w      !     shrink width ×0.9
[ ]      push/pop turtle state   %     cut — discard rest of branch
{ . }    filled polygon: { opens, . drops a vertex, } closes
@O(r)    closed element (sphere / disc) at the turtle position
@o(r)    open element (ring / circle outline)</pre>
<h3>Camera, light &amp; atmosphere</h3>
<b>FOV 0°</b> = orthographic; higher adds perspective. World-space key
light (az/el) plus an <b>Ambient</b> floor; <b>Fog</b> fades far
geometry into the backdrop. <b>Colour by</b> drives the gradient from
derivation depth or world height. <b>Fill opacity</b> applies to
polygons and elements. <b>Shift-drag</b> rolls the view.
<h3>Keyboard (when the canvas has focus — click it first)</h3>
<pre>arrows   orbit (3D) / pan (2D)      + -  zoom
F        fit                        Space  replay growth</pre>
<h3>Shortcuts</h3>
<pre>Ctrl+E export PNG   Ctrl+Shift+C copy image   Ctrl+R replay
Ctrl+0 fit          Ctrl+D derivation         Ctrl+B panel
Ctrl+S save         Ctrl+O load               F1 this help</pre>
<h3>Reading</h3>
A. Lindenmayer (1968) · Prusinkiewicz &amp; Lindenmayer, <i>The Algorithmic
Beauty of Plants</i> (1990) · Rozenberg &amp; Salomaa (1980).
"""

STYLESHEET = """
QWidget { background-color: #12161d; color: #dce4ec; font-size: 13px; }
QSplitter::handle { width: 3px; background: #1c232d; }
QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #1a212b; border: 1px solid #2a3442; border-radius: 6px;
    padding: 4px 6px; }
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus { border-color: #3d6ea8; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: #1a212b; border: 1px solid #2a3442;
    selection-background-color: #2c4a6e; }
QComboBox QAbstractItemView::item { min-height: 32px; }
QPushButton { background: #1f2833; border: 1px solid #303d4d;
    border-radius: 6px; padding: 6px 12px; }
QPushButton:hover { background: #273241; }
QPushButton:pressed { background: #2c3949; }
QPushButton:disabled { color: #5a6675; }
QGroupBox { border: 1px solid #232d3a; border-radius: 8px; margin-top: 12px;
    padding: 10px 6px 6px 6px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: #7fb0e0; }
QSlider::groove:horizontal { height: 4px; background: #2a3442;
    border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -6px 0;
    border-radius: 7px; background: #4d9fe8; }
QSlider::sub-page:horizontal { background: #33628f; border-radius: 2px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #3a4757;
    border-radius: 4px; background: #1a212b; }
QCheckBox::indicator:checked { background: #4d9fe8; border-color: #4d9fe8; }
QToolBar { background: #161b23; border: none;
    border-bottom: 1px solid #1f2733; padding: 4px 6px; spacing: 4px; }
QToolBar QToolButton { background: transparent; border: 1px solid transparent;
    border-radius: 6px; padding: 5px 9px; }
QToolBar QToolButton:hover { background: #222b37; border-color: #303d4d; }
QToolBar QToolButton:checked { background: #2c4a6e; }
QMenuBar { background: #12161d; }
QMenuBar::item { padding: 5px 10px; background: transparent; border-radius: 4px; }
QMenuBar::item:selected { background: #222b37; }
QMenu { background: #1a212b; border: 1px solid #2a3442; padding: 4px; }
QMenu::item { padding: 5px 24px 5px 14px; border-radius: 4px; }
QMenu::item:selected { background: #2c4a6e; }
QMenu::item:disabled { color: #5a6675; }
QMenu::separator { height: 1px; background: #2a3442; margin: 4px 6px; }
QTabWidget::pane { border: 1px solid #232d3a; border-radius: 6px; top: -1px;
    background: #141922; }
QTabBar::tab { background: transparent; color: #8b98a8; padding: 6px 14px;
    border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #e8f0f8; border-bottom: 2px solid #4d9fe8; }
QTabBar::tab:hover:!selected { color: #b9c4d0; }
QStatusBar { background: #12161d; border-top: 1px solid #1f2733; }
QStatusBar::item { border: none; }
QStatusBar QLabel { padding: 0 6px; }
QLabel#brand { font-size: 16px; font-weight: 700; letter-spacing: 3px;
    color: #e8f0f8; }
QLabel#subtitle { color: #6b7889; font-size: 11px; }
QLabel#hint { color: #5c6a7a; font-size: 11px; }
QLabel#stats { color: #8b98a8; font-size: 12px; }
QLabel#class_lbl { color: #86d8b0; font-weight: 600; font-size: 12px; }
"""


def _dspin(lo, hi, dec, step, val, w=76):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    s.setFixedWidth(w)
    return s


# ---- main window --------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self, load_path: str | None = None,
                 preset_name: str | None = None):
        super().__init__()
        self._settings = QSettings(ORG_NAME, APP_NAME)
        self._gen = 0
        self._dirty = False
        self._path: str | None = None
        self._have_result = False
        self._growth_rows = None
        self._pending_cam: dict | None = None
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self.canvas = LSystemCanvas()
        self.setAcceptDrops(True)
        self._build_controls()
        self._build_statusbar()
        self._build_toolbar()
        self._build_menus()
        self._build_sidebar()
        self._wire()
        self._restore_window()
        self._rebuild_recents()
        self._recompute_actions()

        loaded = False
        if load_path:
            loaded = self._load_path(load_path, quiet=True)
        elif preset_name:
            for i, (name, _c) in enumerate(PRESETS):
                if name.lower() == preset_name.lower():
                    self._load_preset(i)
                    loaded = True
                    break
            if not loaded:
                LOG.warning("preset not found: %s", preset_name)
        if not loaded:
            loaded = self._restore_session()
        if not loaded:
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(0)
            self.preset.blockSignals(False)
            self._load_preset(0)
        self._thumb_worker = ThumbsWorker()
        self._thumb_worker.signals.thumbs.connect(self._apply_thumbs)
        QThreadPool.globalInstance().start(self._thumb_worker)

    # ---------- widget creation ----------

    def _build_controls(self):
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.axiom = QLineEdit("F")
        self.rules = QPlainTextEdit()
        self.rules.setFont(fixed)
        self.rules.setFixedHeight(118)
        self.rules.setPlaceholderText(
            "one rule per line:   X -> F[+X][-X]FX\n"
            "parametric:          A(t) : t > 2 -> F(1)A(t-1)\n"
            "context:             b < a -> b      weighted: 0.3 A ; 0.7 B\n"
            "polygon:             L -> [ { . f(0.5) . + (90) f(0.5) . } ]\n"
            "element:             T -> @O(0.2)      ring: @o(0.15)\n"
            "#define k 0.5       # comment")
        self.draw = QLineEdit("FG")
        self.draw.setToolTip("Letters that draw a line; others are placeholders.")
        self.iters = QSpinBox()
        self.iters.setRange(0, 15)
        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setRange(0, 15)
        self.step_slider.setValue(5)
        self.step_lbl = QLabel("n = 5")
        self.angle = _dspin(0.0, 180.0, 2, 0.5, 22.5, 120)
        self.angle.setSuffix(" °")
        self.jitter = QSlider(Qt.Orientation.Horizontal)
        self.jitter.setRange(0, 100)
        self.jitter_val = QLabel("0 %")
        self.seed = QSpinBox()
        self.seed.setRange(0, 999_999)
        self.seed.setValue(1)
        self.dice = QPushButton("🎲 Reroll")
        self.trop_x = _dspin(-2.0, 2.0, 2, 0.05, 0.0, 64)
        self.trop_y = _dspin(-2.0, 2.0, 2, 0.05, 0.0, 64)
        self.trop_z = _dspin(-2.0, 2.0, 2, 0.05, 0.0, 64)
        self.elast = QSlider(Qt.Orientation.Horizontal)
        self.elast.setRange(0, 100)
        self.elast_lbl = QLabel("e = 0.00")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Auto — mesh, falls back past the cap", "auto")
        self.mode_combo.addItem("Mesh — shaded, depth-sorted tubes", "mesh")
        self.mode_combo.addItem("Wire — fast lines", "wire")
        self.color_combo = QComboBox()
        self.color_combo.addItem("Gradient by derivation depth", "depth")
        self.color_combo.addItem("Gradient by height (Y)", "height")
        self.width_spin = _dspin(0.5, 24.0, 1, 0.1, 2.6, 120)
        self.width_spin.setSuffix(" px")
        self.taper_spin = _dspin(0.40, 1.00, 2, 0.01, 0.70, 120)
        self.fillop = QSlider(Qt.Orientation.Horizontal)
        self.fillop.setRange(0, 100)
        self.fillop.setValue(FILL_OPACITY_DEFAULT)
        self.fillop_lbl = QLabel(f"{FILL_OPACITY_DEFAULT} %")
        self.trunk_btn = ColorButton("Trunk", "#8a4a34")
        self.tip_btn = ColorButton("Tips", "#ffb37a")
        self.bg_btn = ColorButton("Backdrop", "#0d151b")
        self.glow = QCheckBox("Neon glow")
        self.glow.setChecked(True)
        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(0, 85)
        self.fov_spin.setValue(0)
        self.fov_spin.setSuffix(" °")
        self.fov_spin.setFixedWidth(120)
        self.light_az = QSlider(Qt.Orientation.Horizontal)
        self.light_az.setRange(0, 360)
        self.light_az.setValue(LIGHT_AZ_DEFAULT)
        self.light_az_lbl = QLabel(f"az {LIGHT_AZ_DEFAULT}°")
        self.light_el = QSlider(Qt.Orientation.Horizontal)
        self.light_el.setRange(-90, 90)
        self.light_el.setValue(LIGHT_EL_DEFAULT)
        self.light_el_lbl = QLabel(f"el {LIGHT_EL_DEFAULT}°")
        self.amb = QSlider(Qt.Orientation.Horizontal)
        self.amb.setRange(0, 60)
        self.amb.setValue(AMBIENT_DEFAULT)
        self.amb_lbl = QLabel(f"{AMBIENT_DEFAULT} %")
        self.fog = QSlider(Qt.Orientation.Horizontal)
        self.fog.setRange(0, 80)
        self.fog.setValue(0)
        self.fog_lbl = QLabel("0 %")
        self.grid_chk = QCheckBox("Ground grid (3D)")
        self.anim_check = QCheckBox("Animate on change")
        self.anim_check.setChecked(True)
        self.anim_speed = QSlider(Qt.Orientation.Horizontal)
        self.anim_speed.setRange(1, 50)
        self.anim_speed.setValue(22)
        self.speed_val = QLabel("22")
        self.replay_btn = QPushButton("↻ Replay growth")
        self.orbit_chk = QCheckBox("Auto-orbit (3D scenes)")
        self.chart = GrowthChart()
        self.growth_txt = QPlainTextEdit()
        self.growth_txt.setFont(fixed)
        self.growth_txt.setReadOnly(True)
        self.growth_txt.setFixedHeight(118)
        self.deriv_btn = QPushButton("Derivation…")
        self.csv_btn = QPushButton("Growth CSV…")
        self.preset = QComboBox()
        for name, _cfg in PRESETS:
            self.preset.addItem(name)
        self.preset.addItem("Custom")
        self.preset.setIconSize(QSize(30, 30))

    # ---------- chrome ----------

    def _mk_action(self, text, slot=None, shortcut=None, checkable=False,
                   checked=False, tip=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if checkable:
            a.setCheckable(True)
            a.setChecked(checked)
        if slot:
            a.triggered.connect(slot)
        if tip:
            a.setStatusTip(tip)
        return a

    def _build_toolbar(self):
        tb = QToolBar("main")
        tb.setMovable(False)
        tb.setObjectName("MainToolbar")
        self.addToolBar(tb)
        self.act_side = self._mk_action("☰ Panel", None, "Ctrl+B",
                                        checkable=True, checked=True,
                                        tip="Toggle the control panel")
        tb.addAction(self.act_side)
        tb.addSeparator()
        tb.addWidget(QLabel(" Preset "))
        tb.addWidget(self.preset)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        tb.addAction(self._mk_action("↻ Replay", self.canvas.replay, "Ctrl+R",
                                     tip="Replay the growth animation"))
        tb.addAction(self._mk_action("⤢ Fit", self._fit, "Ctrl+0",
                                     tip="Fit the system to the view"))
        tb.addAction(self._mk_action("⬇ PNG…", self._export, "Ctrl+E",
                                     tip="Export a high-resolution PNG"))
        tb.addAction(self._mk_action("？ Syntax", self._show_help, "F1",
                                     tip="Grammar syntax & references"))

    def _build_menus(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        m_file.addAction(self._mk_action("&New", self._new_system, "Ctrl+N",
                                         tip="Start from an empty system"))
        m_file.addAction(self._mk_action("&Open…", self._open_dialog, "Ctrl+O",
                                         tip="Load a system from JSON"))
        self.menu_recent = QMenu("Open &Recent", self)
        m_file.addMenu(self.menu_recent)
        m_file.addSeparator()
        m_file.addAction(self._mk_action("&Save", self._save_json, "Ctrl+S",
                                         tip="Save the system as JSON"))
        m_file.addAction(self._mk_action("Save &As…", self._save_json_as,
                                         "Ctrl+Shift+S"))
        m_file.addSeparator()
        self.act_export = self._mk_action("&Export PNG…", self._export,
                                          "Ctrl+E")
        m_file.addAction(self.act_export)
        self.act_copy = self._mk_action("&Copy image", self._copy_image,
                                        "Ctrl+Shift+C",
                                        tip="Copy a render of the current "
                                            "view to the clipboard")
        m_file.addAction(self.act_copy)
        self.act_svg = self._mk_action("Export &SVG…", self._export_svg,
                                       tip="Vector export of the current view")
        m_file.addAction(self.act_svg)
        self.act_turn = self._mk_action("Export &turntable frames…",
                                        self._export_turntable,
                                        tip="Render a 360° PNG sequence")
        m_file.addAction(self.act_turn)
        m_file.addSeparator()
        m_file.addAction(self._mk_action("E&xit", self.close, "Ctrl+Q"))

        m_view = mb.addMenu("&View")
        m_view.addAction(self._mk_action("Fit view", self._fit, "Ctrl+0"))
        m_view.addAction(self._mk_action("Replay growth", self.canvas.replay,
                                         "Ctrl+R"))
        m_view.addAction(self._mk_action("Reset rotation",
                                         lambda: self.canvas.set_roll(0.0),
                                         tip="Reset the 2D view roll"))
        m_view.addSeparator()
        m_cam = m_view.addMenu("Camera")
        for label, yaw, pitch in (("Front", 0, 0), ("Side", 90, 0),
                                  ("Top", 0, 89), ("¾ view", 30, 18)):
            m_cam.addAction(self._mk_action(
                label, lambda y=yaw, p=pitch: self.canvas.set_camera(y, p)))
        self.act_grid = self._mk_action("Ground grid", None, None,
                                        checkable=True, checked=False)
        m_view.addAction(self.act_grid)
        m_view.addSeparator()
        self.act_orbit = self._mk_action("Auto-orbit", None, None,
                                         checkable=True)
        m_view.addAction(self.act_orbit)
        m_view.addSeparator()
        m_view.addAction(self.act_side)

        m_ana = mb.addMenu("&Analyze")
        self.act_deriv = self._mk_action("Derivation sequence…",
                                         self._show_derivation, "Ctrl+D")
        self.act_csv = self._mk_action("Growth table CSV…", self._export_csv)
        m_ana.addAction(self.act_deriv)
        m_ana.addAction(self.act_csv)

        m_help = mb.addMenu("&Help")
        m_help.addAction(self._mk_action("Syntax & references",
                                         self._show_help, "F1"))
        m_help.addAction(self._mk_action("About Fauna", self._show_about))

    def _build_statusbar(self):
        sb = self.statusBar()
        self.stats = QLabel("ready")
        self.stats.setObjectName("stats")
        sb.addWidget(self.stats, 1)
        self.class_lbl = QLabel("")
        self.class_lbl.setObjectName("class_lbl")
        self.class_lbl.setToolTip(
            "L-system taxonomy: D0L = deterministic, context-free "
            "(Lindenmayer 1968); 1L/2L = context-sensitive; parametric & "
            "conditional per Prusinkiewicz & Lindenmayer (1990).")
        sb.addPermanentWidget(self.class_lbl)

    def _build_sidebar(self):
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(10, 8, 8, 8)
        v.setSpacing(8)
        brand = QLabel("FAUNA")
        brand.setObjectName("brand")
        sub = QLabel(f"L-system laboratory · v{__version__}")
        sub.setObjectName("subtitle")
        v.addWidget(brand)
        v.addWidget(sub)

        tabs = QTabWidget()

        g = QWidget()
        gf = QFormLayout(g)
        gf.setVerticalSpacing(6)
        gf.addRow("Axiom", self.axiom)
        gf.addRow("Productions", self.rules)
        gf.addRow("Draw", self.draw)
        hint = QLabel("F1 · grammar syntax, turtle alphabet & references\n"
                      "click the canvas, then use arrows / + − / F / Space")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        gf.addRow(hint)

        t = QWidget()
        tf = QFormLayout(t)
        tf.setVerticalSpacing(6)
        tf.addRow("Depth", self.iters)
        srow = QHBoxLayout()
        srow.addWidget(self.step_slider, 1)
        srow.addWidget(self.step_lbl)
        sw = QWidget(); sw.setLayout(srow)
        self.step_slider.setToolTip(
            "Render the intermediate word ωₙ of the derivation.")
        tf.addRow("Show step", sw)
        tf.addRow("Angle δ", self.angle)
        jrow = QHBoxLayout()
        jrow.addWidget(self.jitter, 1)
        jrow.addWidget(self.jitter_val)
        jw = QWidget(); jw.setLayout(jrow)
        tf.addRow("Jitter", jw)
        drow = QHBoxLayout()
        drow.addWidget(self.seed, 1)
        drow.addWidget(self.dice)
        dw = QWidget(); dw.setLayout(drow)
        tf.addRow("Seed", dw)
        env = QGroupBox("Environment · tropism")
        ef = QFormLayout(env)
        ef.setVerticalSpacing(6)
        trow = QHBoxLayout()
        for s in (self.trop_x, self.trop_y, self.trop_z):
            trow.addWidget(s)
        trow.addStretch(1)
        tw = QWidget(); tw.setLayout(trow)
        tw.setToolTip("Tropism vector g (e.g. gravity = (0, −1, 0))")
        ef.addRow("g = (x y z)", tw)
        erow = QHBoxLayout()
        erow.addWidget(self.elast, 1)
        erow.addWidget(self.elast_lbl)
        ew = QWidget(); ew.setLayout(erow)
        ef.addRow("Elasticity", ew)
        tf.addRow(env)

        s = QWidget()
        sf = QFormLayout(s)
        sf.setVerticalSpacing(6)
        sf.addRow("Render mode", self.mode_combo)
        sf.addRow("Colour by", self.color_combo)
        sf.addRow("Trunk width", self.width_spin)
        sf.addRow("Taper", self.taper_spin)
        frow = QHBoxLayout()
        frow.addWidget(self.fillop, 1)
        frow.addWidget(self.fillop_lbl)
        fw = QWidget(); fw.setLayout(frow)
        sf.addRow("Fill opacity", fw)
        crow = QHBoxLayout()
        crow.addWidget(self.trunk_btn, 1)
        crow.addWidget(self.tip_btn, 1)
        crow.addWidget(self.bg_btn, 1)
        cw = QWidget(); cw.setLayout(crow)
        sf.addRow("Colours", cw)
        sf.addRow("", self.glow)
        cam = QGroupBox("Camera · light · atmosphere")
        cf = QFormLayout(cam)
        cf.setVerticalSpacing(6)
        cf.addRow("FOV", self.fov_spin)
        arow2 = QHBoxLayout()
        arow2.addWidget(self.light_az, 1)
        arow2.addWidget(self.light_az_lbl)
        aw2 = QWidget(); aw2.setLayout(arow2)
        cf.addRow("Light az", aw2)
        erow2 = QHBoxLayout()
        erow2.addWidget(self.light_el, 1)
        erow2.addWidget(self.light_el_lbl)
        ew2 = QWidget(); ew2.setLayout(erow2)
        cf.addRow("Light el", ew2)
        mrow = QHBoxLayout()
        mrow.addWidget(self.amb, 1)
        mrow.addWidget(self.amb_lbl)
        mw = QWidget(); mw.setLayout(mrow)
        cf.addRow("Ambient", mw)
        grow2 = QHBoxLayout()
        grow2.addWidget(self.fog, 1)
        grow2.addWidget(self.fog_lbl)
        gw = QWidget(); gw.setLayout(grow2)
        cf.addRow("Fog", gw)
        cf.addRow(self.grid_chk)
        sf.addRow(cam)
        grow = QGroupBox("Growth animation")
        gf2 = QFormLayout(grow)
        gf2.setVerticalSpacing(6)
        gf2.addRow(self.anim_check)
        arow = QHBoxLayout()
        arow.addWidget(self.anim_speed, 1)
        arow.addWidget(self.speed_val)
        aw = QWidget(); aw.setLayout(arow)
        gf2.addRow("Speed", aw)
        gf2.addRow(self.replay_btn)
        gf2.addRow(self.orbit_chk)
        sf.addRow(grow)

        a = QWidget()
        av = QVBoxLayout(a)
        av.setContentsMargins(8, 8, 8, 8)
        av.setSpacing(8)
        av.addWidget(self.chart)
        av.addWidget(self.growth_txt)
        brow = QHBoxLayout()
        brow.addWidget(self.deriv_btn)
        brow.addWidget(self.csv_btn, 1)
        av.addLayout(brow)
        av.addStretch(1)

        tabs.addTab(g, "Grammar")
        tabs.addTab(t, "Turtle")
        tabs.addTab(s, "Style")
        tabs.addTab(a, "Analysis")
        v.addWidget(tabs, 1)
        self.side_host = host
        self.act_side.toggled.connect(self.side_host.setVisible)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("MainSplitter")
        host.setMinimumWidth(350)
        host.setMaximumWidth(520)
        splitter.addWidget(host)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 920])
        self.setCentralWidget(splitter)
        self._splitter = splitter

    # ---------- wiring ----------

    def _wire(self):
        self.preset.currentIndexChanged.connect(self._load_preset)
        for w in (self.axiom, self.draw):
            w.textChanged.connect(self._user_changed)
        self.rules.textChanged.connect(self._user_changed)
        self.iters.valueChanged.connect(self._iters_changed)
        self.angle.valueChanged.connect(self._user_changed)
        self.jitter.valueChanged.connect(self._user_changed)
        self.seed.valueChanged.connect(self._user_changed)
        self.width_spin.valueChanged.connect(self._user_changed)
        self.taper_spin.valueChanged.connect(self._user_changed)
        self.glow.toggled.connect(self._user_changed)
        for b in (self.trunk_btn, self.tip_btn, self.bg_btn):
            b.clicked.connect(self._user_changed)
        for s in (self.trop_x, self.trop_y, self.trop_z):
            s.valueChanged.connect(self._user_changed)
        self.elast.valueChanged.connect(self._user_changed)
        self.step_slider.valueChanged.connect(self._user_changed)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.color_combo.currentIndexChanged.connect(self._opts_changed)

        self.fov_spin.valueChanged.connect(
            lambda v: (self.canvas.set_fov(v), self._mark_dirty_only()))
        self.light_az.valueChanged.connect(self._light_changed)
        self.light_el.valueChanged.connect(self._light_changed)
        self.grid_chk.toggled.connect(self._grid_changed)
        self.fillop.valueChanged.connect(self._opts_changed)
        self.amb.valueChanged.connect(self._opts_changed)
        self.fog.valueChanged.connect(self._opts_changed)

        self.jitter.valueChanged.connect(
            lambda v: self.jitter_val.setText(f"{v} %"))
        self.light_az.valueChanged.connect(
            lambda v: self.light_az_lbl.setText(f"az {v}°"))
        self.light_el.valueChanged.connect(
            lambda v: self.light_el_lbl.setText(f"el {v}°"))
        self.fillop.valueChanged.connect(
            lambda v: self.fillop_lbl.setText(f"{v} %"))
        self.amb.valueChanged.connect(
            lambda v: self.amb_lbl.setText(f"{v} %"))
        self.fog.valueChanged.connect(
            lambda v: self.fog_lbl.setText(f"{v} %"))
        self.anim_speed.valueChanged.connect(
            lambda v: self.speed_val.setText(str(v)))
        self.elast.valueChanged.connect(
            lambda v: self.elast_lbl.setText(f"e = {v / 100:.2f}"))
        self.step_slider.valueChanged.connect(
            lambda v: self.step_lbl.setText(f"n = {v}"))

        self.anim_check.toggled.connect(
            lambda on: self.canvas.set_anim(on, self.anim_speed.value()))
        self.anim_speed.valueChanged.connect(
            lambda v: self.canvas.set_anim(self.anim_check.isChecked(), v))
        self.replay_btn.clicked.connect(self.canvas.replay)
        self.orbit_chk.toggled.connect(self.canvas.set_auto_orbit)
        self.act_orbit.toggled.connect(self.orbit_chk.setChecked)
        self.orbit_chk.toggled.connect(self.act_orbit.setChecked)
        self.act_grid.toggled.connect(self.grid_chk.setChecked)
        self.grid_chk.toggled.connect(self.act_grid.setChecked)

        self.deriv_btn.clicked.connect(self._show_derivation)
        self.csv_btn.clicked.connect(self._export_csv)
        self.dice.clicked.connect(
            lambda: self.seed.setValue(random.randint(0, 999_999)))
        self.canvas.exportRequested.connect(self._export)

        self._highlighter = RuleHighlighter(self.rules.document())

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._request_compute)

        sc = QShortcut(QKeySequence("Ctrl+D"), self)
        sc.activated.connect(self._show_derivation)

    # ---------- thumbnails ----------

    def _apply_thumbs(self, payload):
        icon, thumbs = payload
        if icon and not icon.isNull():
            self.setWindowIcon(QIcon(QPixmap.fromImage(icon)))
        for idx, img in thumbs:
            if not img.isNull():
                img.setDevicePixelRatio(3.0)
                self.preset.setItemIcon(idx, QIcon(QPixmap.fromImage(img)))

    # ---------- persistence ----------

    def _restore_window(self):
        s = self._settings
        geo = s.value("win/geometry")
        if geo:
            self.restoreGeometry(geo)
        st = s.value("win/state")
        if st:
            self.restoreState(st)
        sp = s.value("win/splitter")
        if sp:
            self._splitter.restoreState(sp)
        if not geo and not st:
            self.resize(1320, 860)

    def _persist_window(self):
        s = self._settings
        s.setValue("win/geometry", self.saveGeometry())
        s.setValue("win/state", self.saveState())
        s.setValue("win/splitter", self._splitter.saveState())

    def _save_session(self) -> None:
        """Snapshot the full current system (incl. camera & dirty flag)."""
        try:
            snap = json.dumps(dict(state=self._collect_state(),
                                   path=self._path, dirty=self._dirty))
            self._settings.setValue("session/state", snap)
        except Exception:
            LOG.exception("session snapshot failed")

    def _restore_session(self) -> bool:
        snap = self._settings.value("session/state", "")
        if not snap:
            return False
        try:
            obj = json.loads(snap)
            st = obj.get("state")
            if not isinstance(st, dict) or "axiom" not in st:
                return False
            self._apply_state(st, custom=True)
            self._path = obj.get("path") or None
            self._dirty = bool(obj.get("dirty", False))
            self._update_title()
            if self._path:
                self._push_recent(self._path)
            LOG.info("session restored%s",
                     f" ({self._path})" if self._path else "")
            return True
        except Exception:
            LOG.exception("session restore failed")
            return False

    def _recents(self) -> list[str]:
        val = self._settings.value("recents", [])
        return list(val) if isinstance(val, (list, tuple)) else \
            ([val] if val else [])

    def _push_recent(self, path: str):
        items = [p for p in self._recents() if p != path]
        items.insert(0, path)
        self._settings.setValue("recents", items[:8])
        self._rebuild_recents()

    def _rebuild_recents(self):
        self.menu_recent.clear()
        recents = [p for p in self._recents() if os.path.isfile(p)]
        if not recents:
            act = self.menu_recent.addAction("(no recent systems)")
            act.setEnabled(False)
            return
        for p in recents[:8]:
            self.menu_recent.addAction(
                os.path.basename(p), lambda p=p: self._load_path(p))

    # ---------- compute orchestration ----------

    def _current_gen(self) -> int:
        return self._gen

    def _request_compute(self):
        self._gen += 1
        req = ComputeRequest(
            axiom=self.axiom.text().strip() or "F",
            rules_text=self.rules.toPlainText(),
            step=self.step_slider.value(),
            angle=self.angle.value(),
            jitter=self.jitter.value() / 100.0,
            seed=self.seed.value(),
            draw=self.draw.text().strip() or "FG",
            trop=(self.trop_x.value(), self.trop_y.value(), self.trop_z.value()),
            elast=self.elast.value() / 100.0)
        self._set_status("computing…")
        worker = ComputeWorker(req, self._gen, self._current_gen)
        worker.signals.done.connect(self._on_result)
        self._pool.start(worker)

    def _on_result(self, res: ComputeResult):
        if res.gen != self._gen:
            return
        if not res.ok:
            self.class_lbl.setText("")
            self._recompute_actions(False)
            self._set_status(f"⚠ {res.error}", error=True)
            if res.detail:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Icon.Critical)
                box.setWindowTitle("Computation error")
                box.setText("An unexpected error occurred. "
                            "The details have been logged.")
                box.setDetailedText(res.detail)
                box.exec()
            return
        self.canvas.set_data(res.groups, res.polys, res.elems, res.total,
                             res.bbox3, res.max_depth, res.width_explicit)
        self.canvas.set_style(self.width_spin.value(), self.taper_spin.value(),
                              self.trunk_btn.color(), self.tip_btn.color(),
                              self.bg_btn.color(), self.glow.isChecked())
        if self._pending_cam:
            self.canvas.apply_camera(self._pending_cam)
            self._pending_cam = None
        nelem = len(res.elems[0])
        tag = "3D" if res.is3d else "2D"
        stats = (f"ω_{res.step} · {res.symbol_count:,} symbols · "
                 f"{res.total:,} strokes · depth {res.max_depth} · {tag}")
        if res.polys:
            stats += f" · {len(res.polys):,} polygons"
        if nelem:
            stats += f" · {nelem:,} elements"
        if res.truncated:
            stats += " · (symbol cap reached)"
        self._set_status(stats)
        self.class_lbl.setText(
            f"{res.class_text}   —   |V| = {res.alphabet}")
        self._have_result = True
        self._growth_rows = res.growth[0] if res.growth else None
        self._last_ctx = ({"axiom": res.ctx[0], "rules": res.ctx[1],
                           "defines": res.ctx[2], "simple": res.ctx[3]}
                          if res.ctx else None)
        self._update_growth(res)
        self._recompute_actions()
        self._save_session()

    def _recompute_actions(self, have: bool | None = None):
        have = self._have_result if have is None else have
        self.act_export.setEnabled(have)
        self.act_copy.setEnabled(have)
        self.act_svg.setEnabled(have)
        self.act_turn.setEnabled(have)
        self.act_deriv.setEnabled(have)
        self.act_csv.setEnabled(self._growth_rows is not None)
        self.deriv_btn.setEnabled(have)
        self.csv_btn.setEnabled(self._growth_rows is not None)

    def _update_growth(self, res: ComputeResult):
        if not res.growth:
            self.chart.set_data(None, None, res.step)
            self.growth_txt.setPlainText(
                "Growth analysis n/a — the matrix method (|ωₙ₊₁| = M·|ωₙ|)\n"
                "applies to context-free, condition-free systems (D0L family).")
            return
        rows, lam, freqs, stoch = res.growth
        self.chart.set_data(rows, lam, res.step, stoch)
        lines = [f"Growth function ({'expected values, ' if stoch else ''}D0L)",
                 " n     |ωₙ|        |ωₙ|/|ωₙ₋₁|"]
        for k, t, r in rows:
            rs = "—" if r is None else f"{r:.4f}"
            lines.append(f" {k:<5} {t:<10.6g} {rs}")
        kind = ("exponential" if lam > 1.001 else
                "decay/extinction" if lam < 0.999 else "sub-exponential")
        lines.append(f" Perron–Frobenius λ ≈ {lam:.6f} → {kind} growth")
        if freqs:
            lines.append(" limit composition: " +
                         " · ".join(f"{s} {f * 100:.1f}%" for s, f in freqs[:6]))
        if res.word_freq:
            tot = sum(c for _, c in res.word_freq) or 1
            lines.append(" word ω composition: " +
                         " · ".join(f"{s} {c / tot * 100:.1f}%"
                                    for s, c in res.word_freq[:8]))
        self.growth_txt.setPlainText("\n".join(lines))

    # ---------- dirty / title ----------

    def _user_changed(self, *_):
        self._dirty = True
        if self.preset.currentIndex() != len(PRESETS):
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(len(PRESETS))
            self.preset.blockSignals(False)
        self._update_title()
        self._debounce.start()

    def _mark_dirty_only(self):
        self._dirty = True
        self._update_title()

    def _mode_changed(self, _idx):
        self.canvas.set_render_mode(self.mode_combo.currentData() or "auto")
        self._mark_dirty_only()

    def _light_changed(self, *_):
        self.canvas.set_light(self.light_az.value(), self.light_el.value())
        self._mark_dirty_only()

    def _grid_changed(self, on: bool):
        self.canvas.set_grid(on)
        self._mark_dirty_only()

    def _opts_changed(self, *_):
        self.canvas.set_render_opts(
            self.color_combo.currentData() or "depth",
            self.fog.value() / 100.0,
            self.amb.value() / 100.0,
            self.fillop.value() / 100.0)
        self._mark_dirty_only()

    def _update_title(self):
        base = f"{APP_NAME} — L-System Laboratory"
        if self._path:
            base += f" — {os.path.basename(self._path)}"
        if self._dirty:
            base += " •"
        self.setWindowTitle(base)

    def _mark_clean(self, path: str | None):
        self._dirty = False
        self._path = path
        self._update_title()
        if path:
            self._push_recent(path)

    def _iters_changed(self, v):
        self.step_slider.blockSignals(True)
        self.step_slider.setRange(0, max(v, 0))
        self.step_slider.setValue(v)
        self.step_slider.blockSignals(False)
        self.step_lbl.setText(f"n = {v}")
        self._user_changed()

    # ---------- presets / files ----------

    def _load_preset(self, idx: int):
        if idx < 0 or idx >= len(PRESETS):
            return
        _name, cfg = PRESETS[idx]
        self._apply_state(self._preset_state(cfg), custom=False)
        self._mark_clean(None)

    def _preset_state(self, cfg: dict) -> dict:
        trop = cfg.get("trop", (0.0, 0.0, 0.0))
        return dict(axiom=cfg["axiom"], rules=cfg["rules"], draw=cfg["draw"],
                    iters=cfg["iters"], step=cfg["iters"], angle=cfg["angle"],
                    jitter=round(cfg["jitter"] * 100),
                    seed=cfg.get("seed", 1), trunk_w=cfg["trunk_w"],
                    taper=cfg["taper"], glow=True, trunk=cfg["trunk"],
                    tip=cfg["tip"], bg=cfg["bg"], tropism=list(trop),
                    elast=round(cfg.get("elast", 0.0) * 100),
                    render_mode="auto", fov=0,
                    light_az=LIGHT_AZ_DEFAULT, light_el=LIGHT_EL_DEFAULT,
                    grid=False, color_by="depth",
                    fill_opacity=FILL_OPACITY_DEFAULT, fog=0,
                    ambient=AMBIENT_DEFAULT)

    def _apply_state(self, st: dict, custom: bool = True):
        ws = (self.axiom, self.rules, self.draw, self.iters, self.angle,
              self.jitter, self.seed, self.width_spin, self.taper_spin,
              self.glow, self.step_slider, self.trop_x, self.trop_y,
              self.trop_z, self.elast, self.anim_check, self.anim_speed,
              self.orbit_chk, self.fov_spin, self.light_az, self.light_el,
              self.grid_chk, self.fillop, self.amb, self.fog)
        for w in ws:
            w.blockSignals(True)
        self.axiom.setText(st.get("axiom", "F"))
        self.rules.setPlainText(st.get("rules", ""))
        self.draw.setText(st.get("draw", "FG"))
        self.iters.setValue(int(st.get("iters", 5)))
        self.step_slider.setRange(0, self.iters.value())
        self.step_slider.setValue(int(st.get("step", self.iters.value())))
        self.angle.setValue(float(st.get("angle", 22.5)))
        self.jitter.setValue(int(st.get("jitter", 0)))
        self.seed.setValue(int(st.get("seed", 1)))
        self.width_spin.setValue(float(st.get("trunk_w", 2.6)))
        self.taper_spin.setValue(float(st.get("taper", 0.7)))
        self.glow.setChecked(bool(st.get("glow", True)))
        tr = st.get("tropism", [0.0, 0.0, 0.0])
        self.trop_x.setValue(float(tr[0]))
        self.trop_y.setValue(float(tr[1]))
        self.trop_z.setValue(float(tr[2]))
        self.elast.setValue(int(st.get("elast", 0)))
        self.anim_check.setChecked(bool(st.get("animate", True)))
        self.anim_speed.setValue(int(st.get("speed", 22)))
        self.orbit_chk.setChecked(bool(st.get("orbit", False)))
        self.fov_spin.setValue(int(st.get("fov", 0)))
        self.light_az.setValue(int(st.get("light_az", LIGHT_AZ_DEFAULT)))
        self.light_el.setValue(int(st.get("light_el", LIGHT_EL_DEFAULT)))
        self.grid_chk.setChecked(bool(st.get("grid", False)))
        self.fillop.setValue(int(st.get("fill_opacity",
                                        FILL_OPACITY_DEFAULT)))
        self.amb.setValue(int(st.get("ambient", AMBIENT_DEFAULT)))
        self.fog.setValue(int(st.get("fog", 0)))
        mode = st.get("render_mode", "auto")
        mi = self.mode_combo.findData(mode)
        self.mode_combo.setCurrentIndex(mi if mi >= 0 else 0)
        cb = st.get("color_by", "depth")
        ci = self.color_combo.findData(cb)
        self.color_combo.setCurrentIndex(ci if ci >= 0 else 0)
        for w in ws:
            w.blockSignals(False)
        self.jitter_val.setText(f"{self.jitter.value()} %")
        self.step_lbl.setText(f"n = {self.step_slider.value()}")
        self.elast_lbl.setText(f"e = {self.elast.value() / 100:.2f}")
        self.speed_val.setText(str(self.anim_speed.value()))
        self.light_az_lbl.setText(f"az {self.light_az.value()}°")
        self.light_el_lbl.setText(f"el {self.light_el.value()}°")
        self.fillop_lbl.setText(f"{self.fillop.value()} %")
        self.amb_lbl.setText(f"{self.amb.value()} %")
        self.fog_lbl.setText(f"{self.fog.value()} %")
        self.trunk_btn.set_color(st.get("trunk", "#8a4a34"))
        self.tip_btn.set_color(st.get("tip", "#ffb37a"))
        self.bg_btn.set_color(st.get("bg", "#0d151b"))
        cam = st.get("camera")
        self._pending_cam = dict(cam) if isinstance(cam, dict) else None
        self.canvas.set_render_mode(mode)
        self.canvas.set_light(self.light_az.value(), self.light_el.value())
        self.canvas.set_grid(self.grid_chk.isChecked())
        self.canvas.set_fov(self.fov_spin.value())
        self.canvas.set_render_opts(cb, self.fog.value() / 100.0,
                                    self.amb.value() / 100.0,
                                    self.fillop.value() / 100.0)
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(len(PRESETS) if custom else
                                    min(self.preset.currentIndex(),
                                        len(PRESETS)))
        self.preset.blockSignals(False)
        self._request_compute()

    def _collect_state(self) -> dict:
        return dict(
            version=6, app=APP_NAME, app_version=__version__,
            axiom=self.axiom.text(), rules=self.rules.toPlainText(),
            draw=self.draw.text(), iters=self.iters.value(),
            step=self.step_slider.value(), angle=self.angle.value(),
            jitter=self.jitter.value(), seed=self.seed.value(),
            trunk_w=self.width_spin.value(), taper=self.taper_spin.value(),
            trunk=self.trunk_btn.color().name(), tip=self.tip_btn.color().name(),
            bg=self.bg_btn.color().name(), glow=self.glow.isChecked(),
            tropism=[self.trop_x.value(), self.trop_y.value(),
                     self.trop_z.value()],
            elast=self.elast.value(), animate=self.anim_check.isChecked(),
            speed=self.anim_speed.value(), orbit=self.orbit_chk.isChecked(),
            render_mode=self.mode_combo.currentData() or "auto",
            fov=self.fov_spin.value(), light_az=self.light_az.value(),
            light_el=self.light_el.value(), grid=self.grid_chk.isChecked(),
            color_by=self.color_combo.currentData() or "depth",
            fill_opacity=self.fillop.value(), ambient=self.amb.value(),
            fog=self.fog.value(), camera=self.canvas.get_camera())

    def _open_dialog(self):
        start = os.path.dirname(self._path) if self._path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Load L-system", start,
                                              "L-system (*.json)")
        if path:
            self._load_path(path)

    def _load_path(self, path: str, quiet: bool = False) -> bool:
        try:
            with open(path, encoding="utf-8") as f:
                st = json.load(f)
            if not isinstance(st, dict) or "axiom" not in st:
                raise ValueError("not a Fauna system file")
        except Exception as e:
            LOG.warning("load failed: %s (%s)", path, e)
            if not quiet:
                QMessageBox.warning(self, "Load failed",
                                    f"Could not load {path}:\n{e}")
            self._set_status(f"⚠ load failed: {e}", error=True)
            return False
        self._apply_state(st, custom=True)
        self._mark_clean(path)
        LOG.info("loaded %s", path)
        if not quiet:
            self.statusBar().showMessage(f"Loaded {path}", 4000)
        return True

    def _new_system(self):
        self._apply_state(dict(axiom="F", rules="F -> F+F-F-F+F", draw="FG",
                               iters=4, step=4, angle=22.5, jitter=0, seed=1,
                               trunk_w=2.6, taper=0.72, glow=True,
                               trunk="#8a4a34", tip="#ffb37a", bg="#0d151b",
                               tropism=[0, 0, 0], elast=0,
                               render_mode="auto", fov=0,
                               light_az=LIGHT_AZ_DEFAULT,
                               light_el=LIGHT_EL_DEFAULT, grid=False,
                               color_by="depth",
                               fill_opacity=FILL_OPACITY_DEFAULT, fog=0,
                               ambient=AMBIENT_DEFAULT), custom=True)
        self._mark_clean(None)

    def _save_json(self) -> bool:
        if not self._path:
            return self._save_json_as()
        return self._write_json(self._path)

    def _save_json_as(self) -> bool:
        start = os.path.dirname(self._path) if self._path else ""
        path, _ = QFileDialog.getSaveFileName(self, "Save L-system",
                                              start or "fauna_system.json",
                                              "L-system (*.json)")
        if not path:
            return False
        return self._write_json(path)

    def _write_json(self, path: str) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._collect_state(), f, indent=2)
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return False
        self._mark_clean(path)
        self.statusBar().showMessage(f"Saved {path}", 4000)
        LOG.info("saved %s", path)
        return True

    # ---------- analysis actions ----------

    def _show_derivation(self):
        ctx = getattr(self, "_last_ctx", None)
        if not ctx:
            return
        try:
            text = derivation_text(ctx["axiom"], ctx["rules"], ctx["defines"],
                                   self.step_slider.value(), ctx["simple"],
                                   self.seed.value())
        except (ValueError, ZeroDivisionError) as e:
            self._set_status(f"⚠ {e}", error=True)
            return
        TextDialog("Derivation sequence ω₀ → ωₙ", text, parent=self,
                   save_name="derivation.txt").exec()

    def _export_csv(self):
        rows = self._growth_rows
        if not rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export growth table",
                                              "growth.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("n,length,ratio\n")
            for k, t, r in rows:
                rs = "" if r is None or r != r else f"{r:.6f}"
                f.write(f"{k},{t:.6g},{rs}\n")
        self.statusBar().showMessage(f"Saved {path}", 4000)

    # ---------- export / view ----------

    def _fit(self):
        self.canvas._fit_full()

    def _copy_image(self):
        if self.canvas.copy_image_to_clipboard():
            self.statusBar().showMessage("Image copied to clipboard", 3000)
        else:
            self.statusBar().showMessage("Nothing to copy", 3000)

    def _export(self):
        dlg = ExportDialog(self)
        s = self._settings
        dlg.size.setCurrentIndex(int(s.value("export/size_idx", 3)))
        dlg.quality.setCurrentIndex(int(s.value("export/quality_idx", 0)))
        dlg.trans.setChecked(s.value("export/trans", "false") in
                             ("true", True))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        target, transparent, quality = dlg.values()
        s.setValue("export/size_idx", dlg.size.currentIndex())
        s.setValue("export/quality_idx", dlg.quality.currentIndex())
        s.setValue("export/trans", dlg.trans.isChecked())
        start = s.value("last_export_dir", "")
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG",
                                              os.path.join(start, "fauna.png"),
                                              "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        s.setValue("last_export_dir", os.path.dirname(path))
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok = self.canvas.export_image(path, target, transparent, quality)
        finally:
            QApplication.restoreOverrideCursor()
        if ok:
            self.statusBar().showMessage(f"Saved {path}", 4000)
        else:
            self.statusBar().showMessage("Nothing to export", 4000)

    def _export_svg(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG",
                                              "fauna.svg", "SVG (*.svg)")
        if not path:
            return
        if not path.lower().endswith(".svg"):
            path += ".svg"
        if self.canvas.export_svg(path):
            self.statusBar().showMessage(f"Saved {path}", 4000)
        else:
            self.statusBar().showMessage("Nothing to export", 4000)

    def _export_turntable(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder for "
                                                    "turntable frames")
        if not d:
            return
        frames, ok = QInputDialog.getInt(
            self, "Turntable export", "Number of frames (full 360°):",
            24, 4, 144)
        if not ok:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            n = self.canvas.export_turntable(d, frames)
            self.statusBar().showMessage(f"Wrote {n} frames to {d}", 6000)
        finally:
            QApplication.restoreOverrideCursor()

    def _show_help(self):
        TextDialog("Syntax & references", HELP_HTML, html=True,
                   parent=self).exec()

    def _show_about(self):
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> v{__version__}<br><br>"
            "An academic L-system laboratory: D0L/IL systems, parametric "
            "&amp; conditional productions, context sensitivity, weighted "
            "stochastic rules, filled polygons, sphere/disc elements, 3D "
            "turtle with tropism, growth-matrix analysis, and a shaded-tube "
            "mesh renderer with perspective camera, world lighting and fog."
            "<br><br>"
            "<i>Reading:</i> Lindenmayer (1968); Prusinkiewicz &amp; "
            "Lindenmayer, <i>The Algorithmic Beauty of Plants</i> (1990); "
            "Rozenberg &amp; Salomaa (1980).")

    # ---------- status / drag & drop / close ----------

    def _set_status(self, text, error=False):
        self.stats.setText(text)
        self.stats.setStyleSheet("color:#ff9d9d;" if error else "")

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls() and any(
                u.toLocalFile().lower().endswith(".json")
                for u in ev.mimeData().urls()):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            p = url.toLocalFile()
            if p.lower().endswith(".json") and self._load_path(p):
                break

    def closeEvent(self, ev):
        if self._dirty:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle("Unsaved changes")
            box.setText("The current system has unsaved changes.")
            save = box.addButton(QMessageBox.StandardButton.Save)
            box.addButton(QMessageBox.StandardButton.Discard)
            cancel = box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel:
                ev.ignore()
                return
            if clicked is save and not self._save_json():
                ev.ignore()
                return
        self._save_session()
        self._persist_window()
        self._pool.clear()
        self._pool.waitForDone(2000)
        LOG.info("session ended")
        super().closeEvent(ev)


# ---- CLI / entry point --------------------------------------------------------


def _cfg_from_cli(args) -> dict:
    name = args.preset or PRESETS[0][0]
    cfg = None
    for n, c in PRESETS:
        if n.lower() == name.lower():
            cfg = dict(c)
            break
    if cfg is None:
        raise SystemExit(f"preset not found: {name!r} (see --list-presets)")
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.depth is not None:
        cfg["iters"] = args.depth
    if args.angle is not None:
        cfg["angle"] = args.angle
    return cfg


def run_headless_export(args) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication
    _app = QGuiApplication(sys.argv)
    try:
        cfg = _cfg_from_cli(args)
        if args.load:
            with open(args.load, encoding="utf-8") as f:
                st = json.load(f)
            cfg.update({k: st[k] for k in
                        ("axiom", "rules", "draw", "iters", "angle",
                         "trunk_w", "taper", "trunk", "tip", "bg") if k in st})
            if "tropism" in st:
                cfg["trop"] = tuple(st["tropism"])
            cfg["elast"] = st.get("elast", 0.0) / 100.0
            cfg["jitter"] = st.get("jitter", 0) / 100.0
            cfg["fov"] = st.get("fov", 0)
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.depth is not None:
            cfg["iters"] = args.depth
        if args.angle is not None:
            cfg["angle"] = args.angle
        if args.fov is not None:
            cfg["fov"] = args.fov
        step = args.step if args.step is not None else cfg["iters"]
        step = max(0, min(step, cfg["iters"]))
        req = ComputeRequest(
            axiom=cfg["axiom"], rules_text=cfg["rules"], step=step,
            angle=cfg["angle"], jitter=cfg.get("jitter", 0.0),
            seed=cfg.get("seed", 1), draw=cfg["draw"],
            trop=tuple(cfg.get("trop", (0.0, 0.0, 0.0))),
            elast=cfg.get("elast", 0.0))
        res = compute_system(req)
        has3d = res.is3d
        img = render_image(
            res.groups, res.polys, res.elems, res.total, res.bbox3,
            trunk=QColor(cfg["trunk"]), tip=QColor(cfg["tip"]),
            bg=QColor(cfg["bg"]), trunk_w=cfg["trunk_w"], taper=cfg["taper"],
            width_explicit=res.width_explicit, glow=True,
            max_depth=res.max_depth, target=args.size,
            transparent=args.transparent, quality=args.quality,
            persp=persp_distance(res.bbox3, cfg.get("fov", 0)),
            light_w=light_dir(LIGHT_AZ_DEFAULT, LIGHT_EL_DEFAULT),
            fog=args.fog / 100.0, ambient=args.ambient / 100.0,
            yaw=30.0 if has3d else 0.0, pitch=18.0 if has3d else 0.0)
        path = args.export
        if not path.lower().endswith(".png"):
            path += ".png"
        if not img.save(path):
            LOG.error("saving %s failed", path)
            return 1
        print(f"wrote {path} ({img.width()}×{img.height()}, "
              f"{res.total:,} strokes, {len(res.polys):,} polygons, "
              f"{len(res.elems[0]):,} elements)")
        return 0
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    except Exception:
        LOG.exception("headless export failed")
        sys.stderr.write(traceback.format_exc())
        return 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=APP_NAME.lower(),
        description=f"{APP_NAME} {__version__} — academic L-system laboratory",
        epilog="examples:\n"
               "  %(prog)s --preset \"Berry Bush (3D elements)\"\n"
               "  %(prog)s --preset \"Whorl Bush (3D)\" --fov 55 --fog 30 "
               "--export bush.png --size 3200\n"
               "  %(prog)s --selftest",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version=f"{APP_NAME} {__version__}")
    ap.add_argument("--preset", metavar="NAME",
                    help="start with this preset (see --list-presets)")
    ap.add_argument("--list-presets", action="store_true",
                    help="print available presets and exit")
    ap.add_argument("--load", metavar="FILE.json",
                    help="load a system on startup")
    ap.add_argument("--export", metavar="FILE.png",
                    help="render headlessly to PNG and exit")
    ap.add_argument("--size", type=int, default=2400, metavar="N",
                    help="long-edge pixel size for --export (default 2400)")
    ap.add_argument("--quality", choices=["auto", "mesh", "wire"],
                    default="mesh",
                    help="render quality for --export (default mesh)")
    ap.add_argument("--fov", type=int, metavar="DEG",
                    help="perspective FOV for --export (0 = orthographic)")
    ap.add_argument("--fog", type=int, default=0, metavar="0-80",
                    help="distance-fog strength for --export (percent)")
    ap.add_argument("--ambient", type=int, default=AMBIENT_DEFAULT,
                    metavar="0-60",
                    help="ambient floor for --export (percent)")
    ap.add_argument("--transparent", action="store_true",
                    help="transparent background for --export")
    ap.add_argument("--seed", type=int, metavar="N",
                    help="override the random seed")
    ap.add_argument("--depth", type=int, choices=range(0, 16), metavar="0-15",
                    help="override derivation depth")
    ap.add_argument("--step", type=int, metavar="N",
                    help="override the rendered derivation step")
    ap.add_argument("--angle", type=float, metavar="DEG",
                    help="override the turtle angle")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="console log level (file log is always DEBUG)")
    return ap


def main() -> int:
    setup_logging(logging.INFO)
    args = build_parser().parse_args()
    for h in LOG.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler):
            h.setLevel(args.log_level)

    if args.list_presets:
        for name, _cfg in PRESETS:
            print(name)
        return 0
    if args.export:
        return run_headless_export(args)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setOrganizationName(ORG_NAME)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app.setStyleSheet(STYLESHEET)

    def excepthook(kind, value, tb):
        LOG.critical("unhandled exception", exc_info=(kind, value, tb))
        if QApplication.instance():
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle(f"{APP_NAME} — unexpected error")
            box.setText("An unexpected error occurred and has been logged. "
                        "The application will try to continue.")
            box.setDetailedText("".join(traceback.format_exception(
                kind, value, tb)))
            box.exec()
    sys.excepthook = excepthook

    win = MainWindow(load_path=args.load, preset_name=args.preset)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())