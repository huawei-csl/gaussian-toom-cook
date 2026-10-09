#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Generate gtc_linear_arm.S: the linear kernels of the Gaussian products
for aarch64.  Four have the contracts of their x86-64 namesakes in
gtc_linear_x86.S (gen_linear.py):

  mp_limb_t gtc_aors2_n (mp_ptr P, mp_ptr Q, mp_srcptr xP, mp_srcptr xQ,
                         mp_size_t n, unsigned signs)

    {P, n} += {xP, n} and {Q, n} += {xQ, n}; signs & 1 (& 2) subtracts xP
    (xQ) instead.  Returns cP + 2 cQ, the carries out, where a subtraction's
    carry is 1 when it did not borrow -- what the complement addition of the
    x86-64 kernel returns, and what the flag C of aarch64's sbcs is anyway.

  mp_limb_t gtc_add3_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp,
                        mp_srcptr wp, mp_size_t n, unsigned signs)

    {rp, n} = {up, n} + {vp, n} + {wp, n}; signs & 1 (& 2) subtracts vp
    (wp) instead.  Returns the sum of the two carries, 0, 1 or 2, in the same
    sense.  rp may equal up.

  mp_limb_t gtc_rsh1add3_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
                            mp_srcptr cp, mp_size_t n, unsigned comp)

    {rp, n} = (a + b + c mod B^n) >> 1, an arithmetic shift, or with comp
    = 1 (b + c - a mod B^n) >> 1.  n >= 1; rp may equal ap, bp or cp.
    Returns 0.

  void gtc_kara_recomp2 (mp_ptr P, mp_ptr Q, mp_srcptr uP, mp_srcptr uQ,
                         mp_size_t k, mp_limb_t *io)

    The passes of the Karatsuba recomposition (gauss_toom22) for both
    components C = P, Q, each holding a0, a1 (of V0) at C, C+k and b0, b1
    (of V2) at C+2k, C+3k:
      A: t  = a1 + b0   over k limbs, into C+2k;
      B: X1 = t + a0    over k limbs, into C+k;
      C: X2 = t + b1    over k limbs, into C+2k, b1 of io[0] = nb limbs
                        (0 <= nb <= k), zero above;
      D: C[k..3k) += u XOR mask, with carry-in mask & 1 (u = uP, uQ).
    io[1], io[2]: the masks of P and Q, 0 or ~0.  The carries out, 0 or 1,
    to io[3..10]: A of P, A of Q, B of P, B of Q, C of P, C of Q, D of P,
    D of Q.  io has 19 limbs; this kernel leaves io[11..18] alone.

and six of their own:

  mp_limb_t gtc_aors2x_n (mp_ptr dP, mp_ptr dQ, mp_srcptr uP, mp_srcptr uQ,
                          mp_srcptr vP, mp_srcptr vQ, mp_size_t n,
                          unsigned signs)

    {dP, n} = {uP, n} + {vP, n} and {dQ, n} = {uQ, n} + {vQ, n}; signs & 1
    (& 2) subtracts vP (vQ) instead.  Returns cP + 2 cQ as gtc_aors2_n.
    dP may equal uP or vP, dQ uQ or vQ.  (src/gtc.c, reg_add2.)

  mp_limb_t gtc_rsh1add_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp,
                           mp_size_t n)
  mp_limb_t gtc_rsh1sub_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp,
                           mp_size_t n)

    {rp, n} = ({up, n} +- {vp, n} mod B^n) >> 1, the sum read as a signed
    n-limb number and shifted arithmetically (GMP's mpn_rsh1add_n and
    mpn_rsh1sub_n put the carry or borrow on top instead); return the bit
    shifted out.  n >= 1; rp may equal up or vp.

  mp_limb_t gtc_add_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
  mp_limb_t gtc_sub_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)

    GMP's mpn_add_n and mpn_sub_n, but n >= 0: the pass of gtc_aors2_n,
    called directly rather than through libgmp's PLT.

  mp_limb_t gtc_div15_lsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                             mp_srcptr zp, mp_size_t n, unsigned signs)

    {rp, n} = (x + 4 y + 8 z) / 15 modulo B^n, exact; signs & 1 (& 2)
    subtracts 4 y (8 z) instead.  Returns the final h of GMP's
    mpn_bdiv_dbm1c, whose step it is.  n >= 0; rp may equal xp.  (The
    x86-64 kernel's contract; src/gtc.c, the Gaussian Toom-4's q5.)

  mp_limb_t gtc_addlsh123_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
                             mp_srcptr cp, mp_srcptr dp, mp_size_t n)

    {rp, n} = a + 2 b + 4 c + 8 d; returns the limb above (at most 14).
    n >= 1; rp may equal ap.  (The x86-64 kernel's contract; the Gaussian
    Toom-4's evaluation at 2.)

  mp_limb_t gtc_div5_lsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                            mp_size_t n, unsigned sub)

    {rp, n} = (x + 4 y) / 5 modulo B^n, exact; sub & 1 subtracts 4 y
    instead.  Returns the final h of GMP's mpn_bdiv_dbm1c.  n >= 0; rp
    may equal xp.  (The x86-64 kernel's contract; the Toom-4 style
    reduction's q5.)

x86-64 runs two additions along two carry chains at once, CF and OF.
aarch64 has one carry flag.  gtc_aors2_n and gtc_aors2x_n therefore are
two passes in one call, over a leaner loop than GMP's add_n and sub_n
(8 limbs per iteration).  The kernels that need two chains in one pass
(gtc_add3_n, gtc_rsh1add3_n, gtc_kara_recomp2) keep them by blocks of 8
limbs: the block along the flag, its carry parked in a register (cset),
then the next chain over the same limbs with its carry restored (cmp c, #1
sets C = c for c in {0, 1}), and so on.  Two instructions per chain and
block buy one store and one load per limb.  The halvings form the sums
along the flag and shift by extr, which leaves it alone, the results two
limbs behind the sums: stored in the pairs the sums are loaded in.  Every
kernel takes the n mod 8 limbs in steps of 1, 2 and 4 limbs chosen by the
bits of n, through which a chain's carry passes untouched.

Run:  ./tools/gen_linear_arm.py > src/gtc_linear_arm.S  (or: make asm)
"""
import sys


out = []
emit = out.append
label_no = [0]


def lab(stem):
    label_no[0] += 1
    return f".L{stem}{label_no[0]}"


def prologue(name):
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, %function")
    emit("\t.p2align\t4")
    emit(f"{name}:")


def epilogue(name):
    emit("\tret")
    emit(f"\t.size\t{name}, .-{name}")


def dispatch(labels, reg="w5"):
    """Branch on reg's two sign bits to labels[(bit0, bit1)]."""
    one = lab("s1_")
    emit(f"\ttbnz\t{reg}, #0, {one}")
    emit(f"\ttbnz\t{reg}, #1, {labels[(0, 1)]}")
    emit(f"\tb\t{labels[(0, 0)]}")
    emit(f"{one}:")
    emit(f"\ttbnz\t{reg}, #1, {labels[(1, 1)]}")
    emit(f"\tb\t{labels[(1, 0)]}")


def set_carry(sub):
    # C = 0 for an addition (0 + 0 does not carry), 1 for a subtraction
    # (0 - 0 does not borrow)
    emit("\tcmp\txzr, xzr" if sub else "\tcmn\txzr, xzr")


def ldst(op, regs, p, wb, off=0):
    """ldr/ldp (str/stp) of regs at [p + off]; wb: post-increment p by wb
    bytes (only with off = 0)."""
    ins = {"ld": ("ldr", "ldp"), "st": ("str", "stp")}[op][len(regs) - 1]
    rs = ", ".join(regs)
    if wb:
        emit(f"\t{ins}\t{rs}, [{p}], #{wb}")
    elif off:
        emit(f"\t{ins}\t{rs}, [{p}, #{off}]")
    else:
        emit(f"\t{ins}\t{rs}, [{p}]")


class Grp:
    """Memory accesses within a group of G limbs: the first access through
    a pointer register is at offset 0 and post-increments it by 8G; the
    later ones address relative to the moved pointer."""

    def __init__(self, G):
        self.G, self.moved = G, set()

    def acc(self, op, regs, p, off):
        if p not in self.moved:
            assert off == 0, (p, off)
            self.moved.add(p)
            ldst(op, regs, p, 8 * self.G)
        else:
            ldst(op, regs, p, 0, 8 * (off - self.G))


# ---------------------------------------------------------------- passes

def pass_(d, u, v, n, sub, res, cond="cs"):
    """{d, n} = {u, n} +- {v, n}, n >= 0 in the register n (kept), C as
    cond to res.  d may equal u or v; d and u may be the same register (in
    place), which then moves with the stores only.  Uses x8..x15, x17.

    8 limbs per iteration, two groups of 4 (ldp, ldp, ldp, ldp, 4 adcs,
    stp, stp): 4-12% faster than 4 per iteration on the Kunpeng 920."""
    op = "sbcs" if sub else "adcs"
    inpl = d == u
    rem, grp, done = lab("r"), lab("g"), lab("d")
    R = ["x8", "x9", "x10", "x11"]
    S = ["x12", "x13", "x14", "x15"]

    def step(ra, rb):
        # len(ra) limbs, 1 or 2
        ldst("ld", ra, u, 0 if inpl else 8 * len(ra))
        ldst("ld", rb, v, 8 * len(rb))
        for x, y in zip(ra, rb):
            emit(f"\t{op}\t{x}, {x}, {y}")
        ldst("st", ra, d, 8 * len(ra))

    def quad(g, base):
        # 4 limbs at group offset base
        g.acc("ld", R[:2], u, base)
        g.acc("ld", R[2:], u, base + 2)
        g.acc("ld", S[:2], v, base)
        g.acc("ld", S[2:], v, base + 2)
        for x, y in zip(R, S):
            emit(f"\t{op}\t{x}, {x}, {y}")
        g.acc("st", R[:2], d, base)
        g.acc("st", R[2:], d, base + 2)

    emit(f"\tlsr\tx17, {n}, #3")
    set_carry(sub)
    # The n mod 8 limbs first, one step each of 1, 2 and 4 limbs on n's
    # bits (a loop of single limbs cost the Toom-3 interpolation up to 16%
    # at n = 3 mod 4 on the Kunpeng-920).
    emit(f"\ttbz\t{n}, #0, {rem}_1")
    step(R[:1], S[:1])
    emit(f"{rem}_1:")
    emit(f"\ttbz\t{n}, #1, {rem}_2")
    step(R[:2], S[:2])
    emit(f"{rem}_2:")
    emit(f"\ttbz\t{n}, #2, {rem}e")
    quad(Grp(4), 0)
    emit(f"{rem}e:")
    emit(f"\tcbz\tx17, {done}")
    emit(f"{grp}:")
    g = Grp(8)
    quad(g, 0)
    quad(g, 4)
    emit("\tsub\tx17, x17, #1")
    emit(f"\tcbnz\tx17, {grp}")
    emit(f"{done}:")
    emit(f"\tcset\t{res}, {cond}")


def addsub_n(name, sub):
    """gtc_add_n, gtc_sub_n: one pass, the carry (borrow) out.  Used for
    all of gtc.c's add_n, sub_n, mpn_add and mpn_sub (the latter by
    gtc_add, gtc_sub): 5-18% faster than libgmp's (through the PLT) from 9
    limbs."""
    prologue(name)
    pass_("x0", "x1", "x2", "x3", sub, "x0", cond="cc" if sub else "cs")
    epilogue(name)


def aors2():
    name = "gtc_aors2_n"
    prologue(name)
    # P first, by bit 0; then Q, by bit 1.
    for c, x, bit, res in (("x0", "x2", 0, "x6"), ("x1", "x3", 1, "x7")):
        sub_l, next_l = lab("sub"), lab("next")
        emit(f"\ttbnz\tw5, #{bit}, {sub_l}")
        pass_(c, c, x, "x4", False, res)
        emit(f"\tb\t{next_l}")
        emit(f"{sub_l}:")
        pass_(c, c, x, "x4", True, res)
        emit(f"{next_l}:")
    emit("\tadd\tx0, x6, x7, lsl #1")
    epilogue(name)


def aors2x():
    name = "gtc_aors2x_n"
    prologue(name)
    # x0..x7 = dP, dQ, uP, uQ, vP, vQ, n, signs.  P first, by bit 0, its
    # carry into x2 (uP is done with by then); then Q, by bit 1, into x3.
    for d, u, v, bit, res in (("x0", "x2", "x4", 0, "x2"),
                              ("x1", "x3", "x5", 1, "x3")):
        sub_l, next_l = lab("sub"), lab("next")
        emit(f"\ttbnz\tw7, #{bit}, {sub_l}")
        pass_(d, u, v, "x6", False, res)
        emit(f"\tb\t{next_l}")
        emit(f"{sub_l}:")
        pass_(d, u, v, "x6", True, res)
        emit(f"{next_l}:")
    emit("\tadd\tx0, x2, x3, lsl #1")
    epilogue(name)


# ---------------------------------------------------------------- add3

def chain_block(g, U, nl, src, op, rot, park, store=None):
    """The nl limbs of U (registers) op= {src}, along the flag restored
    from park (cmp park, #1) and parked again (cset).  The loads of src
    rotate through the register pair rot; with store, the results go to
    store (a pointer) right after."""
    if park:
        emit(f"\tcmp\t{park}, #1")
    for i in range(0, nl, 2):
        k = min(2, nl - i)
        g.acc("ld", rot[:k], src, i)
        for j in range(k):
            emit(f"\t{op}\t{U[i + j]}, {U[i + j]}, {rot[j]}")
        if store:
            g.acc("st", U[i:i + k], store, i)
    if park:
        emit(f"\tcset\t{park}, cs")


def add3():
    """gtc_add3_n: per sign combination, the n mod 8 limbs first, one pass
    per chain over steps of 1, 2 and 4 limbs chosen by n's bits (the carry
    passing through the branches: one park per chain for all of them, the
    sums in x8..x14); then blocks of 8, the u + v sums in x8..x15, the
    loads of v and w rotating through x16, x17, no callee-saved register.
    On the Kunpeng 920 6-12% faster than blocks of 1, 2, 4 limbs each
    parked at 9 to 25 limbs (most calls).  The signs are applied by
    branches, not by eor with masks: the signs mostly repeat."""
    name = "gtc_add3_n"
    prologue(name)
    labels = {sv: lab(f"v{sv[0]}{sv[1]}_") for sv in
              ((0, 0), (0, 1), (1, 0), (1, 1))}
    end = lab("end")
    dispatch(labels)
    U = ["x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
    rot = ["x16", "x17"]
    for (nv, nw), l in labels.items():
        opv = "sbcs" if nv else "adcs"
        opw = "sbcs" if nw else "adcs"
        emit(f"{l}:")
        slots, at = {}, 0
        for b in (0, 1, 2):
            slots[b] = (at, 1 << b)
            at += 1 << b
        # u + v of the n mod 8 limbs, parked in x6
        set_carry(nv)
        for b in (0, 1, 2):
            skip = lab("rv")
            s0, nl = slots[b]
            emit(f"\ttbz\tx4, #{b}, {skip}")
            g = Grp(nl)
            for i in range(0, nl, 2):
                k = min(2, nl - i)
                g.acc("ld", U[s0 + i:s0 + i + k], "x1", i)
            for i in range(0, nl, 2):
                k = min(2, nl - i)
                g.acc("ld", rot[:k], "x2", i)
                for j in range(k):
                    r = U[s0 + i + j]
                    emit(f"\t{opv}\t{r}, {r}, {rot[j]}")
            emit(f"{skip}:")
        emit("\tcset\tx6, cs")
        # + w, parked in x7, stored
        set_carry(nw)
        for b in (0, 1, 2):
            skip = lab("rw")
            s0, nl = slots[b]
            emit(f"\ttbz\tx4, #{b}, {skip}")
            chain_block(Grp(nl), U[s0:s0 + nl], nl, "x3", opw, rot, None,
                        store="x0")
            emit(f"{skip}:")
        emit("\tcset\tx7, cs")
        # the blocks of 8
        grp, done = lab("g"), lab("d")
        emit("\tlsr\tx5, x4, #3")
        emit(f"\tcbz\tx5, {done}")
        emit("\t.p2align\t4")
        emit(f"{grp}:")
        g = Grp(8)
        for i in range(0, 8, 2):
            g.acc("ld", U[i:i + 2], "x1", i)
        chain_block(g, U, 8, "x2", opv, rot, "x6")
        chain_block(g, U, 8, "x3", opw, rot, "x7", store="x0")
        emit("\tsub\tx5, x5, #1")
        emit(f"\tcbnz\tx5, {grp}")
        emit(f"{done}:")
        emit(f"\tb\t{end}")
    emit(f"{end}:")
    emit("\tadd\tx0, x6, x7")
    epilogue(name)


# ---------------------------------------------------------------- halvings

def misaligned_head(regs, t0, t1, skip):
    """Branch to skip unless two of the three pointers regs are 8 mod 16
    (bit 3 of the majority, in t0)."""
    a, b, c = regs
    emit(f"\tand\t{t0}, {a}, {b}")
    emit(f"\torr\t{t1}, {a}, {b}")
    emit(f"\tand\t{t1}, {t1}, {c}")
    emit(f"\torr\t{t0}, {t0}, {t1}")
    emit(f"\ttbz\t{t0}, #3, {skip}")


def rsh1(name, sub, sh=1):
    """gtc_rsh1add_n, gtc_rsh1sub_n in one pass: the sums along the flag,
    the shift by extr, the results two limbs behind the sums (in x6, x7
    between blocks), so that they are stored in the pairs the sums are
    loaded in: aligned where rp has the parity of up and vp.  Two sums
    before the blocks, three (the first result stored alone) when two of
    rp, up, vp are 8 mod 16; blocks of 8; the rest in steps of 4, 2, 1 on
    n's bits; the last result shifted arithmetically (a signed halving;
    GMP's put the carry on top).
    With sh > 1 the sum shifted by sh bits (the bits shifted out
    returned)."""
    op0, op = ("subs", "sbcs") if sub else ("adds", "adcs")
    prologue(name)
    S = ["x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
    rot = ["x16", "x17"]
    p0, p1 = "x6", "x7"
    one = lab("one")
    emit("\tldr\tx16, [x1], #8")
    emit("\tldr\tx17, [x2], #8")
    emit(f"\t{op0}\t{p0}, x16, x17")
    emit(f"\tand\tx5, {p0}, #{(1 << sh) - 1}\t\t/* the bit{'s' if sh > 1 else ''} shifted out */")
    emit("\tsub\tx3, x3, #1")
    emit(f"\tcbz\tx3, {one}")
    emit("\tldr\tx16, [x1], #8")
    emit("\tldr\tx17, [x2], #8")
    emit(f"\t{op}\t{p1}, x16, x17")
    emit("\tsub\tx3, x3, #1")
    h = lab("h")
    misaligned_head(("x0", "x1", "x2"), "x4", "x16", h)
    emit(f"\tcbz\tx3, {h}")
    emit("\tldr\tx16, [x1], #8")
    emit("\tldr\tx17, [x2], #8")
    emit(f"\t{op}\tx8, x16, x17")
    emit(f"\textr\tx16, {p1}, {p0}, #{sh}")
    emit("\tstr\tx16, [x0], #8")
    emit(f"\tmov\t{p0}, {p1}")
    emit(f"\tmov\t{p1}, x8")
    emit("\tsub\tx3, x3, #1")
    emit(f"{h}:")

    def block(g, nl):
        # nl sums into S; nl results, two behind
        for i in range(0, nl, 2):
            k = min(2, nl - i)
            g.acc("ld", S[i:i + k], "x1", i)
        allr = [p0, p1] + S[:nl]	# s_{-2}, s_{-1}, s_0, ...
        for i in range(0, nl, 2):
            k = min(2, nl - i)
            g.acc("ld", rot[:k], "x2", i)
            for j in range(k):
                emit(f"\t{op}\t{S[i + j]}, {S[i + j]}, {rot[j]}")
            for j in range(k):
                lo, hi = allr[i + j], allr[i + j + 1]
                emit(f"\textr\t{rot[j]}, {hi}, {lo}, #{sh}")
            g.acc("st", rot[:k], "x0", i)
        emit(f"\tmov\t{p0}, {allr[nl]}")
        emit(f"\tmov\t{p1}, {allr[nl + 1]}")

    grp, done = lab("g"), lab("d")
    emit("\tlsr\tx4, x3, #3")
    emit(f"\tcbz\tx4, {done}")
    emit("\t.p2align\t4")
    emit(f"{grp}:")
    block(Grp(8), 8)
    emit("\tsub\tx4, x4, #1")
    emit(f"\tcbnz\tx4, {grp}")
    emit(f"{done}:")
    for b in (2, 1, 0):
        skip = lab("t")
        emit(f"\ttbz\tx3, #{b}, {skip}")
        block(Grp(1 << b), 1 << b)
        emit(f"{skip}:")
    # the last limb shifted arithmetically: the halving of a signed value
    # at width n (GMP's kernels put the carry on top instead)
    emit(f"\tasr\tx17, {p1}, #{sh}")
    emit(f"\textr\tx16, {p1}, {p0}, #{sh}")
    emit("\tstp\tx16, x17, [x0]")
    emit("\tmov\tx0, x5")
    emit("\tret")
    emit(f"{one}:")
    emit(f"\tasr\tx16, {p0}, #{sh}")
    emit("\tstr\tx16, [x0]")
    emit("\tmov\tx0, x5")
    epilogue(name)


def rsh1add3(name="gtc_rsh1add3_n", sh=1):
    """gtc_rsh1add3_n: the two chains of the three-operand sum parked by
    blocks as in gtc_add3_n (a +- b along the flag, then + c), the halving
    as in rsh1: results two limbs behind the sums (x19, x20 between
    blocks), three sums first when two of ap, bp, cp are 8 mod 16, the
    rest in steps of 4, 2, 1; the last sum shifted arithmetically.  comp
    = 1 makes the first chain b - a (sbcs).  0.61-0.95 the cycles of
    GMP's add_n (sub_n) and rsh1add_n.  With sh > 1 the sum shifted by sh bits (gtc_rsh2add3_n,
    gtc_rsh3add3_n)."""
    prologue(name)
    S = ["x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
    rot = ["x16", "x17"]
    p0, p1 = "x19", "x20"
    emit(f"\tstp\t{p0}, {p1}, [sp, #-16]!")
    end = lab("h3e")
    lc = lab("h3c")
    emit(f"\ttbnz\tw5, #0, {lc}")
    for comp in (0, 1):
        if comp:
            emit(f"{lc}:")
        one = lab("one")

        def op1(r, b, first=False):
            if comp:
                emit(f"\t{'subs' if first else 'sbcs'}\t{r}, {b}, {r}")
            else:
                emit(f"\t{'adds' if first else 'adcs'}\t{r}, {r}, {b}")

        def op2(r, c, first=False):
            emit(f"\t{'adds' if first else 'adcs'}\t{r}, {r}, {c}")

        def single(dst, first=False):
            # one limb of both chains into x8 (then dst)
            emit("\tldr\tx8, [x1], #8")
            emit("\tldr\tx16, [x2], #8")
            if not first:
                emit("\tcmp\tx6, #1")
            op1("x8", "x16", first)
            emit("\tcset\tx6, cs")
            emit("\tldr\tx16, [x3], #8")
            if not first:
                emit("\tcmp\tx7, #1")
            op2("x8", "x16", first)
            emit("\tcset\tx7, cs")
            if dst != "x8":
                emit(f"\tmov\t{dst}, x8")

        single(p0, True)
        emit("\tsub\tx4, x4, #1")
        emit(f"\tcbz\tx4, {one}")
        single(p1)
        emit("\tsub\tx4, x4, #1")
        h = lab("h")
        misaligned_head(("x1", "x2", "x3"), "x9", "x10", h)
        emit(f"\tcbz\tx4, {h}")
        single("x8")
        emit(f"\textr\tx16, {p1}, {p0}, #{sh}")
        emit("\tstr\tx16, [x0], #8")
        emit(f"\tmov\t{p0}, {p1}")
        emit(f"\tmov\t{p1}, x8")
        emit("\tsub\tx4, x4, #1")
        emit(f"{h}:")

        def sec1(g, nl, s0):
            # a +- b of nl limbs into S[s0:]
            for i in range(0, nl, 2):
                k = min(2, nl - i)
                g.acc("ld", S[s0 + i:s0 + i + k], "x1", i)
            for i in range(0, nl, 2):
                k = min(2, nl - i)
                g.acc("ld", rot[:k], "x2", i)
                for j in range(k):
                    op1(S[s0 + i + j], rot[j])

        def sec2(g, nl, s0):
            # + c, the results two behind
            allr = [p0, p1] + S[s0:s0 + nl]
            for i in range(0, nl, 2):
                k = min(2, nl - i)
                g.acc("ld", rot[:k], "x3", i)
                for j in range(k):
                    op2(S[s0 + i + j], rot[j])
                for j in range(k):
                    emit(f"\textr\t{rot[j]}, {allr[i + j + 1]}, {allr[i + j]}, #{sh}")
                g.acc("st", rot[:k], "x0", i)
            emit(f"\tmov\t{p0}, {allr[nl]}")
            emit(f"\tmov\t{p1}, {allr[nl + 1]}")

        grp, done = lab("g"), lab("d")
        emit("\tlsr\tx5, x4, #3")
        emit(f"\tcbz\tx5, {done}")
        emit("\t.p2align\t4")
        emit(f"{grp}:")
        g = Grp(8)
        emit("\tcmp\tx6, #1")
        sec1(g, 8, 0)
        emit("\tcset\tx6, cs")
        emit("\tcmp\tx7, #1")
        sec2(g, 8, 0)
        emit("\tcset\tx7, cs")
        emit("\tsub\tx5, x5, #1")
        emit(f"\tcbnz\tx5, {grp}")
        emit(f"{done}:")
        slots = {2: 0, 1: 4, 0: 6}
        emit("\tcmp\tx6, #1")
        for b in (2, 1, 0):
            skip = lab("t1")
            emit(f"\ttbz\tx4, #{b}, {skip}")
            sec1(Grp(1 << b), 1 << b, slots[b])
            emit(f"{skip}:")
        emit("\tcmp\tx7, #1")
        for b in (2, 1, 0):
            skip = lab("t2")
            emit(f"\ttbz\tx4, #{b}, {skip}")
            sec2(Grp(1 << b), 1 << b, slots[b])
            emit(f"{skip}:")
        # the last two results; the last sum shifted arithmetically
        emit(f"\textr\tx16, {p1}, {p0}, #{sh}")
        emit(f"\tasr\tx17, {p1}, #{sh}")
        emit("\tstp\tx16, x17, [x0]")
        emit(f"\tb\t{end}")
        emit(f"{one}:")
        emit(f"\tasr\tx16, {p0}, #{sh}")
        emit("\tstr\tx16, [x0]")
        emit(f"\tb\t{end}")
    emit(f"{end}:")
    emit(f"\tldp\t{p0}, {p1}, [sp], #16")
    emit("\tmov\tx0, #0")
    epilogue(name)


# ---------------------------------------------------------------- Karatsuba

def kara_recomp2():
    """gtc_kara_recomp2: per component two passes, the chains parked by
    blocks of 8 as in gtc_add3_n.  Pass 1 over the low k limbs: A, B and D
    in the same eight registers (t = a1 + b0 stored, then t + a0, then
    + u ^ mask, stored as X1); pass 2 over the high k: C and D (t + b1,
    then + u_hi ^ mask), then the k - nb limbs above b1 one at a time.
    The mask is applied by eor, not by a branch on its (random) sign.  The
    components are a loop of two (x0, x2 then x1, x3; io advanced by one
    limb), which halves the code, for the instruction cache.  0.69-0.93
    the cycles of kara_recompose3 twice (GMP's add_n and two gtc_add3_n
    each) at k = 4..32."""
    name = "gtc_kara_recomp2"
    prologue(name)
    T = ["x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
    rot = ["x16", "x17"]
    emit("\tstp\tx19, x20, [sp, #-64]!")
    emit("\tstp\tx21, x22, [sp, #16]")
    emit("\tstp\tx23, x24, [sp, #32]")
    emit("\tstp\tx25, x26, [sp, #48]")
    pa0, pa1r, pa1w, pb0, pu, cnt = "x6", "x7", "x19", "x20", "x21", "x22"
    cA, cB, cD, msk = "x23", "x24", "x25", "x26"
    slots = {2: 0, 1: 4, 0: 6}

    def sec(g, nl, s0, src, park, ldT=None, stT=None, xor=False):
        """One chain over nl limbs at T[s0:]: T (first loaded from ldT)
        += src (^ mask, or 0 with src "zero"), then stored to stT."""
        R = T[s0:s0 + nl]
        if ldT:
            for i in range(0, nl, 2):
                k = min(2, nl - i)
                g.acc("ld", R[i:i + k], ldT, i)
        for i in range(0, nl, 2):
            k = min(2, nl - i)
            g.acc("ld", rot[:k], src, i)
            if xor:
                for j in range(k):
                    emit(f"\teor\t{rot[j]}, {rot[j]}, {msk}")
            for j in range(k):
                emit(f"\tadcs\t{R[i + j]}, {R[i + j]}, {rot[j]}")
            if stT:
                g.acc("st", R[i:i + k], stT, i)

    def chains(spec, nbits, loop_cnt):
        """spec: (park, kwargs) per chain; blocks of 8 counted by loop_cnt,
        then the bits 2, 1, 0 of nbits"""
        grp, done = lab("g"), lab("d")
        emit(f"\tcbz\t{loop_cnt}, {done}")
        emit("\t.p2align\t4")
        emit(f"{grp}:")
        g = Grp(8)
        for park, kw in spec:
            emit(f"\tcmp\t{park}, #1")
            sec(g, 8, 0, park=park, **kw)
            emit(f"\tcset\t{park}, cs")
        emit(f"\tsub\t{loop_cnt}, {loop_cnt}, #1")
        emit(f"\tcbnz\t{loop_cnt}, {grp}")
        emit(f"{done}:")
        for park, kw in spec:
            emit(f"\tcmp\t{park}, #1")
            for b in (2, 1, 0):
                skip = lab("t")
                emit(f"\ttbz\t{nbits}, #{b}, {skip}")
                sec(Grp(1 << b), 1 << b, slots[b], park=park, **kw)
                emit(f"{skip}:")
            emit(f"\tcset\t{park}, cs")

    def component(C, u):
        # io (x5) points one limb further for Q: io[1 + c] at [x5, #8]
        emit(f"\tldr\t{msk}, [x5, #8]")
        emit(f"\tand\t{cD}, {msk}, #1")
        emit(f"\tmov\t{cA}, #0")
        emit(f"\tmov\t{cB}, #0")
        emit(f"\tmov\t{pa0}, {C}")
        emit(f"\tadd\t{pa1r}, {C}, x4, lsl #3")
        emit(f"\tmov\t{pa1w}, {pa1r}")
        emit(f"\tadd\t{pb0}, {pa1r}, x4, lsl #3")
        emit(f"\tmov\t{pu}, {u}")
        emit(f"\tlsr\t{cnt}, x4, #3")
        # pass 1: t = a1 + b0 (stored), then t + a0, then + u ^ mask
        chains([(cA, dict(src=pb0, ldT=pa1r, stT=pb0)),
                (cB, dict(src=pa0)),
                (cD, dict(src=pu, stT=pa1w, xor=True))], "x4", cnt)
        emit(f"\tstr\t{cA}, [x5, #24]")
        emit(f"\tstr\t{cB}, [x5, #40]")
        # pass 2: X2 = t + b1 + u_hi ^ mask; pb0 and pu have arrived at
        # b1 and u + k
        ptr, ptw, cC, nb = "x6", "x7", "x23", "x24"
        emit(f"\tadd\t{ptr}, {C}, x4, lsl #4")
        emit(f"\tmov\t{ptw}, {ptr}")
        emit(f"\tmov\t{cC}, #0")
        emit(f"\tmov\t{nb}, x27")
        emit(f"\tlsr\t{cnt}, {nb}, #3")
        chains([(cC, dict(src=pb0, ldT=ptr)),
                (cD, dict(src=pu, stT=ptw, xor=True))], nb, cnt)
        # the k - nb limbs above b1: t + 0 + u_hi ^ mask
        zl, zd = lab("z"), lab("zd")
        emit(f"\tsubs\t{nb}, x4, {nb}")
        emit(f"\tb.eq\t{zd}")
        emit(f"{zl}:")
        emit(f"\tldr\tx8, [{ptr}], #8")
        emit(f"\tcmp\t{cC}, #1")
        emit("\tadcs\tx8, x8, xzr")
        emit(f"\tcset\t{cC}, cs")
        emit(f"\tldr\tx16, [{pu}], #8")
        emit(f"\teor\tx16, x16, {msk}")
        emit(f"\tcmp\t{cD}, #1")
        emit("\tadcs\tx8, x8, x16")
        emit(f"\tcset\t{cD}, cs")
        emit(f"\tstr\tx8, [{ptw}], #8")
        emit(f"\tsub\t{nb}, {nb}, #1")
        emit(f"\tcbnz\t{nb}, {zl}")
        emit(f"{zd}:")
        emit(f"\tstr\t{cC}, [x5, #56]")
        emit(f"\tstr\t{cD}, [x5, #72]")

    comp = lab("comp")
    emit("\tstp\tx27, x28, [sp, #-16]!")
    emit("\tldr\tx27, [x5]\t\t/* nb */")
    emit("\tmov\tx28, #2")
    emit(f"{comp}:")
    component("x0", "x2")
    emit("\tmov\tx0, x1")
    emit("\tmov\tx2, x3")
    emit("\tadd\tx5, x5, #8")
    emit("\tsub\tx28, x28, #1")
    emit(f"\tcbnz\tx28, {comp}")
    emit("\tldp\tx27, x28, [sp], #16")
    emit("\tldp\tx21, x22, [sp, #16]")
    emit("\tldp\tx23, x24, [sp, #32]")
    emit("\tldp\tx25, x26, [sp, #48]")
    emit("\tldp\tx19, x20, [sp], #64")
    epilogue(name)



def lsh_chain(g, U, L, A, nl, src, prev, sh, op, t0, t1):
    """U[i] op= the limbs of src shifted left by 64 - sh bits (extr from
    the limb below, prev before the first), along the flag.  Blocks of 4
    and 8 load their pairs into L and A in turn, the last limb straight
    into prev, so that no mov carries the limb below; smaller ones load
    into L and copy the top limb to prev."""
    if nl >= 4:
        P = [tuple(L), tuple(A)]
        hp = prev
        for j in range(nl // 2):
            lo, hi = P[j % 2]
            if j == nl // 2 - 1:
                hi = prev
            g.acc("ld", [lo, hi], src, 2 * j)
            emit(f"\textr\t{t0}, {lo}, {hp}, #{sh}")
            emit(f"\textr\t{t1}, {hi}, {lo}, #{sh}")
            emit(f"\t{op}\t{U[2 * j]}, {U[2 * j]}, {t0}")
            emit(f"\t{op}\t{U[2 * j + 1]}, {U[2 * j + 1]}, {t1}")
            hp = hi
        return
    for i in range(0, nl, 2):
        k = min(2, nl - i)
        g.acc("ld", L[:k], src, i)
        emit(f"\textr\t{t0}, {L[0]}, {prev}, #{sh}")
        if k == 2:
            emit(f"\textr\t{t1}, {L[1]}, {L[0]}, #{sh}")
        emit(f"\t{op}\t{U[i]}, {U[i]}, {t0}")
        if k == 2:
            emit(f"\t{op}\t{U[i + 1]}, {U[i + 1]}, {t1}")
        emit(f"\tmov\t{prev}, {L[k - 1]}")


def dbm1_step(u, h, p0, p1, bd, nomov=True):
    """GMP's mpn_bdiv_dbm1c step on the limb u, the quotient limb into u:
    by subs itself, or (nomov false) into h and copied, as the div5_add
    kernels do, where it was faster at small sizes."""
    emit(f"\tmul\t{p0}, {u}, {bd}")
    emit(f"\tumulh\t{p1}, {u}, {bd}")
    if nomov:
        emit(f"\tsubs\t{u}, {h}, {p0}")
        emit(f"\tsbc\t{h}, {u}, {p1}")
    else:
        emit(f"\tsubs\t{h}, {h}, {p0}")
        emit(f"\tmov\t{u}, {h}")
        emit(f"\tsbc\t{h}, {h}, {p1}")


def div15_lsh():
    """gtc_div15_lsh_n: {rp, n} = (x + 4 y + 8 z) / 15 modulo B^n, exact,
    signs & 1 (& 2) subtracting 4 y (8 z) instead (the contract of the
    x86-64 kernel).  Per block of 8 limbs (the n mod 8 first, by n's bits
    in blocks of 1, 2 and 4): x loaded, the y chain along the flag (the
    shifted limbs by extr, which leaves it alone), parked; the z chain
    likewise; then GMP's mpn_bdiv_dbm1c step on each limb (mul, umulh,
    subs, sbc: its flag lives within a limb), the quotients stored.
    (Blocks of 16, with every register but x18, were measured 4-6% slower
    on the Kunpeng 920 from 41 to 1001 limbs; on the x86-64 they pay.)"""
    name = "gtc_div15_lsh_n"
    prologue(name)
    emit("\tstp\tx19, x20, [sp, #-64]!")
    emit("\tstp\tx21, x22, [sp, #16]")
    emit("\tstp\tx23, x24, [sp, #32]")
    emit("\tstp\tx25, x26, [sp, #48]")
    U = ["x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
    L = ["x16", "x17"]
    yp_, zp_, t0, t1, bd, h, p0, p1 = (f"x{r}" for r in range(19, 27))
    labels = {sv: lab(f"qv{sv[0]}{sv[1]}_") for sv in
              ((0, 0), (0, 1), (1, 0), (1, 1))}
    end = lab("qend")
    emit(f"\tmov\t{bd}, #0x1111111111111111\t// (B-1)/15")
    emit(f"\tmov\t{h}, xzr")
    emit(f"\tmov\t{yp_}, xzr\t\t// the limbs below y and z: 0")
    emit(f"\tmov\t{zp_}, xzr")
    dispatch(labels)

    def chain(g, nl, src, prev, sh, op, park):
        emit(f"\tcmp\t{park}, #1")
        lsh_chain(g, U, L, (p0, p1), nl, src, prev, sh, op, t0, t1)
        emit(f"\tcset\t{park}, cs")

    def block(nl, opy, opz):
        g = Grp(nl)
        for i in range(0, nl, 2):
            g.acc("ld", U[i:i + min(2, nl - i)], "x1", i)
        chain(g, nl, "x2", yp_, 62, opy, "x6")
        chain(g, nl, "x3", zp_, 61, opz, "x7")
        for i in range(nl):
            dbm1_step(U[i], h, p0, p1, bd)
        for i in range(0, nl, 2):
            g.acc("st", U[i:i + min(2, nl - i)], "x0", i)

    for (ny, nz), l in labels.items():
        opy = "sbcs" if ny else "adcs"
        opz = "sbcs" if nz else "adcs"
        emit(f"{l}:")
        # the parked carries: C = 1 where a chain subtracts (no borrow)
        emit(f"\tmov\tx6, #{ny}")
        emit(f"\tmov\tx7, #{nz}")
        for b in (0, 1, 2):
            skip = lab("qr")
            emit(f"\ttbz\tx4, #{b}, {skip}")
            block(1 << b, opy, opz)
            emit(f"{skip}:")
        grp, done = lab("qg"), lab("qd")
        emit("\tlsr\tx5, x4, #3")
        emit(f"\tcbz\tx5, {done}")
        emit("\t.p2align\t4")
        emit(f"{grp}:")
        block(8, opy, opz)
        emit("\tsub\tx5, x5, #1")
        emit(f"\tcbnz\tx5, {grp}")
        emit(f"{done}:")
        emit(f"\tb\t{end}")
    emit(f"{end}:")
    emit(f"\tmov\tx0, {h}")
    emit("\tldp\tx21, x22, [sp, #16]")
    emit("\tldp\tx23, x24, [sp, #32]")
    emit("\tldp\tx25, x26, [sp, #48]")
    emit("\tldp\tx19, x20, [sp], #64")
    epilogue(name)


def div5_lsh():
    """gtc_div5_lsh_n: {rp, n} = (x + 4 y) / 5 modulo B^n, exact, sub & 1
    subtracting 4 y instead (the contract of the x86-64 kernel).
    gtc_div15_lsh_n with one chain: per block of 8 limbs (the n mod 8
    first, by n's bits in blocks of 1, 2 and 4) x loaded, the y chain along
    the flag, parked; then GMP's mpn_bdiv_dbm1c step with (B-1)/5."""
    name = "gtc_div5_lsh_n"
    prologue(name)
    emit("\tstp\tx19, x20, [sp, #-64]!")
    emit("\tstp\tx21, x22, [sp, #16]")
    emit("\tstp\tx23, x24, [sp, #32]")
    emit("\tstp\tx25, x26, [sp, #48]")
    U = ["x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
    L = ["x16", "x17"]
    yp_, t0, t1, bd, h, p0, p1 = (f"x{r}" for r in range(19, 26))
    labels = {v: lab(f"dv{v}_") for v in (0, 1)}
    end = lab("dend")
    emit(f"\tmov\t{bd}, #0x3333333333333333\t// (B-1)/5")
    emit(f"\tmov\t{h}, xzr")
    emit(f"\tmov\t{yp_}, xzr\t\t// the limb below y: 0")
    emit(f"\ttbnz\tw4, #0, {labels[1]}")
    emit(f"\tb\t{labels[0]}")

    def block(nl, opy):
        g = Grp(nl)
        for i in range(0, nl, 2):
            g.acc("ld", U[i:i + min(2, nl - i)], "x1", i)
        emit("\tcmp\tx6, #1")
        lsh_chain(g, U, L, (p0, p1), nl, "x2", yp_, 62, opy, t0, t1)
        emit("\tcset\tx6, cs")
        for i in range(nl):
            dbm1_step(U[i], h, p0, p1, bd)
        for i in range(0, nl, 2):
            g.acc("st", U[i:i + min(2, nl - i)], "x0", i)

    for ny, l in labels.items():
        opy = "sbcs" if ny else "adcs"
        emit(f"{l}:")
        emit(f"\tmov\tx6, #{ny}\t\t// C = 1 where the chain subtracts")
        for b in (0, 1, 2):
            skip = lab("dr")
            emit(f"\ttbz\tx3, #{b}, {skip}")
            block(1 << b, opy)
            emit(f"{skip}:")
        grp, done = lab("dg"), lab("dd")
        emit("\tlsr\tx5, x3, #3")
        emit(f"\tcbz\tx5, {done}")
        emit("\t.p2align\t4")
        emit(f"{grp}:")
        block(8, opy)
        emit("\tsub\tx5, x5, #1")
        emit(f"\tcbnz\tx5, {grp}")
        emit(f"{done}:")
        emit(f"\tb\t{end}")
    emit(f"{end}:")
    emit(f"\tmov\tx0, {h}")
    emit("\tldp\tx21, x22, [sp, #16]")
    emit("\tldp\tx23, x24, [sp, #32]")
    emit("\tldp\tx25, x26, [sp, #48]")
    emit("\tldp\tx19, x20, [sp], #64")
    epilogue(name)


def div5_add():
    """gtc_div5_add_n: {rp, n} = (x + y) / 5 modulo B^n, exact; mode 1:
    (x - y) / 5, mode 2: (y - x) / 5 (the contract of the x86-64 kernel).
    div5_lsh without the shift of y: per block of 8 limbs (the n mod 8
    first, by n's bits in blocks of 1, 2 and 4) the sum along the flag,
    parked; then GMP's mpn_bdiv_dbm1c step with (B-1)/5."""
    name = "gtc_div5_add_n"
    prologue(name)
    emit("\tstp\tx19, x20, [sp, #-64]!")
    emit("\tstp\tx21, x22, [sp, #16]")
    emit("\tstp\tx23, x24, [sp, #32]")
    emit("\tstp\tx25, x26, [sp, #48]")
    U = ["x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
    L = ["x16", "x17"]
    bd, h, p0, p1 = (f"x{r}" for r in range(22, 26))
    labels = {v: lab(f"da{v}_") for v in (0, 1, 2)}
    end = lab("daend")
    emit(f"\tmov\t{bd}, #0x3333333333333333\t// (B-1)/5")
    emit(f"\tmov\t{h}, xzr")
    emit("\tcmp\tw4, #1")
    emit(f"\tb.eq\t{labels[1]}")
    emit(f"\tb.hi\t{labels[2]}")
    emit(f"\tb\t{labels[0]}")

    def block(nl, mode):
        g = Grp(nl)
        for i in range(0, nl, 2):
            g.acc("ld", U[i:i + min(2, nl - i)], "x1", i)
        emit("\tcmp\tx6, #1")
        for i in range(0, nl, 2):
            k = min(2, nl - i)
            g.acc("ld", L[:k], "x2", i)
            for j in range(k):
                if mode == 0:
                    emit(f"\tadcs\t{U[i + j]}, {U[i + j]}, {L[j]}")
                elif mode == 1:
                    emit(f"\tsbcs\t{U[i + j]}, {U[i + j]}, {L[j]}")
                else:
                    emit(f"\tsbcs\t{U[i + j]}, {L[j]}, {U[i + j]}")
        emit("\tcset\tx6, cs")
        for i in range(nl):
            dbm1_step(U[i], h, p0, p1, bd, False)
        for i in range(0, nl, 2):
            g.acc("st", U[i:i + min(2, nl - i)], "x0", i)

    for mode, l in labels.items():
        emit(f"{l}:")
        emit(f"\tmov\tx6, #{1 if mode else 0}\t\t// C = 1 where the chain subtracts")
        for b in (0, 1, 2):
            skip = lab("dar")
            emit(f"\ttbz\tx3, #{b}, {skip}")
            block(1 << b, mode)
            emit(f"{skip}:")
        grp, done = lab("dag"), lab("dad")
        emit("\tlsr\tx5, x3, #3")
        emit(f"\tcbz\tx5, {done}")
        emit("\t.p2align\t4")
        emit(f"{grp}:")
        block(8, mode)
        emit("\tsub\tx5, x5, #1")
        emit(f"\tcbnz\tx5, {grp}")
        emit(f"{done}:")
        emit(f"\tb\t{end}")
    emit(f"{end}:")
    emit(f"\tmov\tx0, {h}")
    emit("\tldp\tx21, x22, [sp, #16]")
    emit("\tldp\tx23, x24, [sp, #32]")
    emit("\tldp\tx25, x26, [sp, #48]")
    emit("\tldp\tx19, x20, [sp], #64")
    epilogue(name)


def addlsh123():
    """gtc_addlsh123_n: {rp, n} = a + 2 b + 4 c + 8 d, returning the limb
    above (the contract of the x86-64 kernel).  Per block of 8 limbs (the n
    mod 8 first, by n's bits in blocks of 1, 2 and 4): a loaded, then the
    chains of 2 b, 4 c and 8 d along the flag one after the other, each
    parked between blocks (cset, cmp), the shifted limbs by extr."""
    name = "gtc_addlsh123_n"
    prologue(name)
    emit("\tstp\tx19, x20, [sp, #-64]!")
    emit("\tstp\tx21, x22, [sp, #16]")
    emit("\tstp\tx23, x24, [sp, #32]")
    emit("\tstp\tx25, x26, [sp, #48]")
    U = ["x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
    L = ["x16", "x17"]
    pb, pc, pd, t0, t1, kd = "x19", "x20", "x21", "x22", "x23", "x24"
    for r in (pb, pc, pd, "x6", "x7", kd):
        emit(f"\tmov\t{r}, xzr")

    # the loop's count in x5 itself, x25 and x26 the chains' second load
    # pair
    cnt = "x5"

    def chain(g, nl, src, prev, sh, park):
        emit(f"\tcmp\t{park}, #1")
        lsh_chain(g, U, L, ("x25", "x26"), nl, src, prev, sh, "adcs", t0, t1)
        emit(f"\tcset\t{park}, cs")

    def block(nl):
        g = Grp(nl)
        for i in range(0, nl, 2):
            g.acc("ld", U[i:i + min(2, nl - i)], "x1", i)
        chain(g, nl, "x2", pb, 63, "x6")
        chain(g, nl, "x3", pc, 62, "x7")
        chain(g, nl, "x4", pd, 61, kd)
        for i in range(0, nl, 2):
            g.acc("st", U[i:i + min(2, nl - i)], "x0", i)

    for b in (0, 1, 2):
        skip = lab("lr")
        emit(f"\ttbz\tx5, #{b}, {skip}")
        block(1 << b)
        emit(f"{skip}:")
    grp, done = lab("lg"), lab("ld")
    emit(f"\tlsr\t{cnt}, x5, #3")
    emit(f"\tcbz\t{cnt}, {done}")
    emit("\t.p2align\t4")
    emit(f"{grp}:")
    block(8)
    emit(f"\tsub\t{cnt}, {cnt}, #1")
    emit(f"\tcbnz\t{cnt}, {grp}")
    emit(f"{done}:")
    emit(f"\tadd\tx0, x6, x7")
    emit(f"\tadd\tx0, x0, {kd}")
    emit(f"\tadd\tx0, x0, {pb}, lsr #63")
    emit(f"\tadd\tx0, x0, {pc}, lsr #62")
    emit(f"\tadd\tx0, x0, {pd}, lsr #61")
    emit("\tldp\tx21, x22, [sp, #16]")
    emit("\tldp\tx23, x24, [sp, #32]")
    emit("\tldp\tx25, x26, [sp, #48]")
    emit("\tldp\tx19, x20, [sp], #64")
    epilogue(name)

def t5r():
    """gtc_t5r_n: the Gaussian Toom-5's t5 pass (gtc.c, eval5_t5r) from its
    second limb, the contract of the x86-64 kernel: {d, m} = x0 - 4 x4 (or
    4 x4 - x0) and {dm, m} = 4 y - d, the shifted limbs by extr with the
    limb before (x4[-1] and y[-1] read first); x5 = m << 3 | neg << 2 | c
    << 1 | b, m a multiple of 4, m >= 4; returns b | c << 1.  Per block of
    4 limbs each chain of sbcs runs from its flag restored (cmp) to its
    flag parked (cset)."""
    name = "gtc_t5r_n"
    prologue(name)
    emit("\tstp\tx19, x20, [sp, #-48]!")
    emit("\tstp\tx21, x22, [sp, #16]")
    emit("\tstr\tx23, [sp, #32]")
    Z = ["x11", "x12", "x13", "x14"]
    F = ["x15", "x16", "x17", "x19"]
    A = ["x20", "x21", "x22", "x23"]
    emit("\tldur\tx8, [x3, #-8]\t\t// x4[-1]")
    emit("\tldur\tx9, [x4, #-8]\t\t// y[-1]")
    emit("\tlsr\tx10, x5, #5\t\t// m / 4 blocks")
    emit("\tand\tx6, x5, #1")
    emit("\teor\tx6, x6, #1\t\t// C of the first chain: 1 - b")
    emit("\tubfx\tx7, x5, #1, #1")
    emit("\teor\tx7, x7, #1\t\t// C of the second: 1 - c")
    neg_l, out_ = lab("t5n"), lab("t5o")
    emit(f"\ttbnz\tx5, #2, {neg_l}")
    for neg in (0, 1):
        if neg:
            emit(f"{neg_l}:")
        top = lab("t5l")
        emit("\t.p2align\t4")
        emit(f"{top}:")
        emit(f"\tldp\t{Z[0]}, {Z[1]}, [x3], #16")
        emit(f"\tldp\t{Z[2]}, {Z[3]}, [x3], #16")
        emit(f"\tldp\t{A[0]}, {A[1]}, [x2], #16")
        emit(f"\tldp\t{A[2]}, {A[3]}, [x2], #16")
        prev = "x8"
        for j in range(4):
            emit(f"\textr\t{F[j]}, {Z[j]}, {prev}, #62")
            prev = Z[j]
        emit(f"\tmov\tx8, {Z[3]}")
        emit("\tcmp\tx6, #1")
        for j in range(4):
            if neg:
                emit(f"\tsbcs\t{A[j]}, {F[j]}, {A[j]}")
            else:
                emit(f"\tsbcs\t{A[j]}, {A[j]}, {F[j]}")
        emit("\tcset\tx6, cs")
        emit(f"\tstp\t{A[0]}, {A[1]}, [x0], #16")
        emit(f"\tstp\t{A[2]}, {A[3]}, [x0], #16")
        emit(f"\tldp\t{Z[0]}, {Z[1]}, [x4], #16")
        emit(f"\tldp\t{Z[2]}, {Z[3]}, [x4], #16")
        prev = "x9"
        for j in range(4):
            emit(f"\textr\t{F[j]}, {Z[j]}, {prev}, #62")
            prev = Z[j]
        emit(f"\tmov\tx9, {Z[3]}")
        emit("\tcmp\tx7, #1")
        for j in range(4):
            emit(f"\tsbcs\t{F[j]}, {F[j]}, {A[j]}")
        emit("\tcset\tx7, cs")
        emit(f"\tstp\t{F[0]}, {F[1]}, [x1], #16")
        emit(f"\tstp\t{F[2]}, {F[3]}, [x1], #16")
        emit("\tsubs\tx10, x10, #1")
        emit(f"\tb.ne\t{top}")
        if not neg:
            emit(f"\tb\t{out_}")
    emit(f"{out_}:")
    emit("\teor\tx6, x6, #1\t\t// b")
    emit("\teor\tx7, x7, #1\t\t// c")
    emit("\tadd\tx0, x6, x7, lsl #1")
    emit("\tldr\tx23, [sp, #32]")
    emit("\tldp\tx21, x22, [sp, #16]")
    emit("\tldp\tx19, x20, [sp], #48")
    epilogue(name)


def div5_add_fused(kind):
    """gtc_div5_add_sub_n: gtc_div5_add_n's {rp, n} = q (mode & 3), and
    {sp, n} = z - q modulo B^n in the same pass, z = y with mode & 4, else
    x (the contract of the x86-64 kernel): per block a third chain of sbcs
    over z and the quotient limbs still in their registers, parked in x20,
    stored through x19 = sp.  rp and sp overlap neither xp nor yp nor each
    other."""
    # kind "acc": gtc_div5_add_acc_n, {ap, n} += q (mode & 4: -= q) along
    # the third chain through x19 = ap, returning its carry (or borrow);
    # rp may equal xp or yp there, ap overlaps none of them
    name = "gtc_div5_add_sub_n" if kind == "sub" else "gtc_div5_add_acc_n"
    prologue(name)
    emit("\tstp\tx19, x20, [sp, #-64]!")
    emit("\tstp\tx21, x22, [sp, #16]")
    emit("\tstp\tx23, x24, [sp, #32]")
    emit("\tstp\tx25, x26, [sp, #48]")
    U = ["x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
    L = ["x16", "x17"]
    bd, h, p0, p1 = (f"x{r}" for r in range(22, 26))
    labels = {(v, zy): lab(f"ds{v}_") for v in (0, 1, 2) for zy in (0, 1)}
    end = lab("dsend")
    emit("\tmov\tx19, x5\t\t// sp")
    if kind == "sub":
        emit("\tmov\tx20, #1\t\t// z - q's C: no borrow")
    emit(f"\tmov\t{bd}, #0x3333333333333333\t// (B-1)/5")
    emit(f"\tmov\t{h}, xzr")
    emit("\tand\tw7, w4, #3")
    zy1 = lab("dsz")
    emit(f"\ttbnz\tw4, #2, {zy1}")
    for zy in (0, 1):
        if zy:
            emit(f"{zy1}:")
        emit("\tcmp\tw7, #1")
        emit(f"\tb.eq\t{labels[(1, zy)]}")
        emit(f"\tb.hi\t{labels[(2, zy)]}")
        emit(f"\tb\t{labels[(0, zy)]}")

    def block(nl, mode, zy):
        g = Grp(nl)
        z = "x2" if zy else "x1"
        for i in range(0, nl, 2):
            g.acc("ld", U[i:i + min(2, nl - i)], "x1", i)
        emit("\tcmp\tx6, #1")
        for i in range(0, nl, 2):
            k = min(2, nl - i)
            g.acc("ld", L[:k], "x2", i)
            for j in range(k):
                if mode == 0:
                    emit(f"\tadcs\t{U[i + j]}, {U[i + j]}, {L[j]}")
                elif mode == 1:
                    emit(f"\tsbcs\t{U[i + j]}, {U[i + j]}, {L[j]}")
                else:
                    emit(f"\tsbcs\t{U[i + j]}, {L[j]}, {U[i + j]}")
        emit("\tcset\tx6, cs")
        for i in range(nl):
            dbm1_step(U[i], h, p0, p1, bd, False)
        for i in range(0, nl, 2):
            g.acc("st", U[i:i + min(2, nl - i)], "x0", i)
        emit("\tcmp\tx20, #1")
        for i in range(0, nl, 2):
            k = min(2, nl - i)
            if kind == "sub":
                g.acc("ld", L[:k], z, i)
                for j in range(k):
                    emit(f"\tsbcs\t{L[j]}, {L[j]}, {U[i + j]}")
            else:
                g.acc("ld", L[:k], "x19", i)
                for j in range(k):
                    emit(f"\t{'sbcs' if zy else 'adcs'}\t{L[j]}, {L[j]}, {U[i + j]}")
            g.acc("st", L[:k], "x19", i)
        emit("\tcset\tx20, cs")

    for (mode, zy), l in labels.items():
        emit(f"{l}:")
        emit(f"\tmov\tx6, #{1 if mode else 0}\t\t// C = 1 where the chain subtracts")
        if kind == "acc":
            emit(f"\tmov\tx20, #{zy}\t\t// the third chain's C likewise")
        for b in (0, 1, 2):
            skip = lab("dsr")
            emit(f"\ttbz\tx3, #{b}, {skip}")
            block(1 << b, mode, zy)
            emit(f"{skip}:")
        grp, done = lab("dsg"), lab("dsd")
        emit("\tlsr\tx5, x3, #3")
        emit(f"\tcbz\tx5, {done}")
        emit("\t.p2align\t4")
        emit(f"{grp}:")
        block(8, mode, zy)
        emit("\tsub\tx5, x5, #1")
        emit(f"\tcbnz\tx5, {grp}")
        emit(f"{done}:")
        emit(f"\tb\t{end}")
    emit(f"{end}:")
    if kind == "sub":
        emit(f"\tmov\tx0, {h}")
    else:
        # a carry is C; a borrow is 1 - C (the paths with zy subtract)
        emit("\tmov\tx0, x20")
        emit("\ttbz\tw4, #2, 1f")
        emit("\teor\tx0, x0, #1")
        emit("1:")
    emit("\tldp\tx21, x22, [sp, #16]")
    emit("\tldp\tx23, x24, [sp, #32]")
    emit("\tldp\tx25, x26, [sp, #48]")
    emit("\tldp\tx19, x20, [sp], #64")
    epilogue(name)


def main():
    emit("/* Copyright (c) 2026 Huawei Technologies Co., Ltd.")
    emit("   SPDX-License-Identifier: BSD-3-Clause-Clear  */")
    emit("/* Generated by tools/gen_linear_arm.py -- do not edit.")
    emit("   The linear kernels of the Gaussian products, aarch64:")
    emit("   gtc_aors2_n, gtc_add3_n, gtc_aors2x_n, gtc_rsh1add_n,")
    emit("   gtc_rsh1sub_n, gtc_kara_recomp2, gtc_add_n, gtc_sub_n,")
    emit("   gtc_rsh1add3_n, gtc_div15_lsh_n, gtc_addlsh123_n, gtc_div5_lsh_n,")
    emit("   gtc_rsh2add_n, gtc_rsh2sub_n, gtc_rsh2add3_n,")
    emit("   gtc_rsh3add3_n, gtc_div5_add_n, gtc_t5r_n, gtc_div5_add_sub_n,")
    emit("   gtc_div5_add_acc_n, gtc_rsh3add_n and gtc_rsh3sub_n.")
    emit("   See that file for what they compute and how.  */")
    emit("#if defined (__aarch64__) && defined (__ELF__) && !defined (GTC_NO_ASM)")
    emit("\t.text")
    # (the order is the measured layout: the kernels share the Kunpeng-920's
    # instruction cache with the textbook kernels linked after them)
    for f in (aors2, add3, aors2x, lambda: rsh1("gtc_rsh1add_n", False),
              lambda: rsh1("gtc_rsh1sub_n", True), kara_recomp2,
              lambda: addsub_n("gtc_add_n", False),
              lambda: addsub_n("gtc_sub_n", True), rsh1add3, div15_lsh, addlsh123,
              div5_lsh,
              lambda: rsh1("gtc_rsh2add_n", False, 2),
              lambda: rsh1("gtc_rsh2sub_n", True, 2),
              lambda: rsh1add3("gtc_rsh2add3_n", 2),
              lambda: rsh1add3("gtc_rsh3add3_n", 3), div5_add, t5r, lambda: div5_add_fused("sub"),
              lambda: div5_add_fused("acc"),
              lambda: rsh1("gtc_rsh3add_n", False, 3),
              lambda: rsh1("gtc_rsh3sub_n", True, 3)):
        emit("")
        f()
    emit('\t.section\t.note.GNU-stack,"",%progbits')
    emit("#endif")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
