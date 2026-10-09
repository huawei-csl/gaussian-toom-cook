#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Generate gtc_kara_arm.S: one level of the Gaussian Karatsuba, one
function per size, 4..64 limbs, aarch64.

  void gtc_gkaraN (mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
                   mp_srcptr br, mp_srcptr bi, const fn *tab)

computes P + iQ = (ar + i ai)(br + i bi) for N-limb components >= 0, P and Q
of 2N+1 limbs in two's complement, as gtc.c's gauss_toom22 does, where the
products of the halves (k = ceil(N/2) and s = floor(N/2) limbs) are
tab[k] and tab[s]: functions of the same form, the textbook kernels of
gtc_textbook_arm.S (which ignore tab) or these functions again:
  x = x0 + x1 X,  X = B^k,
  v0 = x0 y0,  vinf = x1 y1,  vm1 = (x0 - x1)(y0 - y1),
  x y = v0 + (v0 + vinf - vm1) X + vinf X^2.

The level of gen_kara.py (x86-64), with what aarch64 changes.  Everything
around the three calls is inline and unrolled for the size:

* The four differences |x0 - x1|, read through the argument registers:
  first the four compares from the top (each sets its sign and picks the
  minuend and subtrahend by csel), then the four subtractions, whose
  pointers are then ready long before their loads.
* The rotation of vm1's factors into the first quadrant, by csel.
* The calls through tab: v0 (the arguments still in their registers),
  vinf, vm1 into the frame.
* The recomposition, two passes per component: T = H0 + Li (v0 = L0 + H0 X
  + c0 X^2 and vinf = Li + Hi X + ...), its first limbs in registers, any
  others over Li; then over 2k limbs U = T + L0 | T + Hi, and U - vm1.
  There is one carry flag, so the two chains go by blocks of 8 limbs,
  each block's carry parked in a register (cset) and restored (cmp c, #1),
  as gtc_add3_n does.  -vm1 is added as vm1 xor a mask (0 or -1) with the
  mask's low bit as carry-in: eor sets no flags here, so the sign costs one
  eor a limb in the chain and no pass of its own.
* The carries and the top limbs go in last, as signed limbs whose
  propagation is a rare branch.

Each level has two entries: name_ns, which saves no callee-saved register
(the level keeps nothing in registers across its calls) and calls the _ns
entries of its halves through tab, and the ABI entry name, which saves
x19..x30 and calls name_ns.  See ctx.c's leaf_ns.

Run:  ./tools/gen_kara_arm.py > src/gtc_kara_arm.S  (or: make asm)"""
import sys

NMIN, NMAX = 4, 64
BLOCK = 8                   # limbs per block of the two chains

# Registers of the recomposition besides x0..x7: a block (U), two for
# vm1's limbs (V), the rest for T.  x18 is the platform register; x30 is
# saved in the frame.
POOL = [f"x{i}" for i in range(8, 18)] + [f"x{i}" for i in range(19, 31)]
U = POOL[:BLOCK]
V = POOL[BLOCK:BLOCK + 2]
TR = POOL[BLOCK + 2:]


def gen(n):
    k, s = (n + 1) // 2, n // 2
    name = f"gtc_gkara{n}"
    out, tail = [], []
    e = out.append
    labels = [0]

    # The frame: the slots, then the differences and vm1.
    slots, off = {}, 0
    for nm in ("lr", "tab", "P", "Q", "ar", "ai", "br", "bi", "vxr", "vxi",
               "vyr", "vyi", "srcP", "mP", "srcQ", "mQ", "p0", "q0"):
        slots[nm] = off
        off += 8
    for nm in ("dxr", "dxi", "dyr", "dyi"):
        slots[nm] = off
        off += 8 * k
    for nm in ("MP", "MQ"):
        slots[nm] = off
        off += 8 * (2 * k + 1)
    frame = (off + 15) // 16 * 16
    assert frame < 4096

    def mem(op, items):
        """Loads (op ld) or stores (st) of (reg, base, byte offset), two
        consecutive limbs on one base as ldp/stp."""
        i = 0
        while i < len(items):
            r, b, o = items[i]
            if i + 1 < len(items):
                r2, b2, o2 = items[i + 1]
                if (b2 == b and o2 == o + 8 and -512 <= o <= 504
                        and not (op == "ld" and r2 == r)):
                    e(f"\t{op}p\t{r}, {r2}, [{b}, #{o}]")
                    i += 2
                    continue
            e(f"\t{op}r\t{r}, [{b}, #{o}]")
            i += 1

    # The ABI entry: x19..x30 around the call of name_ns.
    pairs = (("x21", "x22"), ("x23", "x24"), ("x25", "x26"), ("x27", "x28"),
             ("x29", "x30"))
    e(f"\t.globl\t{name}")
    e(f"\t.type\t{name}, %function")
    e("\t.p2align\t4")
    e(f"{name}:")
    e("\tstp\tx19, x20, [sp, #-96]!")
    for j, (a, b) in enumerate(pairs):
        e(f"\tstp\t{a}, {b}, [sp, #{16 * (j + 1)}]")
    e(f"\tbl\t{name}_ns")
    for j, (a, b) in enumerate(pairs):
        e(f"\tldp\t{a}, {b}, [sp, #{16 * (j + 1)}]")
    e("\tldp\tx19, x20, [sp], #96")
    e("\tret")
    e(f"\t.size\t{name}, .-{name}")

    e(f"\t.globl\t{name}_ns")
    e(f"\t.type\t{name}_ns, %function")
    e("\t.p2align\t5")
    e(f"{name}_ns:")
    e(f"\tsub\tsp, sp, #{frame}")
    e(f"\tstp\tx30, x6, [sp, #{slots['lr']}]")
    e(f"\tstp\tx0, x1, [sp, #{slots['P']}]")
    e(f"\tstp\tx2, x3, [sp, #{slots['ar']}]")
    e(f"\tstp\tx4, x5, [sp, #{slots['br']}]")

    # Evaluation at -1: |x0 - x1| and its sign, x1 of s limbs (a zero top
    # limb when s < k), read through the argument registers.
    def compare(R, sign, pm, ps):
        """The sign of x0 - x1 to sign, the pointers to the minuend and the
        subtrahend to pm and ps."""
        labels[0] += 1
        done = f".Ld{n}_{labels[0]}"
        top = k - 1
        if s < k:
            e(f"\tldr\tx15, [{R}, #{8 * (k - 1)}]")
            e("\tcmp\tx15, #0")
            e(f"\tb.ne\t{done}")
            top = k - 2
        for j in range(top, -1, -1):
            e(f"\tldr\tx15, [{R}, #{8 * j}]")
            e(f"\tldr\tx16, [{R}, #{8 * (k + j)}]")
            e("\tcmp\tx15, x16")
            if j > 0:
                e(f"\tb.ne\t{done}")
        e(f"{done}:")
        # C clear: x0 < x1 at the deciding limb (set if all are equal).
        e(f"\tcset\t{sign}, lo")
        e(f"\tadd\tx17, {R}, #{8 * k}")
        e(f"\tcsel\t{pm}, x17, {R}, lo")
        e(f"\tcsel\t{ps}, {R}, x17, lo")

    def subtract(R, dst, pm, ps):
        e(f"\tadd\tx27, sp, #{slots[dst]}")
        first = True
        for j in range(0, s, 2):
            m = min(2, s - j)
            a = ["x23", "x24"][:m]
            b = ["x25", "x26"][:m]
            mem("ld", [(r, pm, 8 * (j + i)) for i, r in enumerate(a)])
            mem("ld", [(r, ps, 8 * (j + i)) for i, r in enumerate(b)])
            for x, y in zip(a, b):
                e(f"\t{'subs' if first else 'sbcs'}\t{x}, {x}, {y}")
                first = False
            mem("st", [(r, "x27", 8 * (j + i)) for i, r in enumerate(a)])
        if s < k:
            # x1's missing limb is 0, and x0's top limb is 0 when x0 < x1:
            # the top limb of the difference is x0's minus the borrow.
            e(f"\tldr\tx23, [{R}, #{8 * (k - 1)}]")
            e("\tsbc\tx23, x23, xzr")
            e(f"\tstr\tx23, [x27, #{8 * (k - 1)}]")

    ops = (("x2", "dxr", "x19", "x7", "x8"), ("x3", "dxi", "x20", "x9", "x10"),
           ("x4", "dyr", "x21", "x11", "x12"),
           ("x5", "dyi", "x22", "x13", "x14"))
    for R, dst, sg, pm, ps in ops:
        compare(R, sg, pm, ps)
    for R, dst, sg, pm, ps in ops:
        subtract(R, dst, pm, ps)

    # vm1: each factor d = (Dr + i Di) with signs is (Ar + i Ai) i^e with
    # Ar, Ai >= 0: e = (sr ^ si) + 2 si, and (Ar, Ai) = (Di, Dr) for odd e.
    # vm1 = (MP + i MQ) i^e, and -vm1 adds to P: -MP, +MQ, +MP, -MQ and to
    # Q: -MQ, -MP, +MQ, +MP for e = 0, 1, 2, 3: P's source is MP for even
    # e, MQ for odd, subtracted for e = 0, 3; Q's the other, subtracted for
    # e = 0, 1.  A subtraction is the addition of the complement (eor with
    # the mask -1) with carry-in 1.
    e("\teor\tx23, x19, x20")
    e("\teor\tx24, x21, x22")
    e("\tadd\tx25, x23, x20, lsl #1")
    e("\tadd\tx25, x25, x24")
    e("\tadd\tx25, x25, x22, lsl #1")
    e("\tand\tx25, x25, #3")            # e
    for a, b, t, sl in (("dxr", "dxi", "x23", "vxr"),
                        ("dyr", "dyi", "x24", "vyr")):
        e(f"\tadd\tx7, sp, #{slots[a]}")
        e(f"\tadd\tx8, sp, #{slots[b]}")
        e(f"\tcmp\t{t}, #0")
        e("\tcsel\tx9, x8, x7, ne")
        e("\tcsel\tx10, x7, x8, ne")
        e(f"\tstp\tx9, x10, [sp, #{slots[sl]}]")
    e(f"\tadd\tx7, sp, #{slots['MP']}")
    e(f"\tadd\tx8, sp, #{slots['MQ']}")
    e("\ttst\tx25, #1")
    e("\tcsel\tx9, x8, x7, ne")          # srcP
    e("\tcsel\tx11, x7, x8, ne")         # srcQ
    e("\tadd\tx12, x25, #1")
    e("\tand\tx12, x12, #3")
    e("\tcmp\tx12, #2")
    e("\tcsetm\tx10, lo")                # mP: e = 3 or 0
    e("\tcmp\tx25, #2")
    e("\tcsetm\tx12, lo")                # mQ: e = 0 or 1
    e(f"\tstp\tx9, x10, [sp, #{slots['srcP']}]")
    e(f"\tstp\tx11, x12, [sp, #{slots['srcQ']}]")

    # v0 in place (the arguments are still in their registers), then its
    # top limbs saved; vinf over them; vm1 into the frame.
    e(f"\tldr\tx7, [x6, #{8 * k}]")
    e("\tblr\tx7")
    e(f"\tldp\tx0, x1, [sp, #{slots['P']}]")
    e(f"\tldr\tx7, [x0, #{16 * k}]")
    e(f"\tldr\tx8, [x1, #{16 * k}]")
    e(f"\tstp\tx7, x8, [sp, #{slots['p0']}]")
    e(f"\tadd\tx0, x0, #{16 * k}")
    e(f"\tadd\tx1, x1, #{16 * k}")
    e(f"\tldp\tx2, x3, [sp, #{slots['ar']}]")
    e(f"\tldp\tx4, x5, [sp, #{slots['br']}]")
    e(f"\tldr\tx6, [sp, #{slots['tab']}]")
    for r in ("x2", "x3", "x4", "x5"):
        e(f"\tadd\t{r}, {r}, #{8 * k}")
    e(f"\tldr\tx7, [x6, #{8 * s}]")
    e("\tblr\tx7")
    e(f"\tadd\tx0, sp, #{slots['MP']}")
    e(f"\tadd\tx1, sp, #{slots['MQ']}")
    e(f"\tldp\tx2, x3, [sp, #{slots['vxr']}]")
    e(f"\tldp\tx4, x5, [sp, #{slots['vyr']}]")
    e(f"\tldr\tx6, [sp, #{slots['tab']}]")
    e(f"\tldr\tx7, [x6, #{8 * k}]")
    e("\tblr\tx7")

    def acc_signed(pos, v):
        """{x0 + 8 pos, 2n+1 - pos} += v, a signed limb in register v.
        Uses x13..x16."""
        labels[0] += 1
        prop, back = f".Lp{n}_{labels[0]}", f".Lb{n}_{labels[0]}"
        e(f"\tldr\tx13, [x0, #{8 * pos}]")
        e(f"\tadds\tx13, x13, {v}")
        e(f"\tstr\tx13, [x0, #{8 * pos}]")
        e(f"\tasr\tx14, {v}, #63")
        e("\tadc\tx14, x14, xzr")         # net carry: -1, 0 or 1
        e(f"\tcbnz\tx14, {prop}")
        e(f"{back}:")
        rest = 2 * n - pos
        tail.append(f"{prop}:")
        if rest == 0:
            tail.append(f"\tb\t{back}")
            return
        up, dn = f".Lu{n}_{labels[0]}", f".Ln{n}_{labels[0]}"
        tail.extend((f"\tadd\tx15, x0, #{8 * (pos + 1)}",
                     f"\tmov\tx16, #{rest}",
                     f"\ttbnz\tx14, #63, {dn}",
                     f"{up}:",
                     "\tldr\tx13, [x15]",
                     "\tadds\tx13, x13, #1",
                     "\tstr\tx13, [x15], #8",
                     f"\tb.cc\t{back}",
                     "\tsubs\tx16, x16, #1",
                     f"\tb.ne\t{up}",
                     f"\tb\t{back}",
                     f"{dn}:",
                     "\tldr\tx13, [x15]",
                     "\tsubs\tx13, x13, #1",
                     "\tstr\tx13, [x15], #8",
                     f"\tb.cs\t{back}",
                     "\tsubs\tx16, x16, #1",
                     f"\tb.ne\t{dn}",
                     f"\tb\t{back}"))

    nr = min(k, len(TR))                # limbs of T in registers
    m2 = 2 * s - k                      # limbs of Hi

    def component(c, top, src):
        """Recompose component c: x0 = C, x1 = vm1's source, x2 = its mask,
        x3 = cT, x4 = C + 8k (H0, then X^1), x5 = C + 16k (Li, T's limbs
        past nr, then X^2), x6 and x7 the parked carries of the chains."""
        e(f"\tldr\tx0, [sp, #{slots[c]}]")
        e(f"\tldp\tx1, x2, [sp, #{slots[src]}]")
        e(f"\tadd\tx4, x0, #{8 * k}")
        e(f"\tadd\tx5, x0, #{16 * k}")
        # T = H0 + Li.
        first = True
        for j in range(0, k, 2):
            js = list(range(j, min(j + 2, k)))
            dst = [TR[i] if i < nr else U[i - j] for i in js]
            mem("ld", [(r, "x4", 8 * i) for r, i in zip(dst, js)])
            mem("ld", [(V[i - j], "x5", 8 * i) for i in js])
            for r, i in zip(dst, js):
                e(f"\t{'adds' if first else 'adcs'}\t{r}, {r}, {V[i - j]}")
                first = False
            mem("st", [(r, "x5", 8 * i) for r, i in zip(dst, js) if i >= nr])
        e("\tcset\tx3, cs")
        # X^1: T + L0 - vm1_lo;  X^2: T + Hi - vm1_hi, Hi of m2 limbs.
        for b0 in range(0, 2 * k, BLOCK):
            blk = list(range(b0, min(b0 + BLOCK, 2 * k)))
            # Chain A: U = T + (L0 | Hi | 0), by pairs, loads then adds.
            if b0:
                e("\tcmp\tx6, #1")
            terms = []
            for i, j in enumerate(blk):
                jj = j % k
                other = (("x0", 8 * j) if j < k else
                         ("x5", 8 * (k + jj)) if jj < m2 else None)
                terms.append((i, jj, other))
            fst = b0 == 0
            for p in range(0, len(terms), 2):
                pr = terms[p:p + 2]
                mem("ld", [(U[i], "x5", 8 * jj) for i, jj, o in pr
                           if jj >= nr])
                mem("ld", [((U[i] if jj < nr else V[q]),) + o
                           for q, (i, jj, o) in enumerate(pr)
                           if o is not None])
                for q, (i, jj, o) in enumerate(pr):
                    if jj < nr:
                        t, x = TR[jj], U[i] if o is not None else "xzr"
                    else:
                        t, x = U[i], V[q] if o is not None else "xzr"
                    e(f"\t{'adds' if fst else 'adcs'}\t{U[i]}, {t}, {x}")
                    fst = False
            e("\tcset\tx6, cs")
            # Chain B: U += vm1 ^ mask, carry-in the mask's low bit.
            e("\tcmp\tx7, #1" if b0 else "\tcmn\tx2, x2")
            for p in range(0, len(blk), 2):
                pj = blk[p:p + 2]
                mem("ld", [(V[q], "x1", 8 * j) for q, j in enumerate(pj)])
                for q in range(len(pj)):
                    e(f"\teor\t{V[q]}, {V[q]}, x2")
                for q, j in enumerate(pj):
                    e(f"\tadcs\t{U[j - b0]}, {U[j - b0]}, {V[q]}")
            e("\tcset\tx7, cs")
            mem("st", [(U[i], "x4", 8 * j) if j < k else
                       (U[i], "x5", 8 * (j - k)) for i, j in enumerate(blk)])
        # At X^2: cT + c0.  At X^3: cT + the chains' carries + c0 + the top
        # limb of -+vm1 (+ ti when s = k).  At B^(k+2s): ti.
        e(f"\tldr\tx8, [x1, #{16 * k}]")
        e("\teor\tx8, x8, x2")
        e(f"\tldr\tx9, [sp, #{slots[top]}]")
        e(f"\tldr\tx12, [x0, #{8 * (2 * k + 2 * s)}]")      # ti
        e("\tadd\tx10, x6, x7")
        e("\tadd\tx10, x10, x8")
        e("\tadd\tx10, x10, x9")
        e("\tadd\tx10, x10, x3")
        if s == k:
            e("\tadd\tx10, x10, x12")
        e("\tadd\tx11, x9, x3")
        acc_signed(2 * k, "x11")
        acc_signed(3 * k, "x10")
        if s != k:
            acc_signed(k + 2 * s, "x12")

    component("P", "p0", "srcP")
    component("Q", "q0", "srcQ")

    e(f"\tldr\tx30, [sp, #{slots['lr']}]")
    e(f"\tadd\tsp, sp, #{frame}")
    e("\tret")
    out.extend(tail)
    e(f"\t.size\t{name}_ns, .-{name}_ns")
    return "\n".join(out)


def main():
    print("/* Copyright (c) 2026 Huawei Technologies Co., Ltd.")
    print("   SPDX-License-Identifier: BSD-3-Clause-Clear  */")
    print("/* Generated by tools/gen_kara_arm.py -- do not edit.")
    print("   One level of the Gaussian Karatsuba for %d..%d limbs, aarch64."
          % (NMIN, NMAX))
    print("   See that file for what these compute and how.  */")
    print("#if defined (__aarch64__) && defined (__ELF__) "
          "&& !defined (GTC_NO_ASM)")
    print("\t.text")
    for n in range(NMIN, NMAX + 1):
        print()
        print(gen(n))
    print('\t.section\t.note.GNU-stack,"",%progbits')
    print("#endif")


if __name__ == "__main__":
    sys.exit(main())
