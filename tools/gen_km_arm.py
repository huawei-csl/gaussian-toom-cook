#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Generate gtc_km_arm.S: the Gaussian Karatsuba over {1, i} -- the
three-multiplication identity -- as a base case for 2..10 limbs, aarch64.

  void gtc_gmul_kmN (mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
                     mp_srcptr br, mp_srcptr bi, const void *ctx)

has the contract of the textbook kernels (gen_textbook_arm.py): P + iQ =
(ar + i ai)(br + i bi) for N-limb components >= 0, P = ar br - ai bi in
two's complement and Q = ar bi + ai br, 2N+1 limbs each.  It is a separate
algorithm under its own names; the library chooses it from gtc_km_thresh
on, below the Gaussian Karatsuba (gtc22_thresh).  In Winograd's form,

  k1 = ar (br + bi),  k2 = bi (ar + ai),  k3 = br (ai - ar),
  P = k1 - k2,  Q = k1 + k3:

3 n^2 limb products where the textbook kernel makes 4 n^2.

On a Kunpeng-920 a limb product is two cycles of the multiplier (mul and
umulh do not dual-issue), and the textbook kernels leave the other pipes
half idle.  So the identity's linear work is cheap where it rides in the
rows, between the multiplies, and costly where it is a pass of its own,
which leaves the multiplier idle.  Hence three operand scans as
gen_textbook_arm.py's rows (a window of n+2 registers, the vector in n
registers), each limb taken care of as it becomes final:

  k1: the vector T' = br + bi (carry tb) is formed in registers as it is
      loaded; row i (multiplier ar_i) adds tb ar_i at limb i + n (in the
      low chain); limb i goes to the frame (K1), and in row i S_i = ar_i +
      ai_i and D_i = ai_i - ar_i (borrow b), two chains parked in
      registers (cset, and cmp c, #1 to restore);
  k2: multiplier bi_i, vector S' (carry sa): row i adds sa bi_i at limb i
      + n; limb i gives P_i = K1_i - k2_i, to P;
  k3: multiplier D'_i, vector br: br D' = br (ai - ar) + b br B^n; limb i
      gives Q_i = K1_i + k3_i, to Q; the tail subtracts b br at limb n (br
      is still the vector) and finishes Q.

Each kernel has the textbook kernels' two entries: name_ns saves no
callee-saved register (x30 goes to the frame), for the Karatsuba levels of
gen_kara_arm.py; name saves the callee-saved registers the kernel uses and
calls name_ns.

The registers:
- x6..x13: the row multiplier, two products, the row's last high half, a
  scratch limb, two parked carries, and an output pointer.
- The vector and the window: from x14..x17, x19..x30 and x0..x5, which is
  22 registers and sets the 10-limb limit.
- The rest, up to 9 limbs: pointers and masks.
The other pointers, and at 10 limbs these too, go to the frame.

Run:  ./tools/gen_km_arm.py > src/gtc_km_arm.S  (or: make asm)
"""
NMIN, NMAX = 2, 10
REGS = [f"x{i}" for i in range(0, 18)] + [f"x{i}" for i in range(19, 31)]
CALLEE = [f"x{i}" for i in range(19, 30)]


def gen_w(n):
    """The kernel for n-limb components; see the top."""
    c = m = n
    name = f"gtc_gmul_km{n}"
    out = []
    e = out.append
    slots, off = {}, 0
    for k in ("lr", "pad", "P", "Q", "br", "bi", "ar", "ai", "sam", "tbm",
              "bm", "pad2"):
        slots[k] = off
        off += 8
    for k, w in (("S", n), ("D", n), ("K1", 2 * n + 1)):
        off = (off + 15) // 16 * 16
        slots[k] = off
        off += 8 * w
    frame = (off + 15) // 16 * 16
    assert frame < 4096

    def fr(k, j=0):
        return f"[sp, #{slots[k] + 8 * j}]"

    mul, t0, t1, h, tq, cq, cp, pp = REGS[6:14]
    rest = REGS[14:] + REGS[:6]
    VEC = rest[:c]
    WIN = rest[c:c + c + 2]
    assert len(WIN) == c + 2, f"km{n}: registers"
    SPARE = rest[2 * c + 2:]
    # Up to 9 limbs: a pointer (ai in k1, bi in k2) and a mask (-tb in k1,
    # -sa in k2, -b in k3's tail); at 10 limbs both from the frame.
    rp_ = SPARE[0] if len(SPARE) >= 1 else None
    rm_ = SPARE[1] if len(SPARE) >= 2 else None

    used = set(REGS[6:14]) | set(VEC) | set(WIN) | {rp_, rm_} - {None}
    pairs = [r for r in CALLEE if r in used] + ["x30"]
    if len(pairs) % 2:
        pairs.append("xzr")
    sz = 8 * len(pairs)
    e(f"\t.globl\t{name}")
    e(f"\t.type\t{name}, %function")
    e("\t.p2align\t4")
    e(f"{name}:")
    e(f"\tstp\t{pairs[0]}, {pairs[1]}, [sp, #-{sz}]!")
    for j in range(2, len(pairs), 2):
        e(f"\tstp\t{pairs[j]}, {pairs[j+1]}, [sp, #{8 * j}]")
    e(f"\tbl\t{name}_ns")
    for j in range(2, len(pairs), 2):
        a, b = pairs[j], pairs[j + 1]
        if b == "xzr":
            e(f"\tldr\t{a}, [sp, #{8 * j}]")
        else:
            e(f"\tldp\t{a}, {b}, [sp, #{8 * j}]")
    e(f"\tldp\t{pairs[0]}, {pairs[1]}, [sp], #{sz}")
    e("\tret")
    e(f"\t.size\t{name}, .-{name}")

    e(f"\t.globl\t{name}_ns")
    e(f"\t.type\t{name}_ns, %function")
    e("\t.p2align\t4")
    e(f"{name}_ns:")
    e(f"\tsub\tsp, sp, #{frame}")
    e(f"\tstr\tx30, {fr('lr')}")
    e(f"\tstp\tx0, x1, {fr('P')}")
    e(f"\tstp\tx4, x5, {fr('br')}")
    e(f"\tstp\tx2, x3, {fr('ar')}")

    def scan(mult, vec_load, sink, tail, ext=None):
        """As gen's scan without the third chain: row i adds mult(i) times
        the vector at limb i, and ext(i, t) one more term at limb i + c in
        the low chain."""
        W = {k: WIN[k] for k in range(c + 2)}
        fresh = set(range(c + 2))
        vec_load()
        for i in range(m):
            mult(i)
            ts = (t0, t1)
            if all((i + j) in fresh for j in range(c)):
                for j in range(c):
                    e(f"\tmul\t{W[i+j]}, {mul}, {VEC[j]}")
                e(f"\tumulh\t{h}, {mul}, {VEC[c-1]}")
                fresh.difference_update(range(i, i + c))
                if ext:
                    assert (i + c) in fresh
                    ext(i, W[i + c])
                    fresh.discard(i + c)
            else:
                for j in range(c):
                    e(f"\tmul\t{ts[j & 1]}, {mul}, {VEC[j]}")
                    if j == c - 1:
                        e(f"\tumulh\t{h}, {mul}, {VEC[j]}")
                    e(f"\t{'adds' if j == 0 else 'adcs'}\t{W[i+j]}, "
                      f"{W[i+j]}, {ts[j & 1]}")
                if ext:
                    t = ts[c & 1]
                    ext(i, t)
                    assert (i + c) not in fresh
                    e(f"\tadcs\t{W[i+c]}, {W[i+c]}, {t}")
                    k = i + c + 1
                    if k in fresh:
                        e(f"\tadc\t{W[k]}, xzr, xzr")
                        fresh.discard(k)
                    else:
                        e(f"\tadc\t{W[k]}, {W[k]}, xzr")
                else:
                    e(f"\tadc\t{h}, {h}, xzr")
            for j in range(c):
                if j < c - 1:
                    e(f"\tumulh\t{ts[j & 1]}, {mul}, {VEC[j]}")
                    t = ts[j & 1]
                else:
                    t = h
                d = W[i + 1 + j]
                src = "xzr" if (i + 1 + j) in fresh else d
                fresh.discard(i + 1 + j)
                e(f"\t{'adds' if j == 0 else 'adcs'}\t{d}, {src}, {t}")
            k = i + c + 1
            if k in fresh:
                e(f"\tadc\t{W[k]}, xzr, xzr")
                fresh.discard(k)
            else:
                e(f"\tadc\t{W[k]}, {W[k]}, xzr")
            sink(i, W[i])
            r = W.pop(i)
            if i + c + 2 <= m + c + 1:
                W[i + c + 2] = r
                fresh.add(i + c + 2)
        tail(W)

    def mask_and(t, slot, reg):
        """t = mul & mask, the mask in reg or the frame slot."""
        if reg:
            e(f"\tand\t{t}, {mul}, {reg}")
        else:
            e(f"\tldr\t{t}, {fr(slot)}")
            e(f"\tand\t{t}, {mul}, {t}")

    # k1 = ar T, T = br + bi = T' + tb B^n formed in the vector registers.
    def vec1():
        e(f"\tldr\t{t0}, {fr('br')}")
        e(f"\tldr\t{t1}, {fr('bi')}")
        for j in range(c):
            e(f"\tldr\t{VEC[j]}, [{t0}, #{8*j}]")
            e(f"\tldr\t{tq}, [{t1}, #{8*j}]")
            e(f"\t{'adds' if j == 0 else 'adcs'}\t{VEC[j]}, {VEC[j]}, {tq}")
        e(f"\tcset\t{tq}, cs")
        e(f"\tneg\t{tq}, {tq}")
        e(f"\tstr\t{tq}, {fr('tbm')}")
        if rm_:
            e(f"\tmov\t{rm_}, {tq}")
    e(f"\tldr\t{pp}, {fr('ar')}")
    if rp_:
        e(f"\tldr\t{rp_}, {fr('ai')}")
    e(f"\tmov\t{cp}, #0")		# S's carry
    e(f"\tmov\t{cq}, #1")		# D's borrow, 1 = none

    def mult1(i):
        e(f"\tldr\t{mul}, [{pp}, #{8*i}]")

    def ext1(i, t):		# tb ar_i
        mask_and(t, "tbm", rm_)

    def sink1(i, reg):
        e(f"\tstr\t{reg}, {fr('K1', i)}")
        if rp_:
            e(f"\tldr\t{tq}, [{rp_}, #{8*i}]")
        else:
            e(f"\tldr\t{tq}, {fr('ai')}")
            e(f"\tldr\t{tq}, [{tq}, #{8*i}]")
        e(f"\tcmp\t{cp}, #1")
        e(f"\tadcs\t{t0}, {mul}, {tq}")
        e(f"\tcset\t{cp}, cs")
        e(f"\tstr\t{t0}, {fr('S', i)}")
        e(f"\tcmp\t{cq}, #1")
        e(f"\tsbcs\t{t1}, {tq}, {mul}")
        e(f"\tcset\t{cq}, cs")
        e(f"\tstr\t{t1}, {fr('D', i)}")

    def tail1(W):
        for k in range(m, 2 * n + 1):
            e(f"\tstr\t{W[k]}, {fr('K1', k)}")
    scan(mult1, vec1, sink1, tail1, ext1)

    # the masks -sa and -b
    e(f"\tneg\t{t0}, {cp}")
    e(f"\tsub\t{t1}, {cq}, #1")
    e(f"\tstr\t{t0}, {fr('sam')}")
    e(f"\tstr\t{t1}, {fr('bm')}")

    # k2 = bi S: P = K1 - k2, the borrow parked in cp (1 = none).
    def vec_frame(slot):
        def f():
            for j in range(c):
                e(f"\tldr\t{VEC[j]}, {fr(slot, j)}")
        return f
    if rp_:
        e(f"\tldr\t{rp_}, {fr('bi')}")
    if rm_:
        e(f"\tmov\t{rm_}, {t0}")
    e(f"\tldr\t{pp}, {fr('P')}")
    e(f"\tmov\t{cp}, #1")

    def mult2(i):
        if rp_:
            e(f"\tldr\t{mul}, [{rp_}, #{8*i}]")
        else:
            e(f"\tldr\t{mul}, {fr('bi')}")
            e(f"\tldr\t{mul}, [{mul}, #{8*i}]")

    def ext2(i, t):		# sa bi_i
        mask_and(t, "sam", rm_)

    def sink2(i, reg):
        e(f"\tldr\t{tq}, {fr('K1', i)}")
        e(f"\tcmp\t{cp}, #1")
        e(f"\tsbcs\t{t0}, {tq}, {reg}")
        e(f"\tcset\t{cp}, cs")
        e(f"\tstr\t{t0}, [{pp}, #{8*i}]")

    def tail2(W):
        e(f"\tcmp\t{cp}, #1")
        for k in range(m, 2 * n + 1):
            e(f"\tldr\t{tq}, {fr('K1', k)}")
            e(f"\tsbcs\t{t0}, {tq}, {W[k]}")
            e(f"\tstr\t{t0}, [{pp}, #{8*k}]")
    scan(mult2, vec_frame("S"), sink2, tail2, ext2)

    # k3 = br D': Q = K1 + k3 - b br B^n, the carry parked in cp.
    def vec3():
        base = VEC[c - 1]
        e(f"\tldr\t{base}, {fr('br')}")
        for j in range(c):
            e(f"\tldr\t{VEC[j]}, [{base}, #{8*j}]")
    e(f"\tldr\t{pp}, {fr('Q')}")
    e(f"\tmov\t{cp}, #0")

    def mult3(i):
        e(f"\tldr\t{mul}, {fr('D', i)}")

    def sink3(i, reg):
        e(f"\tldr\t{tq}, {fr('K1', i)}")
        e(f"\tcmp\t{cp}, #1")
        e(f"\tadcs\t{reg}, {reg}, {tq}")
        e(f"\tcset\t{cp}, cs")
        e(f"\tstr\t{reg}, [{pp}, #{8*i}]")

    def tail3(W):
        # - b br at limb n: br is the vector
        if rm_:
            bmr = rm_
            e(f"\tldr\t{bmr}, {fr('bm')}")
        else:
            bmr = tq
            e(f"\tldr\t{bmr}, {fr('bm')}")
        for j in range(c):
            e(f"\tand\t{t0}, {VEC[j]}, {bmr}")
            e(f"\t{'subs' if j == 0 else 'sbcs'}\t{W[m+j]}, {W[m+j]}, {t0}")
        e(f"\tsbc\t{W[2*n]}, {W[2*n]}, xzr")
        e(f"\tcmp\t{cp}, #1")
        for k in range(m, 2 * n + 1):
            e(f"\tldr\t{tq}, {fr('K1', k)}")
            e(f"\tadcs\t{W[k]}, {W[k]}, {tq}")
            e(f"\tstr\t{W[k]}, [{pp}, #{8*k}]")
    scan(mult3, vec3, sink3, tail3)

    e(f"\tldr\tx30, {fr('lr')}")
    e(f"\tadd\tsp, sp, #{frame}")
    e("\tret")
    e(f"\t.size\t{name}_ns, .-{name}_ns")
    return "\n".join(out)

def main():
    print("/* Copyright (c) 2026 Huawei Technologies Co., Ltd.")
    print("   SPDX-License-Identifier: BSD-3-Clause-Clear  */")
    print("/* Generated by tools/gen_km_arm.py -- do not edit.")
    print("   The Gaussian Karatsuba over {1, i} for %d..%d limbs, aarch64,"
          % (NMIN, NMAX))
    print("   in Winograd's form.")
    print("   See that file for what these compute and how.  */")
    print("#if defined (__aarch64__) && defined (__ELF__) "
          "&& !defined (GTC_NO_ASM)")
    print("\t.text")
    for n in range(NMIN, NMAX + 1):
        print()
        print(gen_w(n))
    print('\t.section\t.note.GNU-stack,"",%progbits')
    print("#endif")


if __name__ == "__main__":
    main()
