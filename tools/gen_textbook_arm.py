#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Generate gtc_textbook_arm.S: the textbook Gaussian product for 2..20
limbs, aarch64.

  void gtc_gmul_tbN (mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
                     mp_srcptr br, mp_srcptr bi, const void *ctx)

computes P + iQ = (ar + i ai)(br + i bi) for N-limb components >= 0:
P = ar br - ai bi in two's complement and Q = ar bi + ai br, 2N+1 limbs
each.  The same contract, and the same algebra, as gen_textbook.py's x86-64
kernels: each component is a sum of two products, so that it needs
additions only,

  P = ar br + (~ai) bi + bi - bi B^N,   since (~ai) bi = (B^N - 1 - ai) bi,
  Q = ar bi + ai br,

by operand scanning.  Row i adds m * w at limb i, where m is a limb of a
row multiplier (ar, ~ai or ai) and w a C-limb vector; the rows go in pairs,
ar_i and then (~)ai_i, into a window of C+2 limbs held in registers, and
after pair i the window's lowest limb is final and is stored.

Where this differs from the x86-64 kernels is the carry.  There, a partial
product's two halves ride two chains at once, CF (adcx) and OF (adox), and
a limb product costs one mulx and two flag adds.  aarch64 has one carry
flag, so the two halves cannot ride side by side; but a 64x64 product is
two instructions here anyway (mul and umulh), and neither touches the
flags.  So each row is two chains one after the other -- the low halves,
then the high halves one limb up -- with the multiplies of the next
partial products interleaved between the adds, where they are free.  A
limb product costs six instructions (two loads, mul, umulh, two adds)
against x86-64's three, but they issue over the two cycles the multiplier
needs anyway: mul and umulh do not dual-issue on a Kunpeng-920, so a
64x64 product is 2 cycles there whatever else is going on.

The window is C+2 registers and aarch64 has 31, so every size up to 20
limbs fits in one window: no column blocking, unlike x86-64 from 11 limbs.

As on x86-64, the pointers stay in their argument registers (the rows
loaded their multiplier's pointer from the frame first, a load more and a
wait per row); the window is written, not cleared and added to, where it
holds nothing yet (Q's first row is mul alone, and each pair's first row
writes its top limb); and the low chain's carry goes into the row's last
high half (at most B - 2), formed early, rather than into the window.
Pair loads (ldp) are slower inside products on a Kunpeng-920: the vectors
come at any 8-byte alignment, and misaligned ldp cost more.

Each kernel has two entries, as on x86-64: name_ns saves no callee-saved
register (only the link register, where the window takes it), for the
Karatsuba levels of gen_kara_arm.py, which keep nothing in registers
across their calls; the ABI entry name saves the callee-saved registers
the kernel uses, by pairs, and calls name_ns (where it uses none, the two
are one).

Run:  ./tools/gen_textbook_arm.py > src/gtc_textbook_arm.S  (or: make asm)
"""
import sys

NMAX = 20

# x18 is the platform register and is left alone; x29 and x30 are the frame
# pointer and the link register, usable once saved.
CALLER = [f"x{i}" for i in range(0, 18)]
CALLEE = [f"x{i}" for i in range(19, 29)] + ["x29", "x30"]
ARGS = {"P": "x0", "Q": "x1", "ar": "x2", "ai": "x3", "br": "x4", "bi": "x5"}


def gen(n):
    """n-limb components, one window of n+2 limbs."""
    m = c = n
    name = f"gtc_gmul_tb{n}"
    out = []
    emit = out.append

    # The pointers stay in the registers they come in: x0 (P, the output),
    # x4 and x5 (the vectors br and bi), and while registers last x2 and x3
    # (the row multipliers ar and ai, which the rows would otherwise reload
    # from the frame one by one) and x1 (Q).  Then the row multiplier, two
    # products in flight and the two limbs they were formed from, h (the
    # last high half of a row, while registers last), and the window.
    regs = CALLER + CALLEE
    has_h = 8 + 1 + 2 + c + 2 <= len(regs)
    keep_rows = 8 + has_h + 2 + c + 2 <= len(regs)
    keep_q = keep_rows and 8 + has_h + 3 + c + 2 <= len(regs)
    fixed = ["x0", "x4", "x5"] + (["x2", "x3"] if keep_rows else []) \
        + (["x1"] if keep_q else [])
    rest = [r for r in regs if r not in fixed]
    mul, t0, t1, v0, v1 = rest[:5]
    h = rest[5] if has_h else None
    win = rest[5 + has_h:5 + has_h + c + 2]
    assert len(win) == c + 2, f"{n}: need {c+2} window registers"
    used = set(fixed) | set(rest[:5 + has_h]) | set(win)
    # The entry name_ns saves only the link register, where the window
    # needs it: the Karatsuba levels (gen_kara_arm.py), which keep nothing
    # in registers across their calls, call it directly.  The ABI entry
    # name saves the callee-saved registers the kernel uses and calls
    # name_ns; where it uses none, the two are one.
    abi = [r for r in CALLEE if r in used and r != "x30"]
    saved = [r for r in CALLEE if r in used and r == "x30"]
    spill = [k for k, r in ARGS.items() if r not in fixed]
    slot = {k: 8 * j for j, k in enumerate(spill)}
    # The frame: the pointers without a register, then the saved
    # registers, 16-aligned; none at all if there is nothing to keep.
    frame = 16 * ((8 * len(saved) + 8 * len(spill) + 15) // 16)
    base_save = frame - 8 * len(saved)

    if abi:
        pairs = abi + ["x30"]
        if len(pairs) % 2:
            pairs.append("xzr")
        sz = 8 * len(pairs)
        emit(f"\t.globl\t{name}")
        emit(f"\t.type\t{name}, %function")
        emit("\t.p2align\t4")
        emit(f"{name}:")
        emit(f"\tstp\t{pairs[0]}, {pairs[1]}, [sp, #-{sz}]!")
        for j in range(2, len(pairs), 2):
            emit(f"\tstp\t{pairs[j]}, {pairs[j+1]}, [sp, #{8 * j}]")
        emit(f"\tbl\t{name}_ns")
        for j in range(2, len(pairs), 2):
            a, b = pairs[j], pairs[j + 1]
            if b == "xzr":
                emit(f"\tldr\t{a}, [sp, #{8 * j}]")
            else:
                emit(f"\tldp\t{a}, {b}, [sp, #{8 * j}]")
        emit(f"\tldp\t{pairs[0]}, {pairs[1]}, [sp], #{sz}")
        emit("\tret")
        emit(f"\t.size\t{name}, .-{name}")
    emit(f"\t.globl\t{name}_ns")
    emit(f"\t.type\t{name}_ns, %function")
    if not abi:
        emit(f"\t.globl\t{name}")
        emit(f"\t.type\t{name}, %function")
    emit("\t.p2align\t4")
    if not abi:
        emit(f"{name}:")
    emit(f"{name}_ns:")
    if frame:
        emit(f"\tsub\tsp, sp, #{frame}")
    for j, r in enumerate(saved):		# save what we borrow
        emit(f"\tstr\t{r}, [sp, #{base_save + 8*j}]")
    for k in spill:				# the pointers without a register
        emit(f"\tstr\t{ARGS[k]}, [sp, #{slot[k]}]")

    # The window's limbs not yet written: a row writes them rather than
    # adding to them, so that the window needs no clearing (Q's first
    # row, the closing limbs of each pair's first row).
    fresh = set()

    def chain(vec, op, dst, add, skip_last=False):
        """The c products of one chain, op (mul or umulh) of the multiplier
        and vector vec, into dst(j); add(j) consumes product j.  The load
        and multiply for j+2 sit after the add for j, so that two products
        are always in flight and the multiplier never waits.  With h, the
        low chain forms the last high half as well, and the high chain
        skips it (skip_last)."""
        vs = (v0, v1)

        def form(j):
            if skip_last and j == c - 1:
                return
            emit(f"\tldr\t{vs[j & 1]}, [{vec}, #{8*j}]")
            emit(f"\t{op}\t{dst(j)}, {mul}, {vs[j & 1]}")
            if h and op == "mul" and j == c - 1:
                emit(f"\tumulh\t{h}, {mul}, {vs[j & 1]}")

        for j in range(min(2, c)):
            form(j)
        for j in range(c):
            add(j)
            if j + 2 < c:
                form(j + 2)

    def row(W, i, mult, vec, complement):
        """W[i..i+c+1] += (~)mult[i] * vec[0..c-1], as two chains."""
        if keep_rows:
            emit(f"\tldr\t{mul}, [{ARGS[mult]}, #{8*i}]")
        else:
            emit(f"\tldr\t{t0}, [sp, #{slot[mult]}]")
            emit(f"\tldr\t{mul}, [{t0}, #{8*i}]")
        if complement:
            emit(f"\tmvn\t{mul}, {mul}")
        ts = (t0, t1)
        # The low halves at limb i: into limbs not yet written, mul writes
        # them directly and there is no chain (Q's first row).
        if all((i + j) in fresh for j in range(c)):
            chain(vec, "mul", lambda j: W[i + j], lambda j: None)
            fresh.difference_update(range(i, i + c))
            carry = False
        else:
            def add(j):
                emit(f"\t{'adds' if j == 0 else 'adcs'}\t"
                     f"{W[i+j]}, {W[i+j]}, {ts[j & 1]}")
            chain(vec, "mul", lambda j: ts[j & 1], add)
            carry = True
        # Close the low chain.  With h, its carry goes into the last high
        # half, which is at most B - 2, so that nothing carries out of it:
        # one instruction, where adding it at limb i+c and its carry at
        # i+c+1 took two.  The partial sums stay below 3 B^(i+c+1), so
        # nothing carries out of the window's top limb.
        if carry and h:
            emit(f"\tadc\t{h}, {h}, xzr")
        elif carry and (i + c) in fresh:
            emit(f"\tadc\t{W[i+c]}, xzr, xzr")
            fresh.discard(i + c)
        elif carry:
            emit(f"\tadcs\t{W[i+c]}, {W[i+c]}, xzr")
            write(W, i + c + 1, f"{W[i+c+1]}, xzr")

        # The high halves at limb i+1.
        def addh(j):
            t = h if (h and j == c - 1) else ts[j & 1]
            d = W[i + 1 + j]
            src = "xzr" if (i + 1 + j) in fresh else d
            fresh.discard(i + 1 + j)
            emit(f"\t{'adds' if j == 0 else 'adcs'}\t{d}, {src}, {t}")
        chain(vec, "umulh", lambda j: ts[j & 1], addh, skip_last=bool(h))
        write(W, i + c + 1, f"{W[i+c+1]}, xzr")

    def write(W, k, operands):
        """adc into limb k, or of nothing into it while it is unwritten."""
        if k in fresh:
            emit(f"\tadc\t{W[k]}, xzr, xzr")
            fresh.discard(k)
        else:
            emit(f"\tadc\t{W[k]}, {operands}")

    def component(re):
        W = {k: win[k] for k in range(c + 2)}
        br, bi = ARGS["br"], ARGS["bi"]
        va, vb = (br, bi) if re else (bi, br)
        if re:
            o = ARGS["P"]
        elif keep_q:
            o = ARGS["Q"]
        else:
            o = ARGS["P"]
            emit(f"\tldr\t{o}, [sp, #{slot['Q']}]")
        fresh.clear()
        if re:			  # the window starts as bi (the + bi term)
            for k in range(c):
                emit(f"\tldr\t{W[k]}, [{vb}, #{8*k}]")
        else:
            fresh.update(range(c))
        fresh.update((c, c + 1))
        for i in range(m):
            row(W, i, "ar", va, False)
            row(W, i, "ai", vb, re)
            emit(f"\tstr\t{W[i]}, [{o}, #{8*i}]")
            r = W.pop(i)		  # the limb just stored frees a register
            if i + c + 2 <= m + c:
                W[i + c + 2] = r
                fresh.add(i + c + 2)
        assert not fresh
        if re:					  # P = S - bi B^m
            for k in range(c):
                emit(f"\tldr\t{v0}, [{vb}, #{8*k}]")
                emit(f"\t{'subs' if k == 0 else 'sbcs'}\t"
                     f"{W[m+k]}, {W[m+k]}, {v0}")
            emit(f"\tsbc\t{W[m+c]}, {W[m+c]}, xzr")
        for k in range(m, m + c + 1):
            emit(f"\tstr\t{W[k]}, [{o}, #{8*k}]")

    component(True)
    component(False)
    for j, r in enumerate(saved):
        emit(f"\tldr\t{r}, [sp, #{base_save + 8*j}]")
    if frame:
        emit(f"\tadd\tsp, sp, #{frame}")
    emit("\tret")
    emit(f"\t.size\t{name}_ns, .-{name}_ns")
    if not abi:
        emit(f"\t.size\t{name}, .-{name}")
    return "\n".join(out)


def main():
    print("/* Copyright (c) 2026 Huawei Technologies Co., Ltd.")
    print("   SPDX-License-Identifier: BSD-3-Clause-Clear  */")
    print("/* Generated by tools/gen_textbook_arm.py -- do not edit.")
    print("   The textbook Gaussian product for 2..%d limbs, aarch64." % NMAX)
    print("   See that file for what these compute and how.  */")
    print("#if defined (__aarch64__) && defined (__ELF__) "
          "&& !defined (GTC_NO_ASM)")
    print("\t.text")
    for n in range(2, NMAX + 1):
        print()
        print(gen(n))
    print('\t.section\t.note.GNU-stack,"",%progbits')
    print("#endif")


if __name__ == "__main__":
    main()
