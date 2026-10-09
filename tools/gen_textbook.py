#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Generate gtc_textbook_x86.S: the textbook Gaussian product for 2..20
limbs, x86-64 with BMI2 (mulx) and ADX (adcx, adox).

  void gtc_gmul_tbN (mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
                     mp_srcptr br, mp_srcptr bi)

computes P + iQ = (ar + i ai)(br + i bi) for N-limb components >= 0:
P = ar br - ai bi in two's complement and Q = ar bi + ai br, 2N+1 limbs each.

Each component is a sum of two products, so that it needs additions only:
  P = ar br + (~ai) bi + bi - bi B^N,  since (~ai) bi = (B^N - 1 - ai) bi,
  Q = ar bi + ai br.
It is computed by operand scanning.  Row i adds m * w at limb i, where m is
a limb of a row multiplier (ar, ~ai or ai; in rdx) and w a C-limb vector
in memory: mulx forms the partial products, the low halves go along the CF
chain (adcx) and the high halves along the OF chain (adox), so that the
two carry chains run side by side.  The rows go in pairs, ar_i and then
(~)ai_i; they all add into a window of C+2 limbs held in registers, and
after pair i the window's lowest limb is final and is stored.  With no
loads or stores per partial product, a product costs four micro-ops: the
two of mulx, adcx and adox.

The window must fit in the registers, C <= 10 (C <= 11 with one register
fewer, gen_rolled_rsp at 11 limbs).  From 12 limbs on, the
vectors br and bi are cut into column blocks of at most 10 limbs (gen_blocked):
one pass over all the rows per block, each with its own window.  The first
pass stores the non-negative sum S1 of its block (for P without the term
-bi B^N); every later pass adds the limbs of the sum so far into the OF
chain of its first row per pair, one limb below where that chain starts
otherwise, and so needs no extra carry chain.  The term -bi B^N is
subtracted at the end.  The blocks run at about the cycles per limb
product of the single-window kernels.

Some options are tuned per microarchitecture (TUNINGS below): where two
tunings differ, the generated file carries both and chooses with #if.

Run:  ./tools/gen_textbook.py > src/gtc_textbook_x86.S  (or: make asm)"""
import sys

NMAX = 10          # one window: c + 2 registers, and 12 are free
BMAX = 20          # column blocks

# Above NMAX the vectors are cut into column blocks: halves up to 18
# limbs, thirds for 19 and 20, whose 10-limb blocks leave no register for a
# pointer; the row multipliers copied to the frame at 13 to 16 limbs and at
# 19 and 20.
def blocks(n):
    """n -> (block widths, whether the row multipliers are copied)."""
    widths = ([(n + 1) // 2, n // 2] if n <= 18
              else [n // 3 + (j < n % 3) for j in range(3)])
    return widths, n in (13, 14, 15, 16, 19, 20)


# The tuned microarchitectures, in the order the generated file tests them.
# Each: the preprocessor condition that selects it and its options:
# "unrolled10", the 10-limb kernel in one window (gen) rather than rolled
# into two passes (gen_rolled), made for the Core's decoders; "limbcopy",
# the copies to the frame by limbs rather than by SSE (copy_to_frame);
# "pcomp" (see gen).  Where the entries' kernels differ, the generated file
# carries both and chooses with #if; where they agree, one copy.
ZEN_IF = ("defined (__znver1__) || defined (__znver2__) "
          "|| defined (__znver3__) || defined (__znver4__) "
          "|| defined (__znver5__)")
TUNINGS = [("zen", ZEN_IF, {"unrolled10", "limbcopy", "pcomp"}),
           ("core", None, set())]   # everything else

# The pointers a kernel keeps in its frame (and a zero limb), for those
# that get no register; not those whose vector or row is copied to the
# frame, which the kernel reads there instead.
FRAME_SLOTS = ["zero", "ar", "ai", "P", "Q", "br", "bi"]


def red_zone_frame(items):
    """Frame offsets from %rsp for items of (name, limbs), in order.  The
    frame starts in the red zone, the 128 bytes below %rsp that a leaf
    function may use, so that the first items, the most used ones, are
    within a byte's displacement of %rsp: a mulx on a vector copied there
    is then 7 bytes, not 10, and at 10 limbs the kernel 7% smaller and 5%
    faster.  Returns the offsets and how far %rsp must come down."""
    slot, at = {}, -128
    for k, w in items:
        slot[k] = at
        at += 8 * w
    return slot, max(at, 0)


def abi_entry(name, saved):
    """The ABI entry of kernel name: it saves the callee-saved registers
    the kernel uses and calls the kernel's own entry, name_ns, which saves
    none.  The Karatsuba levels (gen_kara.py), which keep nothing in
    registers across their calls, call the _ns entries directly."""
    out = [f"\t.globl\t{name}", f"\t.type\t{name}, @function",
           "\t.p2align\t4", f"{name}:"]
    out += [f"\tpush\t%{r}" for r in saved]
    out.append(f"\tcall\t{name}_ns")
    out += [f"\tpop\t%{r}" for r in reversed(saved)]
    out += ["\tret", f"\t.size\t{name}, .-{name}"]
    return out


def copy_to_frame(emit, src, off, c, tmp, by_limbs=False):
    """c limbs from register src's address to off(%rsp), 16 bytes at a
    time by SSE (the loads of up to eight first, then the stores), the odd
    limb by tmp: half the micro-ops of a copy by limbs.  The kernels use no
    other vector registers, and the legacy SSE encoding leaves the upper
    halves of the ymm registers alone.  by_limbs: by tmp, limb by limb
    (the Zen's tuning)."""
    if by_limbs:
        for j in range(c):
            emit(f"\tmov\t{8*j}(%{src}), %{tmp}")
            emit(f"\tmov\t%{tmp}, {off + 8*j}(%rsp)")
        return
    pairs = list(range(0, c - 1, 2))
    for x0 in range(0, len(pairs), 8):
        grp = pairs[x0:x0 + 8]
        for x, j in enumerate(grp):
            emit(f"\tmovdqu\t{8*j}(%{src}), %xmm{x}")
        for x, j in enumerate(grp):
            emit(f"\tmovdqu\t%xmm{x}, {off + 8*j}(%rsp)")
    if c % 2:
        emit(f"\tmov\t{8*(c-1)}(%{src}), %{tmp}")
        emit(f"\tmov\t%{tmp}, {off + 8*(c-1)}(%rsp)")


CALLER = ["rax", "rcx", "rsi", "rdi", "r8", "r9", "r10", "r11"]   # and rdx
CALLEE = ["rbx", "rbp", "r12", "r13", "r14", "r15"]


def r32(r):
    """The 32-bit name of a register, for zero idioms (2 bytes for the
    legacy registers)."""
    legacy = {"rax": "eax", "rbx": "ebx", "rcx": "ecx", "rdx": "edx",
              "rsi": "esi", "rdi": "edi", "rbp": "ebp"}
    return legacy.get(r, r + "d")


def first_row(emit, vec_v, vec_bi, W, c, T, z, bi_of=False, pre=None):
    """The first row of a component (or of a column block), which writes
    the window rather than adding to it: mulx puts each low half straight
    into its limb, the high halves go along the CF chain (alternating
    between the two registers of T), the last one straight into limb c;
    P adds bi along the OF chain (bi_of).  pre: a memory operand added at
    the head of the CF chain (a later block's earlier sum).  It needs CF =
    OF = 0 and z, a zero register (limb c+1), and leaves CF = OF = 0 with
    the OF carry in z.  Against a window cleared (or loaded with bi) and a
    row of adds: c+2 flag micro-ops on p0/p6 fewer for Q, c loads fewer
    for P, and c+1 zero idioms fewer."""
    for j in range(c):
        hi = W[c] if j == c - 1 else T[j % 2]
        emit(f"\tmulx\t{vec_v(j)}, %{W[j]}, %{hi}")
        if j == 0 and pre:
            emit(f"\tadcx\t{pre}, %{W[0]}")
        if j:
            emit(f"\tadcx\t%{T[(j-1) % 2]}, %{W[j]}")
        if bi_of:
            emit(f"\tadox\t{vec_bi(j)}, %{W[j]}")
    emit(f"\tadcx\t%{z}, %{W[c]}")
    if bi_of:
        emit(f"\tadox\t%{z}, %{W[c]}")
        emit(f"\tadox\t%{z}, %{z}")


def closer(emit, zero):
    """The closing of a row's two chains at limb t, the last high half in
    H: H + CF into t along the OF chain, the OF carry into t+1.  Before the
    first row of a pair, t+1 is the window's newest limb, still 0, so that
    row adds it rather than the zero, which may be in memory; the second
    row cannot, t+1 then holding the first row's OF carry."""
    def close(ri, H, t, t1):
        z = f"%{t1}" if ri == 0 else zero
        emit(f"\tadcx\t{z}, %{H}")
        emit(f"\tadox\t%{H}, %{t}")
        emit(f"\tadox\t{z}, %{t1}")
    return close


def gen(n, by_limbs=False, pcomp=False):
    """n-limb components in one window.  Rows: the m = n limbs of ar and
    ai.  Vectors: br and bi, c = n limbs.

    pcomp: ~ai is copied to the frame, complemented, at entry, so that P's
    second rows load their multiplier ready instead of complementing it
    (xor $-1, an ALU op on the row's path), and every second row, of P
    and of Q, starts its flags fresh with a zero idiom on L, which the
    first mulx overwrites.  For the Zen, where every ALU op costs about a
    cycle: the mulx stream (ALU1 only) and the flag adds (ALU0-2) keep the
    integer ports full."""
    m = c = n
    name = f"gtc_gmul_tb{n}"
    out = []
    emit = out.append
    # Registers: rdx (the row multiplier), H and L (mulx's high and low
    # halves), the window of c+2 limbs, then as many pointers as fit, in
    # order of use: the two vectors of the rows, the two row multipliers,
    # a zero, the output.  The rest is reloaded from the frame when needed.
    avail = 15 - 3 - (c + 2)
    keep = ["br", "bi", "ar", "ai", "zero", "out"][:max(avail, 0)]
    copies = [k for k in ["br", "bi"] if k not in keep]
    # Row multipliers without a register are copied to the frame, so that
    # a row loads its limb from there rather than its pointer first.
    rcopies = [k for k in ["ar", "ai"] if k not in keep]
    slot, frame = red_zone_frame([(k + "c", c) for k in copies]
                                 + [("zero", 1)]
                                 + [(k + "c", c) for k in rcopies]
                                 + [(k, 1) for k in FRAME_SLOTS
                                    if k != "zero"
                                    and k not in copies + rcopies]
                                 + ([("naic", c)] if pcomp else []))
    # The pointers that are kept stay in the registers they come in, the
    # output starting as P's; only ar must leave rdx.  Then H and L, the
    # rest of the pointers and the window, caller-saved registers first.
    args = {"P": "rdi", "Q": "rsi", "ar": "rdx", "ai": "rcx", "br": "r8",
            "bi": "r9"}
    stay = {"br": "r8", "bi": "r9", "ai": "rcx", "out": "rdi"}
    ptr = {k: stay[k] for k in keep if k in stay}
    rest = [r for r in CALLER + CALLEE if r not in ptr.values()]
    H, L = rest.pop(0), rest.pop(0)
    for k in keep:
        if k not in ptr:
            ptr[k] = rest.pop(0)
    win = rest[: c + 2]
    assert len(win) == c + 2
    saved = [r for r in CALLEE if r in set(ptr.values()) | set(win)]

    out.extend(abi_entry(name, saved))
    emit(f"\t.globl\t{name}_ns")
    emit(f"\t.type\t{name}_ns, @function")
    emit("\t.p2align\t5")
    emit(f"{name}_ns:")
    if frame:
        emit(f"\tsub\t${frame}, %rsp")
    # To the frame: Q, and what has no register and no copy there.
    for k, r in args.items():
        if k == "Q" or (k == "P" and "out" not in ptr) or (
                k not in ("P", "Q") and k not in ptr
                and k not in copies + rcopies):
            emit(f"\tmov\t%{r}, {slot[k]}(%rsp)")
    if "zero" not in keep:
        emit(f"\tmovq\t$0, {slot['zero']}(%rsp)")
    for k in copies + rcopies:
        copy_to_frame(emit, args[k], slot[k + "c"], c, H, by_limbs)
    if pcomp:
        for j in range(c):
            emit(f"\tmov\t{8*j}(%{args['ai']}), %{H}")
            emit(f"\tnot\t%{H}")
            emit(f"\tmov\t%{H}, {slot['naic'] + 8*j}(%rsp)")
    if "ar" in ptr:
        emit(f"\tmov\t%rdx, %{ptr['ar']}")
    if "zero" in ptr:
        emit(f"\txor\t%{ptr['zero']}, %{ptr['zero']}")
    zero = f"%{ptr['zero']}" if "zero" in ptr else f"{slot['zero']}(%rsp)"
    close = closer(emit, zero)

    def base(k):
        """A register holding pointer k (loaded into L if not kept)."""
        if k in ptr:
            return ptr[k]
        emit(f"\tmov\t{slot[k]}(%rsp), %{L}")
        return L

    def vec(k, j):
        if k in ptr:
            return f"{8*j}(%{ptr[k]})"
        return f"{slot[k + 'c'] + 8*j}(%rsp)"

    def component(re):
        outk = "P" if re else "Q"
        if "out" in ptr and not re:
            emit(f"\tmov\t{slot[outk]}(%rsp), %{ptr['out']}")
        W = {k: win[k] for k in range(c + 2)}
        # The first row writes the window (first_row); only its limb c+1
        # is cleared, which also clears CF and OF.
        emit(f"\txor\t%{r32(W[c+1])}, %{r32(W[c+1])}")
        rows = [("ar", "br" if re else "bi", False),
                ("ai", "bi" if re else "br", re)]
        for i in range(m):
            for ri, (mult, v, complement) in enumerate(rows):
                if pcomp and complement:
                    emit(f"\tmov\t{slot['naic'] + 8*i}(%rsp), %rdx")
                elif mult in rcopies:
                    emit(f"\tmov\t{slot[mult + 'c'] + 8*i}(%rsp), %rdx")
                else:
                    emit(f"\tmov\t{8*i}(%{base(mult)}), %rdx")
                if i == 0 and ri == 0:
                    first_row(emit, lambda j: vec(v, j),
                              lambda j: vec("bi", j), W, c, [L, H],
                              W[c+1], bi_of=re)
                    continue
                if pcomp and ri == 1:
                    # A zero idiom: CF = OF = 0 without waiting for the
                    # row before's closing, and no ALU op.
                    emit(f"\txor\t%{r32(L)}, %{r32(L)}")
                elif complement:
                    # The complement by xor rather than not: it also sets
                    # CF = OF = 0, which they are already, but without
                    # waiting for the row before, so that this row's chains
                    # need not wait for that row's closing.
                    emit("\txor\t$-1, %rdx")
                emit(f"\tmulx\t{vec(v, 0)}, %{L}, %{H}")
                emit(f"\tadcx\t%{L}, %{W[i]}")
                for j in range(1, c):
                    emit(f"\tadox\t%{H}, %{W[i+j]}")
                    emit(f"\tmulx\t{vec(v, j)}, %{L}, %{H}")
                    emit(f"\tadcx\t%{L}, %{W[i+j]}")
                # Close both chains.  Both carries are due at limb i+c, and
                # so is the last high half, which is at most B - 2: the CF
                # carry goes into it, it goes into the window along the OF
                # chain, and only the OF carry is left for limb i+c+1.  The
                # partial sums stay below 3 B^(i+c+1), so nothing carries
                # out of that limb, and CF = OF = 0 afterwards.
                close(ri, H, W[i+c], W[i+c+1])
            emit(f"\tmov\t%{W[i]}, {8*i}(%{base('out' if 'out' in ptr else outk)})")
            r = W.pop(i)
            if i + c + 2 <= m + c:
                W[i + c + 2] = r
                emit(f"\txor\t%{r}, %{r}")
        if re:                         # P = S - bi B^m
            for k in range(c):
                emit(f"\t{'sub' if k == 0 else 'sbb'}\t{vec('bi', k)}, %{W[m+k]}")
            emit(f"\tsbb\t$0, %{W[m+c]}")
        o = base("out" if "out" in ptr else outk)
        for k in range(m, m + c + 1):
            emit(f"\tmov\t%{W[k]}, {8*k}(%{o})")

    component(True)
    component(False)
    if frame:
        emit(f"\tadd\t${frame}, %rsp")
    emit("\tret")
    emit(f"\t.size\t{name}_ns, .-{name}_ns")
    return "\n".join(out)


def gen_blocked(n, widths, copy_rows, by_limbs=False):
    """n-limb components, the vectors in column blocks of the given
    widths: one pass over all rows per block."""
    m = c = n
    name = f"gtc_gmul_tb{n}"
    blocks, j0 = [], 0
    for w in widths:
        blocks.append((j0, w))
        j0 += w
    out = []
    emit = out.append
    cmax = max(widths)
    avail = 15 - 3 - (cmax + 2)
    keep = ["br", "bi", "ar", "ai", "zero", "out"][:max(avail, 0)]
    copies = [k for k in ["br", "bi"] if k not in keep]
    rcopies = [k for k in ["ar", "ai"] if k not in keep] if copy_rows else []
    slot, frame = red_zone_frame([(k + "c", c) for k in copies]
                                 + [("zero", 1)]
                                 + [(k + "c", c) for k in rcopies]
                                 + [(k, 1) for k in FRAME_SLOTS if k != "zero"
                                    and k not in copies + rcopies])
    # Registers as in gen: the kept pointers where they come in.
    args = {"P": "rdi", "Q": "rsi", "ar": "rdx", "ai": "rcx", "br": "r8",
            "bi": "r9"}
    stay = {"br": "r8", "bi": "r9", "ai": "rcx", "out": "rdi"}
    ptr = {k: stay[k] for k in keep if k in stay}
    rest = [r for r in CALLER + CALLEE if r not in ptr.values()]
    H, L = rest.pop(0), rest.pop(0)
    for k in keep:
        if k not in ptr:
            ptr[k] = rest.pop(0)
    win = rest[: cmax + 2]
    assert len(win) == cmax + 2
    saved = [r for r in CALLEE if r in set(ptr.values()) | set(win)]

    out.extend(abi_entry(name, saved))
    emit(f"\t.globl\t{name}_ns")
    emit(f"\t.type\t{name}_ns, @function")
    emit("\t.p2align\t5")
    emit(f"{name}_ns:")
    if frame:
        emit(f"\tsub\t${frame}, %rsp")
    for k, r in args.items():
        if k == "Q" or (k == "P" and "out" not in ptr) or (
                k not in ("P", "Q") and k not in ptr
                and k not in copies + rcopies):
            emit(f"\tmov\t%{r}, {slot[k]}(%rsp)")
    if "zero" not in keep:
        emit(f"\tmovq\t$0, {slot['zero']}(%rsp)")
    for k in copies + rcopies:
        copy_to_frame(emit, args[k], slot[k + "c"], c, H, by_limbs)
    if "ar" in ptr:
        emit(f"\tmov\t%rdx, %{ptr['ar']}")
    if "zero" in ptr:
        emit(f"\txor\t%{ptr['zero']}, %{ptr['zero']}")
    zero = f"%{ptr['zero']}" if "zero" in ptr else f"{slot['zero']}(%rsp)"
    close = closer(emit, zero)
    outnow = ["P"]                     # what the output register holds

    def base(k, reg):
        """A register holding pointer k (loaded into reg if not kept)."""
        if k in ptr:
            return ptr[k]
        emit(f"\tmov\t{slot[k]}(%rsp), %{reg}")
        return reg

    def vec(k, j):
        if k in ptr:
            return f"{8*j}(%{ptr[k]})"
        return f"{slot[k + 'c'] + 8*j}(%rsp)"

    def outbase(re, reg):
        k = "P" if re else "Q"
        if "out" in ptr:
            if outnow[0] != k:
                emit(f"\tmov\t{slot[k]}(%rsp), %{ptr['out']}")
                outnow[0] = k
            return ptr["out"]
        return base(k, reg)

    def component(re):
        for b, (j0, cb) in enumerate(blocks):
            W = {k: win[k] for k in range(cb + 2)}
            # The first row writes the window (first_row), bi's block
            # along OF for P; the xor clears limb cb+1, CF and OF.
            emit(f"\txor\t%{r32(W[cb+1])}, %{r32(W[cb+1])}")
            rows = [("ar", "br" if re else "bi", False),
                    ("ai", "bi" if re else "br", re)]
            for i in range(m):
                for ri, (mult, v, complement) in enumerate(rows):
                    if mult in rcopies:
                        emit(f"\tmov\t{slot[mult + 'c'] + 8*i}(%rsp), %rdx")
                    else:
                        emit(f"\tmov\t{8*i}(%{base(mult, L)}), %rdx")
                    if complement:          # as in gen
                        emit("\txor\t$-1, %rdx")
                    if i == 0 and ri == 0:
                        # a later block adds the earlier blocks' sum at
                        # limb j0 at the head of the CF chain
                        pre = None
                        if b > 0:
                            pre = f"{8*j0}(%{outbase(re, H)})"
                        first_row(emit, lambda j: vec(v, j0 + j),
                                  lambda j: vec("bi", j0 + j), W, cb,
                                  [L, H], W[cb+1], bi_of=re, pre=pre)
                        continue
                    if b > 0 and ri == 0:
                        # The sum of the earlier blocks at limb j0+i, into
                        # the OF chain, which otherwise starts at i+1.  Its
                        # limbs from j0+i on are not stored over before.
                        o = outbase(re, H)
                        emit(f"\tadox\t{8*(j0+i)}(%{o}), %{W[i]}")
                    emit(f"\tmulx\t{vec(v, j0)}, %{L}, %{H}")
                    emit(f"\tadcx\t%{L}, %{W[i]}")
                    for j in range(1, cb):
                        emit(f"\tadox\t%{H}, %{W[i+j]}")
                        emit(f"\tmulx\t{vec(v, j0 + j)}, %{L}, %{H}")
                        emit(f"\tadcx\t%{L}, %{W[i+j]}")
                    # Close both chains, as in gen: the partial sums stay
                    # below 4 B^(i+cb+1) (with the added limbs of the earlier
                    # blocks), so nothing carries out of limb i+cb+1.
                    close(ri, H, W[i+cb], W[i+cb+1])
                o = outbase(re, L)
                emit(f"\tmov\t%{W[i]}, {8*(j0+i)}(%{o})")
                r = W.pop(i)
                if i + 1 < m:
                    W[i + cb + 2] = r
                    emit(f"\txor\t%{r}, %{r}")
            # The window holds limbs j0+m .. j0+m+cb of the sum, which
            # has no more.
            o = outbase(re, L)
            if b > 0:
                # The top limb of the earlier blocks' sum, at j0 + m.
                emit(f"\tadd\t{8*(j0+m)}(%{o}), %{W[m]}")
                for k in range(m + 1, m + cb + 1):
                    emit(f"\tadc\t$0, %{W[k]}")
            if re and b == len(blocks) - 1:
                # P = S - bi B^m: limbs m .. m+j0-1 are in memory, the rest
                # in the window.
                for t in range(j0):
                    emit(f"\tmov\t{8*(m+t)}(%{o}), %rdx")
                    emit(f"\t{'sub' if t == 0 else 'sbb'}\t{vec('bi', t)}, %rdx")
                    emit(f"\tmov\t%rdx, {8*(m+t)}(%{o})")
                for t in range(cb):
                    op = "sub" if j0 == 0 and t == 0 else "sbb"
                    emit(f"\t{op}\t{vec('bi', j0 + t)}, %{W[m+t]}")
                emit(f"\tsbb\t$0, %{W[m+cb]}")
            for k in range(m, m + cb + 1):
                emit(f"\tmov\t%{W[k]}, {8*(j0+k)}(%{o})")

    component(True)
    component(False)
    if frame:
        emit(f"\tadd\t${frame}, %rsp")
    emit("\tret")
    emit(f"\t.size\t{name}_ns, .-{name}_ns")
    return "\n".join(out)


ROLLED = {10}      # sizes generated by gen_rolled rather than gen


def gen_rolled(n):
    """n-limb components (9 or 10) in one window of c+1 registers, P and Q
    by one loop of two passes over the same row code.

    One register fewer for the window: after the first row of a pair, the
    OF carry into limb i+c+1 stays in OF (not in a register) while the
    second row starts on limb i: that row's CF chain follows the first's
    (CF = 0 after its closing), and once its first adcx has made limb i
    final, limb i is stored, its register is cleared by a mov (no flags)
    and takes the pending OF carry by adox: it is limb i+c+1 from then on.
    So the second row of a pair waits for the first's closing, but the
    register saved is a frame base B other than %rsp: every mulx on a
    copied vector is 6 bytes rather than 7 (no SIB byte), and B has a
    disp8 reach of its own.

    The two passes: row A uses br and row B uses bi in both, the
    multipliers being P: (ar, ~ai) and Q: (ai, ar), from a frame
    [ai | ar | ~ai] (each 8c bytes) that %rsp walks down by 8c for the Q
    pass, so that the same displacements address both.  Each low limb is
    stored over the row-B multiplier it has just used (P's over ~ai, Q's
    over ar) and both are copied out at the end.  The first row of each
    pass (first_row) and P's subtraction of bi B^n (a branch) are outside
    the shared code.  The shared code is half the size of gen's, so that
    some of the second pass comes from the micro-op cache even inside a
    product."""
    c = n
    name = f"gtc_gmul_tb{n}"
    out = []
    emit = out.append
    H, L, B = "rax", "rcx", "rbx"
    avail = 15 - 4 - (c + 1)
    keep = ["br", "bi"][:max(avail, 0)]
    ptr = {k: {"br": "r8", "bi": "r9"}[k] for k in keep}
    copies = [k for k in ["br", "bi"] if k not in keep]
    pool = ["rsi", "rdi", "r8", "r9", "r10", "r11", "rbp", "r12", "r13",
            "r14", "r15"]
    win = [r for r in pool if r not in ptr.values()][: c + 1]
    assert len(win) == c + 1
    saved = [r for r in CALLEE if r in set(win) | {B}]
    S = 8 * c                                   # the pass stride
    # Offsets from %rsp in the P pass: the multipliers within a byte's
    # displacement; the pass-indexed slots (P's at q, Q's at q - S) above
    # them, so that nothing live is more than 128 bytes below %rsp (the
    # red zone) while %rsp is at its P-pass value; B's region above all.
    a0 = -112
    o_ai, o_ar, o_nai = a0, a0 + S, a0 + 2 * S
    q, qf = a0 + 4 * S, a0 + 4 * S + 8
    assert a0 + 3 * S <= 128 and q - S >= a0 + 3 * S
    BOFF = a0 + 4 * S + 16 + 128
    frame = BOFF + 128
    bslot = {}
    k0 = 0
    if "bi" in copies:
        for j in range(c):
            bslot[("bi", j)] = 8 * j
        k0 = 8 * c
    bslot["zero"] = k0
    if "br" in copies:
        for j in range(c):
            bslot[("br", j)] = -8 * c + 8 * j

    def vec(k, j):
        if k in ptr:
            return f"{8*j}(%{ptr[k]})" if j else f"(%{ptr[k]})"
        o = bslot[(k, j)]
        return f"{o}(%{B})" if o else f"(%{B})"
    Z = f"{bslot['zero']}(%{B})"
    lab = f".Lrolled{n}"

    out.extend(abi_entry(name, saved))
    emit(f"\t.globl\t{name}_ns")
    emit(f"\t.type\t{name}_ns, @function")
    emit("\t.p2align\t5")
    emit(f"{name}_ns:")
    emit(f"\tsub\t${frame}, %rsp")
    emit(f"\tlea\t{BOFF}(%rsp), %{B}")
    emit(f"\tmov\t%rdi, {q}(%rsp)")
    emit(f"\tmov\t%rsi, {q - S}(%rsp)")
    emit(f"\tmovq\t$1, {qf}(%rsp)")
    emit(f"\tmovq\t$0, {qf - S}(%rsp)")
    emit(f"\tmovq\t$0, {Z}")
    srcs = {"br": "r8", "bi": "r9"}
    for k in copies:
        for j in range(0, c - 1, 2):
            emit(f"\tmovdqu\t{8*j}(%{srcs[k]}), %xmm0")
            emit(f"\tmovdqu\t%xmm0, {vec(k, j)}")
        if c % 2:
            emit(f"\tmov\t{8*(c-1)}(%{srcs[k]}), %{H}")
            emit(f"\tmov\t%{H}, {vec(k, c-1)}")
    rolled_multipliers(emit, c, o_ar, o_ai, o_nai, H)
    rolled_head(emit, win, c, vec, Z, o_ar, True)
    emit(f"{lab}_rows:")
    W = rolled_rows(emit, win, c, vec, Z, o_ar, o_nai)
    rolled_close(emit, W, c, vec, f"{qf}(%rsp)", f"{q}(%rsp)", lab)
    emit(f"\tsub\t${S}, %rsp")
    rolled_head(emit, win, c, vec, Z, o_ar, False)
    emit(f"\tjmp\t{lab}_rows")
    emit(f"{lab}_done:")
    # %rsp back to its P-pass value; Q's low limbs are in the ar region,
    # P's in ~ai
    emit(f"\tadd\t${S}, %rsp")
    rolled_copy_out(emit, c, ((q - S, o_ar), (q, o_nai)), L, H)
    emit(f"\tadd\t${frame}, %rsp")
    emit("\tret")
    emit(f"\t.size\t{name}_ns, .-{name}_ns")
    return "\n".join(out)


ROLLED_RSP = {11}  # sizes generated by gen_rolled_rsp rather than blocks


def gen_rolled_rsp(n):
    """n = 11 limbs in one window of c+1 = 12 registers, P and Q by one
    loop of two passes over the same row code, as gen_rolled does; the
    column blocks need two passes over all the rows.

    Twelve registers for the window leave none for a frame base, so every
    operand is addressed from %rsp, and %rsp itself moves, up by
    S2 = 8(2c+1) for the Q pass.  The frame holds one block
    [br | bi | zero] per pass, each within a byte's displacement of that
    pass's %rsp (the P pass's starts 128 bytes below it, in the red zone;
    it is dead, and below the red zone, in the Q pass), and above them the
    multipliers [~ai | P's slots | ar | Q's slots | ai] at the stride S2,
    so that the same displacements address the P pass's (ar, ~ai) and the
    Q pass's (ai, ar), and each pass's output pointer and flag lie in the
    gap after its row-B multipliers.  At 11 limbs 6.3 KB rather than the
    column blocks' 11.4."""
    c = n
    name = f"gtc_gmul_tb{n}"
    out = []
    emit = out.append
    H, L = "rax", "rcx"
    win = ["rsi", "rdi", "r8", "r9", "r10", "r11", "rbx", "rbp", "r12",
           "r13", "r14", "r15"][: c + 1]
    assert len(win) == c + 1
    saved = [r for r in CALLEE if r in set(win)]
    S = 8 * c
    S2 = 8 * (2 * c + 1)
    # Offsets from %rsp in the P pass (R): the vectors and the zero of
    # the pass's block, then ~ai, P's pointer and flag, ar, Q's pointer
    # and flag, ai.
    d_br, d_bi, d_z = -128, -128 + S, -128 + 2 * S
    o_nai = -128 + 2 * S2
    o_ar, o_ai = o_nai + S2, o_nai + 2 * S2
    q = o_nai + S
    assert q + 16 <= o_ar
    frame = (o_ai + S + 15) & ~15
    Z = f"{d_z}(%rsp)"
    srcs = {"br": "r8", "bi": "r9"}

    def vec(k, j):
        return f"{(d_br if k == 'br' else d_bi) + 8*j}(%rsp)"
    lab = f".Lrolled{n}"

    out.extend(abi_entry(name, saved))
    emit(f"\t.globl\t{name}_ns")
    emit(f"\t.type\t{name}_ns, @function")
    emit("\t.p2align\t5")
    emit(f"{name}_ns:")
    emit(f"\tsub\t${frame}, %rsp")
    emit(f"\tmov\t%rdi, {q}(%rsp)")
    emit(f"\tmov\t%rsi, {q + S2}(%rsp)")
    emit(f"\tmovq\t$1, {q + 8}(%rsp)")
    emit(f"\tmovq\t$0, {q + 8 + S2}(%rsp)")
    emit(f"\tmovq\t$0, {Z}")
    emit(f"\tmovq\t$0, {d_z + S2}(%rsp)")
    for k in ("br", "bi"):                      # into both blocks
        o = d_br if k == "br" else d_bi
        for j in range(0, c - 1, 2):
            emit(f"\tmovdqu\t{8*j}(%{srcs[k]}), %xmm0")
            emit(f"\tmovdqu\t%xmm0, {o + 8*j}(%rsp)")
            emit(f"\tmovdqu\t%xmm0, {o + 8*j + S2}(%rsp)")
        if c % 2:
            emit(f"\tmov\t{8*(c-1)}(%{srcs[k]}), %{H}")
            emit(f"\tmov\t%{H}, {o + 8*(c-1)}(%rsp)")
            emit(f"\tmov\t%{H}, {o + 8*(c-1) + S2}(%rsp)")
    rolled_multipliers(emit, c, o_ar, o_ai, o_nai, H)
    # Each pass: row A from o_ar, row B from o_ar - S2 (P: ar and ~ai;
    # Q, with %rsp up by S2: ai and ar).
    rolled_head(emit, win, c, vec, Z, o_ar, True)
    emit(f"{lab}_rows:")
    W = rolled_rows(emit, win, c, vec, Z, o_ar, o_nai)
    rolled_close(emit, W, c, vec, f"{q + 8}(%rsp)", f"{q}(%rsp)", lab)
    emit(f"\tadd\t${S2}, %rsp")
    rolled_head(emit, win, c, vec, Z, o_ar, False)
    emit(f"\tjmp\t{lab}_rows")
    emit(f"{lab}_done:")
    # %rsp = R + S2: Q's low limbs over ar, P's over ~ai
    rolled_copy_out(emit, c, ((q, o_nai), (q - S2, o_nai - S2)), L, H)
    emit(f"\tadd\t${frame - S2}, %rsp")
    emit("\tret")
    emit(f"\t.size\t{name}_ns, .-{name}_ns")
    return "\n".join(out)


def rolled_multipliers(emit, c, o_ar, o_ai, o_nai, H):
    """The row multipliers ar, ai and ~ai of a rolled kernel copied from
    rdx and rcx to the frame, 16 bytes at a time by SSE."""
    emit("\tpcmpeqd\t%xmm15, %xmm15")
    for j in range(0, c - 1, 2):
        emit(f"\tmovdqu\t{8*j}(%rdx), %xmm0")
        emit(f"\tmovdqu\t{8*j}(%rcx), %xmm1")
        emit(f"\tmovdqu\t%xmm0, {o_ar + 8*j}(%rsp)")
        emit(f"\tmovdqu\t%xmm1, {o_ai + 8*j}(%rsp)")
        emit("\tpxor\t%xmm15, %xmm1")
        emit(f"\tmovdqu\t%xmm1, {o_nai + 8*j}(%rsp)")
    if c % 2:
        j = c - 1
        emit(f"\tmov\t{8*j}(%rdx), %{H}")
        emit(f"\tmov\t%{H}, {o_ar + 8*j}(%rsp)")
        emit(f"\tmov\t{8*j}(%rcx), %{H}")
        emit(f"\tmov\t%{H}, {o_ai + 8*j}(%rsp)")
        emit(f"\tnot\t%{H}")
        emit(f"\tmov\t%{H}, {o_nai + 8*j}(%rsp)")


def rolled_head(emit, win, c, vec, Z, oA, re):
    """Row A of pair 0 of a rolled pass, which writes the window (rax and
    rcx the high halves, the multipliers at oA(%rsp)).  The window has no
    register for limb c+1 here, so the zero Z is the frame's and the OF
    carry is left pending, as after every row A."""
    H, L = "rax", "rcx"
    W = {k: win[k] for k in range(c + 1)}
    emit(f"\txor\t%{r32(L)}, %{r32(L)}")
    emit(f"\tmov\t{oA}(%rsp), %rdx")
    T = [L, H]
    for j in range(c):
        hi = W[c] if j == c - 1 else T[j % 2]
        emit(f"\tmulx\t{vec('br', j)}, %{W[j]}, %{hi}")
        if j:
            emit(f"\tadcx\t%{T[(j-1) % 2]}, %{W[j]}")
        if re:
            emit(f"\tadox\t{vec('bi', j)}, %{W[j]}")
    emit(f"\tadcx\t{Z}, %{W[c]}")
    if re:
        emit(f"\tadox\t{Z}, %{W[c]}")


def rolled_rows(emit, win, c, vec, Z, oA, oB):
    """The rows of a rolled pass, shared by P and Q: row A by br with the
    multipliers at oA(%rsp), row B by bi with those at oB(%rsp), over
    which each low limb is stored.  Returns the window's final map."""
    H, L = "rax", "rcx"
    W = {k: win[k] for k in range(c + 1)}
    for i in range(c):
        if i:
            # row A: the xor breaks the flags' dependency on row B before
            emit(f"\txor\t%{r32(L)}, %{r32(L)}")
            emit(f"\tmov\t{oA + 8*i}(%rsp), %rdx")
            emit(f"\tmulx\t{vec('br', 0)}, %{L}, %{H}")
            emit(f"\tadcx\t%{L}, %{W[i]}")
            for j in range(1, c):
                emit(f"\tadox\t%{H}, %{W[i+j]}")
                emit(f"\tmulx\t{vec('br', j)}, %{L}, %{H}")
                emit(f"\tadcx\t%{L}, %{W[i+j]}")
            emit(f"\tadcx\t{Z}, %{H}")
            emit(f"\tadox\t%{H}, %{W[i+c]}")
        # row B: limb i final after its first adcx, then reused for the
        # pending OF carry, limb i+c+1
        emit(f"\tmov\t{oB + 8*i}(%rsp), %rdx")
        emit(f"\tmulx\t{vec('bi', 0)}, %{L}, %{H}")
        emit(f"\tadcx\t%{L}, %{W[i]}")
        emit(f"\tmov\t%{W[i]}, {oB + 8*i}(%rsp)")
        r = W.pop(i)
        W[i + c + 1] = r
        emit(f"\tmov\t$0, %{r32(r)}")
        emit(f"\tadox\t%{r}, %{r}")
        for j in range(1, c):
            emit(f"\tadox\t%{H}, %{W[i+j]}")
            emit(f"\tmulx\t{vec('bi', j)}, %{L}, %{H}")
            emit(f"\tadcx\t%{L}, %{W[i+j]}")
        emit(f"\tadcx\t{Z}, %{H}")
        emit(f"\tadox\t%{H}, %{W[i+c]}")
        emit(f"\tadox\t{Z}, %{W[i+c+1]}")
    return W


def rolled_close(emit, W, c, vec, flag, ptr, lab):
    """The end of a rolled pass: P's subtraction of bi B^c where the pass
    flag (1 for P, 0 for Q) says so, the high limbs stored through the
    pass's output pointer, and on to {lab}_done after Q's pass."""
    H, L = "rax", "rcx"
    emit(f"\tcmpq\t$0, {flag}")
    emit(f"\tje\t{lab}_q")
    for k in range(c):                          # P = S - bi B^c
        emit(f"\t{'sub' if k == 0 else 'sbb'}\t{vec('bi', k)}, %{W[c+k]}")
    emit(f"\tsbb\t$0, %{W[2*c]}")
    emit(f"{lab}_q:")
    emit(f"\tmov\t{ptr}, %{L}")
    for k in range(c, 2 * c + 1):
        emit(f"\tmov\t%{W[k]}, {8*k}(%{L})")
    emit(f"\tcmpq\t$0, {flag}")
    emit(f"\tje\t{lab}_done")


def rolled_copy_out(emit, c, outs, L, H):
    """The low limbs of P and Q, stored over row B's multipliers during
    the passes, copied to the outputs: outs = ((pointer slot, base), ...)
    as offsets from %rsp."""
    for key, base in outs:
        emit(f"\tmov\t{key}(%rsp), %{L}")
        for j in range(0, c - 1, 2):
            emit(f"\tmovdqu\t{base + 8*j}(%rsp), %xmm{(j//2) % 8}")
            emit(f"\tmovdqu\t%xmm{(j//2) % 8}, {8*j}(%{L})")
        if c % 2:
            emit(f"\tmov\t{base + 8*(c-1)}(%rsp), %{H}")
            emit(f"\tmov\t%{H}, {8*(c-1)}(%{L})")


def kernels(opts=()):
    """size -> code, under a tuning's options."""
    bl = "limbcopy" in opts
    rolled = set() if "unrolled10" in opts else ROLLED
    pc = "pcomp" in opts
    out = {n: gen_rolled(n) if n in rolled else gen(n, bl, pc)
           for n in range(2, NMAX + 1)}
    for n in range(NMAX + 1, BMAX + 1):
        if n in ROLLED_RSP:
            out[n] = gen_rolled_rsp(n)
            continue
        widths, copy_rows = blocks(n)
        out[n] = gen_blocked(n, widths, copy_rows, bl)
    return out


def main():
    tunings = TUNINGS
    code = {name: kernels(opts) for name, _, opts in tunings}

    print("/* Copyright (c) 2026 Huawei Technologies Co., Ltd.")

    print("   SPDX-License-Identifier: BSD-3-Clause-Clear  */")

    print("/* Generated by gen_textbook.py; do not edit.  */")
    print()
    print("#if defined (__x86_64__) && defined (__ELF__) && defined (__BMI2__) "
          "&& defined (__ADX__) && !defined (GTC_NO_ASM)")
    # The section aligned to 64 bytes, so that the kernels keep their place
    # in the 64-byte fetch windows whatever the size of the code before them
    # (gtc.o): with the 32 of their own .p2align, a change of gtc.c moved
    # them by 32 mod 64 at random, 0.1-0.2% of whole products.
    print("\t.text")
    print("\t.p2align\t6")
    for n in range(2, BMAX + 1):
        print()
        variants = [code[name][n] for name, _, _ in tunings]
        if all(v == variants[0] for v in variants):
            print(variants[0])         # every tuning agrees at this size
            continue
        for j, (name, cond, _) in enumerate(tunings):
            if cond is None:           # the last one is the fallback
                print("#else" if j else "")
            else:
                print(f"#{'if' if j == 0 else 'elif'} {cond}")
            print(code[name][n])
        print("#endif")
    print('\t.section\t.note.GNU-stack,"",@progbits')
    print("#endif")



if __name__ == "__main__":
    sys.exit(main())
