#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Generate gtc_linear_x86.S: linear work of the Gaussian products on both
components at once, x86-64 with ADX.  The real part runs along the CF chain
(adcx), the imaginary part along the OF chain (adox), in one loop.  Inside a
chain only instructions that leave CF and OF alone may run: loop control
uses lea and jrcxz, loop counts are loaded with mov, zeros are made with
mov, and a subtraction is the addition of the complement (not) with a
carry-in of 1.  The loops are unrolled four times, after a remainder loop.

Run:  ./tools/gen_linear.py > src/gtc_linear_x86.S  (or: make asm)"""
import sys

UNROLL = 4
out = []
emit = out.append
label_no = [0]


def lab(stem):
    label_no[0] += 1
    return f".L{stem}{label_no[0]}"


def dual_pass(pr, qr, count_slot, sub_p=False, sub_q=False, zero=False):
    """One pass over count limbs (count in the io slot count_slot):
       P: [pr.dst + j] = [pr.a + j] + [pr.b + j]   (CF chain)
       Q: [qr.dst + j] = [qr.a + j] + [qr.b + j]   (OF chain)
    pr, qr: (dst, a, b) base registers; b = None with zero=True adds a zero
    register (%rbp, set to 0).  sub_p / sub_q: add the complement of b
    (the caller sets the carry-in).  Uses rax, rbx, rcx, rdx; rdx is the
    limb index and must be 0 on entry; the chains must be set up on entry.
    The count is split into count % 4 single steps and count / 4 groups,
    read from io[count_slot + 1] and io[count_slot + 2] (precomputed)."""
    def step(off):
        o = 8 * off
        dp, ap, bp = pr
        dq, aq, bq = qr
        emit(f"\tmov\t{o}({ap},%rdx,8), %rax")
        if zero:
            emit("\tadcx\t%rbp, %rax")
        elif sub_p:
            emit(f"\tmov\t{o}({bp},%rdx,8), %rbx")
            emit("\tnot\t%rbx")
            emit("\tadcx\t%rbx, %rax")
        else:
            emit(f"\tadcx\t{o}({bp},%rdx,8), %rax")
        emit(f"\tmov\t%rax, {o}({dp},%rdx,8)")
        emit(f"\tmov\t{o}({aq},%rdx,8), %rax")
        if zero:
            emit("\tadox\t%rbp, %rax")
        elif sub_q:
            emit(f"\tmov\t{o}({bq},%rdx,8), %rbx")
            emit("\tnot\t%rbx")
            emit("\tadox\t%rbx, %rax")
        else:
            emit(f"\tadox\t{o}({bq},%rdx,8), %rax")
        emit(f"\tmov\t%rax, {o}({dq},%rdx,8)")

    rem, loop, done = lab("r"), lab("u"), lab("d")
    emit(f"\tmov\t{8 * count_slot + 8}(%r15), %rcx\t/* count % 4 */")
    emit(f"\tjrcxz\t{rem}e")
    emit(f"{rem}:")
    step(0)
    emit("\tlea\t1(%rdx), %rdx")
    emit("\tlea\t-1(%rcx), %rcx")
    emit(f"\tjrcxz\t{rem}e")
    emit(f"\tjmp\t{rem}")
    emit(f"{rem}e:")
    emit(f"\tmov\t{8 * count_slot + 16}(%r15), %rcx\t/* count / 4 */")
    # jrcxz reaches only 127 bytes: skip the unrolled loop through a jmp
    emit(f"\tjrcxz\t{loop}s")
    emit(f"\tjmp\t{loop}")
    emit(f"{loop}s:")
    emit(f"\tjmp\t{done}")
    emit(f"{loop}:")
    for u in range(UNROLL):
        step(u)
    emit(f"\tlea\t{UNROLL}(%rdx), %rdx")
    emit("\tlea\t-1(%rcx), %rcx")
    emit(f"\tjrcxz\t{done}")
    emit(f"\tjmp\t{loop}")
    emit(f"{done}:")


def save_carries(slot):
    emit("\tsetc\t%al")
    emit("\tseto\t%bl")
    emit("\tmovzbl\t%al, %eax")
    emit("\tmovzbl\t%bl, %ebx")
    emit(f"\tmov\t%rax, {8 * slot}(%r15)")
    emit(f"\tmov\t%rbx, {8 * slot + 8}(%r15)")


def split_count(src_reg, slot):
    """io[slot+1] = src % 4, io[slot+2] = src / 4 (flags clobbered)."""
    emit(f"\tmov\t{src_reg}, %rax")
    emit("\tand\t$3, %eax")
    emit(f"\tmov\t%rax, {8 * slot + 8}(%r15)")
    emit(f"\tmov\t{src_reg}, %rax")
    emit("\tshr\t$2, %rax")
    emit(f"\tmov\t%rax, {8 * slot + 16}(%r15)")


def prologue(name):
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\t5")
    emit(f"{name}:")
    for r in ["rbx", "rbp", "r12", "r13", "r14", "r15"]:
        emit(f"\tpush\t%{r}")


def epilogue(name):
    for r in ["r15", "r14", "r13", "r12", "rbp", "rbx"]:
        emit(f"\tpop\t%{r}")
    emit("\tret")
    emit(f"\t.size\t{name}, .-{name}")


def kara_recomp2():
    """void gtc_kara_recomp2 (mp_ptr P, mp_ptr Q, mp_srcptr uP, mp_srcptr uQ,
                              mp_size_t k, mp_limb_t *io)
    See the header comment in the output."""
    name = "gtc_kara_recomp2"
    prologue(name)
    emit("\tmov\t%rdx, %r12\t\t/* uP */")
    emit("\tmov\t%rcx, %r13\t\t/* uQ */")
    emit("\tmov\t%r8, %r14\t\t/* k */")
    emit("\tmov\t%r9, %r15\t\t/* io */")
    # loop counts: io[11..12] k, io[13..14] nb, io[15..16] k - nb,
    # io[17..18] 2k  (slot s: io[s+1] = % 4, io[s+2] = / 4)
    split_count("%r14", 10)
    emit("\tmov\t(%r15), %rbx\t\t/* nb */")
    split_count("%rbx", 12)
    emit("\tmov\t%r14, %rbx")
    emit("\tsub\t(%r15), %rbx\t\t/* k - nb */")
    split_count("%rbx", 14)
    emit("\tlea\t(%r14,%r14), %rbx\t/* 2k */")
    split_count("%rbx", 16)

    # A: t = a1 + b0, into b0's place
    emit("\tlea\t(%rdi,%r14,8), %r8\t/* a1 of P */")
    emit("\tlea\t(%r8,%r14,8), %r9\t/* b0 of P, then t */")
    emit("\tlea\t(%rsi,%r14,8), %r10\t/* a1 of Q */")
    emit("\tlea\t(%r10,%r14,8), %r11\t/* b0 of Q, then t */")
    emit("\txor\t%edx, %edx\t\t/* index; CF = OF = 0 */")
    dual_pass(("%r9", "%r8", "%r9"), ("%r11", "%r10", "%r11"), 10)
    save_carries(3)
    # B: X1 = t + a0, into a1's place
    emit("\txor\t%edx, %edx")
    dual_pass(("%r8", "%r9", "%rdi"), ("%r10", "%r11", "%rsi"), 10)
    save_carries(5)
    # C: X2 = t + b1 in t's place over nb limbs, then + 0 above
    emit("\tlea\t(%r9,%r14,8), %r8\t/* b1 of P */")
    emit("\tlea\t(%r11,%r14,8), %r10\t/* b1 of Q */")
    emit("\tmov\t$0, %ebp")
    emit("\txor\t%edx, %edx")
    dual_pass(("%r9", "%r9", "%r8"), ("%r11", "%r11", "%r10"), 12)
    dual_pass(("%r9", "%r9", None), ("%r11", "%r11", None), 14, zero=True)
    save_carries(7)
    # D: C[k..3k) += u XOR mask, carry-in mask & 1: one variant per masks
    emit("\tlea\t(%rdi,%r14,8), %r9\t/* P + k */")
    emit("\tlea\t(%rsi,%r14,8), %r11\t/* Q + k */")
    ends = lab("De")
    emit("\tmov\t8(%r15), %r8")
    emit("\tmov\t16(%r15), %r10")
    v = {}
    for sp in (0, 1):
        for sq in (0, 1):
            v[(sp, sq)] = lab(f"D{sp}{sq}_")
    emit("\ttest\t%r8, %r8")
    emit(f"\tjnz\t{lab('Dp')}")
    pl = f".LDp{label_no[0]}"
    emit("\ttest\t%r10, %r10")
    emit(f"\tjnz\t{v[(0, 1)]}")
    emit(f"\tjmp\t{v[(0, 0)]}")
    emit(f"{pl}:")
    emit("\ttest\t%r10, %r10")
    emit(f"\tjnz\t{v[(1, 1)]}")
    emit(f"\tjmp\t{v[(1, 0)]}")
    for (sp, sq), l in v.items():
        emit(f"{l}:")
        emit("\txor\t%edx, %edx\t\t/* CF = OF = 0 */")
        if sq:
            emit("\tmov\t$0x7fffffffffffffff, %rax")
            emit("\tadd\t$1, %rax\t\t/* OF = 1, CF = 0 */")
        if sp:
            emit("\tstc")
        dual_pass(("%r9", "%r9", "%r12"), ("%r11", "%r11", "%r13"), 16,
                  sub_p=bool(sp), sub_q=bool(sq))
        emit(f"\tjmp\t{ends}")
    emit(f"{ends}:")
    save_carries(9)
    epilogue(name)


def rsh1_flagless(name, sub, sh=1):
    """mp_limb_t NAME (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n):
    {rp, n} = ({ap, n} +- {bp, n} mod B^n) >> 1, the sum read as a signed
    n-limb number and shifted arithmetically (GMP's mpn_rsh1add_n and
    mpn_rsh1sub_n put the carry or borrow on top instead), and the bit
    shifted out returned.  n >= 1.  The sum runs along the CF chain; the shift
    uses shrx and shlx and joins the halves with lea, none of which touch
    the flags.  The pointers are advanced (a displacement addresses the
    limb, so that the adc keeps its memory operand fused), and the loop
    counts down with dec, which leaves CF alone; the four sums of an
    unrolled group rotate through four registers, so that no limb is
    copied.  For the Zen, where shrd (rsh1 below) is several micro-ops.  With
    sh > 1 the same shifted by sh bits (the bits shifted out returned)."""
    aorsm = "sbb" if sub else "adc"
    aors = "sub" if sub else "add"
    tag = ("s" if sub else "a") + (str(sh) if sh > 1 else "")
    L = {k: f".Lh{tag}{k}" for k in ("s", "se", "u", "us", "d")}
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\t5")
    emit(f"{name}:")
    for r in ["rbx", "r12", "r13", "r14", "r15"]:
        emit(f"\tpush\t%{r}")
    emit(f"\tmov\t${sh}, %r12d")
    emit(f"\tmov\t${64 - sh}, %r13d")
    emit("\tlea\t-1(%rcx), %r9\t\t/* n - 1 limbs after the first */")
    emit("\tmov\t%r9, %rax")
    emit("\tshr\t$2, %r9\t\t/* groups of 4 */")
    emit("\tand\t$3, %eax\t\t/* single steps */")
    emit("\tmov\t%rax, %rcx")
    emit("\tmov\t(%rsi), %r10")
    emit(f"\t{aors}\t(%rdx), %r10\t/* s0 */")
    emit("\tmov\t%r10, %r8\t\t/* the bit shifted out: s0 & 1 */")
    emit("\tlea\t8(%rsi), %rsi")
    emit("\tlea\t8(%rdx), %rdx")

    def step(off, cur, prev):
        o = 8 * off
        emit(f"\tmov\t{o}(%rsi), {cur}")
        emit(f"\t{aorsm}\t{o}(%rdx), {cur}")
        emit(f"\tshrx\t%r12, {prev}, %r11")
        emit(f"\tshlx\t%r13, {cur}, %r15")
        emit("\tlea\t(%r11,%r15), %r11")
        emit(f"\tmov\t%r11, {o}(%rdi)")

    def advance(k):
        for r in ("%rsi", "%rdx", "%rdi"):
            emit(f"\tlea\t{8 * k}({r}), {r}")

    # single steps: one sum at a time, the previous one in r10
    emit(f"\tjrcxz\t{L['se']}")
    emit(f"{L['s']}:")
    step(0, "%rax", "%r10")
    emit("\tmov\t%rax, %r10")
    advance(1)
    emit("\tdec\t%rcx\t\t\t/* dec leaves CF alone */")
    emit(f"\tjnz\t{L['s']}")
    emit(f"{L['se']}:")
    emit("\tmov\t%r9, %rcx")
    emit(f"\tjrcxz\t{L['d']}")
    emit(f"{L['u']}:")
    regs = ["%rax", "%rbx", "%r14", "%r10"]
    for u in range(4):
        step(u, regs[u], regs[u - 1] if u else "%r10")
    advance(4)
    emit("\tdec\t%rcx")
    emit(f"\tjnz\t{L['u']}")
    emit(f"{L['d']}:")
    # the last limb: the last sum shifted down arithmetically (the halving
    # of a signed value at width n; the carry out is not wanted)
    emit(f"\tsar\t${sh}, %r10")
    emit("\tmov\t%r10, (%rdi)")
    emit("\tmov\t%r8, %rax")
    emit(f"\tand\t${(1 << sh) - 1}, %eax")
    for r in ["r15", "r14", "r13", "r12", "rbx"]:
        emit(f"\tpop\t%{r}")
    emit("\tret")
    emit(f"\t.size\t{name}, .-{name}")


def rsh1(name, sub, sh=1):
    """mp_limb_t NAME (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n):
    {rp, n} = ({ap, n} +- {bp, n} mod B^n) >> 1, the sum read as a signed
    n-limb number and shifted arithmetically (GMP's mpn_rsh1add_n and
    mpn_rsh1sub_n put the carry or borrow on top instead), and the bit
    shifted out returned.  n >= 1.

    The sums run along the CF chain, eight at a time, into nine registers
    (the last sum of the group before, and the eight new ones); then CF is
    saved (sbb), the eight results are formed by shrd, one micro-op each
    on p1, which clobbers the flags, and stored, and the next group
    restores CF (bt).  So a limb costs a load, an adc with its memory
    operand fused, a shrd and a store.  The n - 1 mod 8 limbs before the
    groups go the same way in blocks of 1, 2 and 4 on the bits of n - 1
    (faster than one flags-free step of shrx, shlx and lea per limb).  The
    pointers are advanced (a displacement keeps the
    adc's memory operand fused); the loop counts down with dec, which
    leaves CF alone.  With sh > 1 the same shifted by sh bits (shrd $sh),
    the bits shifted out returned."""
    aorsm = "sbb" if sub else "adc"
    aors = "sub" if sub else "add"
    tag = ("s" if sub else "a") + (str(sh) if sh > 1 else "")
    L = {k: f".Lh{tag}{k}" for k in ("u", "d")}
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\t5")
    emit(f"{name}:")
    for r in ["rbx", "rbp", "r12", "r13", "r14", "r15"]:
        emit(f"\tpush\t%{r}")
    emit("\tlea\t-1(%rcx), %r9\t\t/* n - 1 limbs after the first */")
    emit("\tmov\t%r9, %rcx")
    emit("\tmov\t(%rsi), %r10")
    emit(f"\t{aors}\t(%rdx), %r10\t/* s0 */")
    emit("\tsbb\t%rbp, %rbp\t\t/* CF saved: 0 or -1 */")
    emit("\tmov\t%r10, %r8\t\t/* the bit shifted out: s0 & 1 */")
    emit("\tlea\t8(%rsi), %rsi")
    emit("\tlea\t8(%rdx), %rdx")
    R = ["%r10", "%rax", "%rbx", "%r9", "%r11", "%r12", "%r13", "%r14",
         "%r15"]

    def advance(k):
        for r in ("%rsi", "%rdx", "%rdi"):
            emit(f"\tlea\t{8 * k}({r}), {r}")

    def block(m):
        emit("\tbt\t$0, %rbp\t\t/* CF back */")
        for u in range(m):
            emit(f"\tmov\t{8 * u}(%rsi), {R[u + 1]}")
            emit(f"\t{aorsm}\t{8 * u}(%rdx), {R[u + 1]}")
        emit("\tsbb\t%rbp, %rbp")
        for u in range(m):
            emit(f"\tshrd\t${sh}, {R[u + 1]}, {R[u]}")
            emit(f"\tmov\t{R[u]}, {8 * u}(%rdi)")
        emit(f"\tmov\t{R[m]}, %r10")
        advance(m)

    for bit in (0, 1, 2):
        skip = lab("hb")
        emit(f"\ttest\t${1 << bit}, %cl")
        emit(f"\tjz\t{skip}")
        block(1 << bit)
        emit(f"{skip}:")
    emit("\tshr\t$3, %rcx\t\t/* groups of 8 */")
    emit(f"\tjz\t{L['d']}")
    emit("\t.p2align\t4")
    emit(f"{L['u']}:")
    block(8)
    emit("\tdec\t%rcx")
    emit(f"\tjnz\t{L['u']}")
    emit(f"{L['d']}:")
    # the last limb: the last sum shifted down arithmetically (the halving
    # of a signed value at width n; the carry out is not wanted)
    emit(f"\tsar\t${sh}, %r10")
    emit("\tmov\t%r10, (%rdi)")
    emit("\tmov\t%r8, %rax")
    emit(f"\tand\t${(1 << sh) - 1}, %eax")
    for r in ["r15", "r14", "r13", "r12", "rbp", "rbx"]:
        emit(f"\tpop\t%{r}")
    emit("\tret")
    emit(f"\t.size\t{name}, .-{name}")


def chains(name, doc, ptrs, step, variants, zero_ret, unroll=8, cin=None,
           own_end=None, pre=None):
    """A kernel of two carry chains over n limbs, one loop per sign variant,
    unrolled eight times (or unroll): neither chain may be touched by the
    loop control, so it is lea and jrcxz and a jmp, three micro-ops and two
    branches per turn, besides a lea per pointer, which eight limbs share
    better than four.  The loop is entered at the step that leaves a multiple of the
    unroll behind it, as GMP's kernels do, the pointers moved back
    so that the steps skipped would have read below them: a small n costs a
    jump through a table rather than one loop turn per limb.  The pointers
    in ptrs advance by lea (a displacement from the base register keeps a
    memory operand micro-fused on Intel, which an index would not).  Enters
    with n in %r8 and the signs in %r9d; step(off, nv, nw) emits one limb;
    zero_ret() the return for n = 0, where the carries are the carry-ins.
    cin(nv, nw) gives a variant's carry-ins (CF, OF), by default (nv, nw).
    Ends with CF in eax and OF in ecx, for the caller to combine and return;
    a variant in own_end ends instead with own_end[variant](), emitted after
    its loop, which returns by itself.  pre() emits code at the entry, for a
    kernel whose arguments sit elsewhere; a sign bit no variant uses is
    dispatched as 0.
    Uses rax, rcx and r8..r11 only, all caller-saved."""
    assert unroll & (unroll - 1) == 0
    cin = cin or (lambda nv, nw: (nv, nw))
    own_end = own_end or {}
    emit(f"/* {doc} */")
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\t5")
    emit(f"{name}:")
    if pre:
        pre()
    zero = lab("zero")
    emit("\tmov\t%rcx, %r10\t\t/* the fourth pointer: rcx counts */")
    emit(f"\tlea\t{unroll - 1}(%r8), %rcx")
    emit(f"\tshr\t${unroll.bit_length() - 1}, %rcx"
         "\t\t/* groups, the first one partial */")
    emit("\ttest\t%rcx, %rcx")
    emit(f"\tjz\t{zero}")
    emit("\tneg\t%r8")
    emit(f"\tand\t${unroll - 1}, %r8d\t\t/* the steps the first group skips */")
    emit("\tlea\t(,%r8,8), %r11")
    for p in ptrs:
        emit(f"\tsub\t%r11, {p}")
    labels = {sv: lab(f"v{sv[0]}{sv[1]}_") for sv in variants}
    two = any(nw for nv, nw in variants)
    emit("\ttest\t$1, %r9b")
    one = lab("s1_")
    emit(f"\tjnz\t{one}")
    if two:
        emit("\ttest\t$2, %r9b")
        emit(f"\tjnz\t{labels[(0, 1)]}")
    emit(f"\tjmp\t{labels[(0, 0)]}")
    emit(f"{one}:")
    if two:
        emit("\ttest\t$2, %r9b")
        emit(f"\tjnz\t{labels[(1, 1)]}")
    emit(f"\tjmp\t{labels[(1, 0)]}")
    end = lab("end")
    for (nv, nw), l in labels.items():
        entries = [lab("e") for _ in range(unroll)]
        table, done = lab("t"), lab("d")
        emit(f"{l}:")
        emit(f"\tlea\t{table}(%rip), %rax")
        emit("\tmovslq\t(%rax,%r8,4), %r11")
        emit("\tadd\t%r11, %rax")
        cf, of = cin(nv, nw)
        emit("\txor\t%r11d, %r11d\t\t/* CF = OF = 0 */")
        if of:
            emit("\tmov\t$0x7fffffffffffffff, %r11")
            emit("\tadd\t$1, %r11\t\t/* OF = 1, CF = 0 */")
        if cf:
            emit("\tstc")
        emit("\tjmp\t*%rax")
        for u in range(unroll):
            emit(f"{entries[u]}:")
            step(u, nv, nw)
        for p in ptrs:
            emit(f"\tlea\t{8 * unroll}({p}), {p}")
        emit("\tlea\t-1(%rcx), %rcx")
        emit(f"\tjrcxz\t{done}")
        emit(f"\tjmp\t{entries[0]}")
        emit(f"{done}:")
        if (nv, nw) in own_end:
            own_end[(nv, nw)]()
        else:
            emit(f"\tjmp\t{end}")
        emit("\t.p2align\t2")
        emit(f"{table}:")
        for e in entries:
            emit(f"\t.long\t{e} - {table}")
    emit(f"{zero}:")
    zero_ret()
    emit("\tret")
    emit(f"{end}:")
    emit("\tsetc\t%al")
    emit("\tseto\t%cl")
    emit("\tmovzbl\t%al, %eax")
    emit("\tmovzbl\t%cl, %ecx")


def zero_aors2():
    emit("\tmov\t%r9d, %eax")
    emit("\tand\t$3, %eax\t\t/* the carry-ins: nP + 2 nQ */")


def zero_add3():
    emit("\tmov\t%r9d, %eax")
    emit("\tand\t$1, %eax")
    emit("\tshr\t%r9d")
    emit("\tand\t$1, %r9d")
    emit("\tadd\t%r9d, %eax\t\t/* the carry-ins: nv + nw */")


def aors2():
    """mp_limb_t gtc_aors2_n (mp_ptr P, mp_ptr Q, mp_srcptr xP, mp_srcptr xQ,
                              mp_size_t n, unsigned signs)"""
    def step(off, nv, nw):
        o = 8 * off
        emit(f"\tmov\t{o}(%rdx), %rax")
        if nv:
            emit("\tnot\t%rax")
        emit(f"\tadcx\t{o}(%rdi), %rax")
        emit(f"\tmov\t%rax, {o}(%rdi)")
        emit(f"\tmov\t{o}(%r10), %r9")
        if nw:
            emit("\tnot\t%r9")
        emit(f"\tadox\t{o}(%rsi), %r9")
        emit(f"\tmov\t%r9, {o}(%rsi)")
    chains("gtc_aors2_n",
           "{P, n} += +-{xP, n} along CF, {Q, n} += +-{xQ, n} along OF;\n"
           "   returns CF + 2 OF.  See the header.",
           ["%rdi", "%rsi", "%rdx", "%r10"], step,
           [(0, 0), (0, 1), (1, 0), (1, 1)], zero_aors2)
    emit("\tlea\t(%rax,%rcx,2), %eax")
    emit("\tret")
    emit("\t.size\tgtc_aors2_n, .-gtc_aors2_n")


def add3():
    """mp_limb_t gtc_add3_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp,
                             mp_srcptr wp, mp_size_t n, unsigned signs)

    Unrolled sixteen times: with a subtraction a limb takes six or seven
    micro-ops, and the loop's seven per turn would cost 6-8% at eight
    limbs a turn.  With both subtracted, u - (v + w): the sum along CF,
    its complement added to u along OF with a carry-in of 1, one
    complement per limb (not ~(~u + v + w), two)."""
    def step(off, nv, nw):
        o = 8 * off
        if nv and nw:
            emit(f"\tmov\t{o}(%rdx), %r9")
        else:
            emit(f"\tmov\t{o}(%rsi), %rax")
        if nv and nw:
            # u - (v + w): the sum along CF, its complement added to u
            # along OF with a carry-in of 1: one complement per limb
            emit(f"\tadcx\t{o}(%r10), %r9")
            emit("\tnot\t%r9")
            emit(f"\tadox\t{o}(%rsi), %r9")
            emit(f"\tmov\t%r9, {o}(%rdi)")
            return
        if nv:
            emit(f"\tmov\t{o}(%rdx), %r9")
            emit("\tnot\t%r9")
            emit("\tadcx\t%r9, %rax")
        else:
            emit(f"\tadcx\t{o}(%rdx), %rax")
        if nw:
            emit(f"\tmov\t{o}(%r10), %r9")
            emit("\tnot\t%r9")
            emit("\tadox\t%r9, %rax")
        else:
            emit(f"\tadox\t{o}(%r10), %rax")
        emit(f"\tmov\t%rax, {o}(%rdi)")

    def end_both():
        # v + w = S + c1 B^n (CF), u + ~S + 1 = R + c2 B^n (OF): u - v - w
        # = R + (c2 - 1 - c1) B^n, so the two-complement carry is c2 + 1 - c1
        emit("\tsetc\t%al")
        emit("\tseto\t%cl")
        emit("\tmovzbl\t%al, %eax")
        emit("\tmovzbl\t%cl, %ecx")
        emit("\tneg\t%rax")
        emit("\tlea\t1(%rax,%rcx), %rax")
        emit("\tret")
    chains("gtc_add3_n",
           "{rp, n} = {up, n} +- {vp, n} (CF) +- {wp, n} (OF); returns\n"
           "   CF + OF.  See the header.",
           ["%rdi", "%rsi", "%rdx", "%r10"], step,
           [(0, 0), (0, 1), (1, 0), (1, 1)], zero_add3,
           cin=lambda nv, nw: (0, 1) if nv and nw else (nv, nw),
           own_end={(1, 1): end_both}, unroll=16)
    emit("\tadd\t%rcx, %rax")
    emit("\tret")
    emit("\t.size\tgtc_add3_n, .-gtc_add3_n")


def aorslsh1():
    """mp_limb_t gtc_aorslsh1_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                                 mp_size_t n, unsigned sub)

    {rp, n} = x + 2 y, or x - 2 y with sub & 1; returns the carry, or the
    borrow (0, 1 or 2), as GMP's mpn_addlsh1_n and mpn_sublsh1_n do.  A
    limb of 2 y is y doubled along CF (adcx y, y: the bit shifted out is
    the carry), added along OF (adox), complemented first to subtract, OF's
    carry-in 1 then."""
    def step(off, nv, nw):
        o = 8 * off
        emit(f"\tmov\t{o}(%rdx), %rax")
        emit("\tadcx\t%rax, %rax")
        if nv:
            emit("\tnot\t%rax")
        emit(f"\tadox\t{o}(%rsi), %rax")
        emit(f"\tmov\t%rax, {o}(%rdi)")

    def pre():
        emit("\tmov\t%r8d, %r9d\t\t/* sub */")
        emit("\tmov\t%rcx, %r8\t\t/* n */")

    def zero():
        emit("\txor\t%eax, %eax")

    def end_sub():
        # x + ~(2 y) + 1 = x - 2 y + B^n: the borrow is the doubling's
        # carry plus 1 - OF
        emit("\tsetc\t%al")
        emit("\tseto\t%cl")
        emit("\tmovzbl\t%al, %eax")
        emit("\tmovzbl\t%cl, %ecx")
        emit("\tsub\t%rcx, %rax")
        emit("\tadd\t$1, %rax")
        emit("\tret")
    chains("gtc_aorslsh1_n",
           "{rp, n} = {xp, n} +- 2 {yp, n}; returns the carry or the\n"
           "   borrow.  See the header.",
           ["%rdi", "%rsi", "%rdx"], step, [(0, 0), (1, 0)], zero,
           cin=lambda nv, nw: (0, nv), own_end={(1, 0): end_sub}, pre=pre)
    emit("\tadd\t%rcx, %rax")
    emit("\tret")
    emit("\t.size\tgtc_aorslsh1_n, .-gtc_aorslsh1_n")


def aorsmul1():
    """mp_limb_t gtc_aorsmul_1 (mp_ptr rp, mp_srcptr up, mp_size_t n,
                                mp_limb_t v, unsigned sub)

    {rp, n} += v {up, n}, or -= with sub & 1; returns the high limb or the
    borrow, as GMP's mpn_addmul_1 and mpn_submul_1.  A limb of the product
    is mulx's low half plus the high half before along CF (adcx), added to
    rp along OF (adox), complemented first to subtract, OF's carry-in 1
    then.  Unrolled eight times as chains(), the high halves alternating
    between r10 and r11."""
    emit("/* {rp, n} +-= v {up, n}; returns the high limb or the borrow.")
    emit("   See the header.  */")
    emit("\t.globl\tgtc_aorsmul_1")
    emit("\t.type\tgtc_aorsmul_1, @function")
    emit("\t.p2align\t5")
    emit("gtc_aorsmul_1:")
    zero = lab("zero")
    emit("\tmov\t%rdx, %r9\t\t/* n */")
    emit("\tmov\t%rcx, %rdx\t\t/* v, for mulx */")
    emit("\tlea\t7(%r9), %rcx")
    emit("\tshr\t$3, %rcx\t\t/* groups, the first one partial */")
    emit("\ttest\t%rcx, %rcx")
    emit(f"\tjz\t{zero}")
    emit("\tneg\t%r9")
    emit("\tand\t$7, %r9d\t\t/* the steps the first group skips */")
    emit("\tlea\t(,%r9,8), %rax")
    emit("\tsub\t%rax, %rdi")
    emit("\tsub\t%rax, %rsi")
    one = lab("s1_")
    emit("\ttest\t$1, %r8b")
    emit(f"\tjnz\t{one}")
    for sub in (0, 1):
        if sub:
            emit(f"{one}:")
        entries = [lab("e") for _ in range(8)]
        table, done = lab("t"), lab("d")
        emit(f"\tlea\t{table}(%rip), %rax")
        emit("\tmovslq\t(%rax,%r9,4), %r8")
        emit("\tadd\t%r8, %rax")
        emit("\txor\t%r10d, %r10d")
        emit("\txor\t%r11d, %r11d\t\t/* CF = OF = 0 */")
        if sub:
            emit("\tmov\t$0x7fffffffffffffff, %r8")
            emit("\tadd\t$1, %r8\t\t/* OF = 1, CF = 0 */")
        emit("\tjmp\t*%rax")
        for u in range(8):
            o = 8 * u
            hi, lo = ("%r10", "%r11") if u % 2 == 0 else ("%r11", "%r10")
            emit(f"{entries[u]}:")
            emit(f"\tmulx\t{o}(%rsi), %rax, {hi}")
            emit(f"\tadcx\t{lo}, %rax")
            if sub:
                emit("\tnot\t%rax")
            emit(f"\tadox\t{o}(%rdi), %rax")
            emit(f"\tmov\t%rax, {o}(%rdi)")
        emit("\tlea\t64(%rdi), %rdi")
        emit("\tlea\t64(%rsi), %rsi")
        emit("\tlea\t-1(%rcx), %rcx")
        emit(f"\tjrcxz\t{done}")
        emit(f"\tjmp\t{entries[0]}")
        emit(f"{done}:")
        # the high limb: the last high half plus CF; added, OF on top; or
        # subtracted, as rp + ~h + 1 with OF the carry of rp + ~p: the
        # borrow is h + 1 - OF
        emit("\tmov\t$0, %eax")
        emit("\tadcx\t%r11, %rax")
        emit("\tseto\t%cl")
        emit("\tmovzbl\t%cl, %ecx")
        if sub:
            emit("\tsub\t%rcx, %rax")
            emit("\tadd\t$1, %rax")
        else:
            emit("\tadd\t%rcx, %rax")
        emit("\tret")
        emit("\t.p2align\t2")
        emit(f"{table}:")
        for e in entries:
            emit(f"\t.long\t{e} - {table}")
    emit(f"{zero}:")
    emit("\txor\t%eax, %eax")
    emit("\tret")
    emit("\t.size\tgtc_aorsmul_1, .-gtc_aorsmul_1")


def flag_loop(slot, step, ptrs, unroll):
    """count steps of step(off) along the carry chains, which the loop
    leaves alone: count % unroll single steps, then groups of unroll, the
    two counts in the red zone at slot and slot - 8 (split_rz).  The
    pointers in ptrs advance by lea."""
    rem, grp, done = lab("r"), lab("g"), lab("d")
    emit(f"\tmov\t{slot}(%rsp), %rcx")
    emit(f"\tjrcxz\t{rem}e")
    emit(f"{rem}:")
    step(0)
    for p in ptrs:
        emit(f"\tlea\t8({p}), {p}")
    emit("\tlea\t-1(%rcx), %rcx")
    emit(f"\tjrcxz\t{rem}e")
    emit(f"\tjmp\t{rem}")
    emit(f"{rem}e:")
    emit(f"\tmov\t{slot - 8}(%rsp), %rcx")
    # jrcxz reaches only 127 bytes: past the loop through a jmp
    emit(f"\tjrcxz\t{grp}s")
    emit(f"\tjmp\t{grp}")
    emit(f"{grp}s:")
    emit(f"\tjmp\t{done}")
    emit(f"{grp}:")
    for u in range(unroll):
        step(u)
    for p in ptrs:
        emit(f"\tlea\t{8 * unroll}({p}), {p}")
    emit("\tlea\t-1(%rcx), %rcx")
    emit(f"\tjrcxz\t{done}")
    emit(f"\tjmp\t{grp}")
    emit(f"{done}:")


def split_rz(src, slot, unroll):
    """The counts of flag_loop: src % unroll at slot, src / unroll at
    slot - 8 (flags clobbered, src kept)."""
    emit(f"\tmov\t{src}, %rax")
    emit(f"\tand\t${unroll - 1}, %eax")
    emit(f"\tmov\t%rax, {slot}(%rsp)")
    emit(f"\tmov\t{src}, %rax")
    emit(f"\tshr\t${unroll.bit_length() - 1}, %rax")
    emit(f"\tmov\t%rax, {slot - 8}(%rsp)")


def k2_recomp():
    """void gtc_k2_recomp (mp_ptr rp, mp_srcptr Q, mp_size_t k,
                           mp_size_t s)"""
    name = "gtc_k2_recomp"
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\t5")
    emit(f"{name}:")
    # rdi: rp, r9: c2 = rp + 2k, rsi: Q, all advancing; r8: s, r10: k,
    # r11: 0.  The loop counts in the red zone, all made before the chains
    # start: A k, B 2s - k, B2 2k - 2s, C s + 1.
    emit("\tmov\t%rdx, %r10")
    emit("\tmov\t%rcx, %r8")
    emit("\tlea\t(%rdi,%r10,8), %r9")
    emit("\tlea\t(%r9,%r10,8), %r9\t/* c2 */")
    split_rz("%r10", -8, 8)
    emit("\tlea\t(%r8,%r8), %rdx")
    emit("\tsub\t%r10, %rdx\t\t/* 2s - k */")
    split_rz("%rdx", -24, 8)
    emit("\tlea\t1(%r8), %rdx\t\t/* s + 1 */")
    split_rz("%rdx", -40, 8)
    emit("\tmov\t%r10, %rdx")
    emit("\tsub\t%r8, %rdx")
    emit("\tadd\t%rdx, %rdx\t\t/* 2k - 2s: 0 or 2 */")
    split_rz("%rdx", -56, 4)
    emit("\txor\t%r11d, %r11d\t\t/* CF = OF = 0 */")

    def step_a(off):
        o = 8 * off
        emit(f"\tmov\t{o}(%rdi), %rax")
        emit(f"\tadcx\t{o}(%r9), %rax")
        emit(f"\tmov\t%rax, {o}(%rdi)")

    def step_b(off):
        o = 8 * off
        emit(f"\tmov\t{o}(%rdi), %rax")
        emit(f"\tadcx\t{o}(%r9), %rax")
        emit(f"\tadox\t{o}(%rsi), %rax")
        emit(f"\tmov\t%rax, {o}(%rdi)")

    def step_b2(off):
        o = 8 * off
        emit(f"\tmov\t{o}(%rdi), %rax")
        emit("\tadcx\t%r11, %rax")
        emit(f"\tadox\t{o}(%rsi), %rax")
        emit(f"\tmov\t%rax, {o}(%rdi)")

    def step_c(off):
        o = 8 * off
        emit(f"\tmov\t{o}(%rdi), %rax")
        emit(f"\tadox\t{o}(%rsi), %rax")
        emit(f"\tmov\t%rax, {o}(%rdi)")

    # A: c0 + c2 over the k limbs below c1, along CF.
    flag_loop(-8, step_a, ["%rdi", "%r9"], 8)
    # B: + c1 along OF, over the rest of c2's 2s limbs.
    flag_loop(-24, step_b, ["%rdi", "%r9", "%rsi"], 8)
    # c2's two missing limbs when s < k: the CF carry into them; out of
    # them, at 2k, it vanishes.  c1 goes on along OF.
    flag_loop(-56, step_b2, ["%rdi", "%rsi"], 4)
    # C: c1's other s + 1 limbs along OF, then its carry, which the product
    # absorbs.
    flag_loop(-40, step_c, ["%rdi", "%rsi"], 8)
    done, loop = lab("k2d"), lab("k2p")
    emit(f"\tjno\t{done}")
    emit("\tlea\t-1(%r8), %rcx\t\t/* the s - 1 limbs left */")
    emit(f"{loop}:")
    emit(f"\tjrcxz\t{done}")
    emit("\taddq\t$1, (%rdi)")
    emit(f"\tjnc\t{done}")
    emit("\tlea\t8(%rdi), %rdi")
    emit("\tlea\t-1(%rcx), %rcx")
    emit(f"\tjmp\t{loop}")
    emit(f"{done}:")
    emit("\tret")
    emit(f"\t.size\t{name}, .-{name}")


def rsh1add3(name="gtc_rsh1add3_n", sh=1):
    """mp_limb_t gtc_rsh1add3_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
                                 mp_srcptr cp, mp_size_t n, unsigned comp)

    {rp, n} = (a + b + c mod B^n) >> 1, an arithmetic shift, or with comp
    = 1 (~a + b + c + 1 mod B^n) >> 1, i.e. b + c - a.  n >= 1; rp may
    equal ap, bp or cp.  The halving fused with the three-operand sum
    before it: the sum never goes to memory.  The sums run eight at a time
    into nine registers (the last sum of the group before and the eight
    new ones), b along OF and c along CF (a's complement, where comp, with
    the carry-in on OF); then both flags are saved, the eight results
    formed by shrd, which clobbers them, and stored one limb behind, and
    the next group restores the flags.  They stay in registers, so that
    the chains wait neither on a store forwarded through memory nor on
    each other: OF in a byte F (seto), which an 8-bit add of 0x7f turns
    back into OF, then CF (setc) by clc and an adcx of it to -1, which
    leaves OF alone, with F, free once OF is back, holding the -1.  The add
    leaves CF = 0 too, but the clc keeps CF's chain from waiting on OF's
    restore (2.06 cycles a limb without it, 2.00 with).  The loop ends on a compare of rp with its
    end, where the flags are saved, so that no register counts.  The first group, of n mod 8 sums (or 8),
    is a copy of the sums entered part-way, which then jumps into the
    shifts part-way; the loop's groups run straight through.  With sh > 1
    the sum is shifted by sh bits instead (gtc_rsh2add3_n,
    gtc_rsh3add3_n)."""
    prologue(name)
    R = ["%r11", "%rbx", "%rbp", "%r12", "%r13", "%r14", "%r15", "%r9",
         "%rcx"]
    F, F8 = "%r8", "%r8b"
    ptrs = ["%rdi", "%rsi", "%rdx", "%r10"]
    emit("\tmov\t%rcx, %r10\t\t/* c */")
    emit("\tmov\t%r9d, %ebp\t\t/* comp */")
    emit("\tlea\t7(%r8), %rax")
    emit("\tshr\t$3, %rax\t\t/* groups, the first one partial */")
    emit("\tshl\t$6, %rax")
    emit("\tneg\t%r8")
    emit("\tand\t$7, %r8d\t\t/* the sums the first group skips */")
    emit("\tlea\t(,%r8,8), %rbx")
    for p_ in ptrs:
        emit(f"\tsub\t%rbx, {p_}")
    emit("\tadd\t%rdi, %rax")
    emit("\tmov\t%rax, -16(%rsp)\t\t/* where rp ends */")
    plain, compl, last = lab("h3p"), lab("h3c"), lab("h3e")
    emit("\ttest\t%ebp, %ebp")
    emit(f"\tjnz\t{compl}")
    emit(f"\tjmp\t{plain}")
    for variant, start in ((0, plain), (1, compl)):
        loop = lab("h3l")
        ts, th = lab("h3ts"), lab("h3th")
        e1 = [lab("h3f") for _ in range(8)]
        es = [lab("h3s") for _ in range(8)]
        eh = [lab("h3h") for _ in range(8)]
        hend = lab("h3he")

        def sums(entries):
            for u in range(8):
                emit(f"{entries[u]}:")
                o = 8 * u
                r = R[u + 1]
                emit(f"\tmov\t{o}(%rsi), {r}")
                if variant:
                    emit(f"\tnot\t{r}")
                emit(f"\tadox\t{o}(%rdx), {r}")
                emit(f"\tadcx\t{o}(%r10), {r}")
            emit("\tsetc\t%al\t\t\t/* CF to rax */")
            emit(f"\tseto\t{F8}\t\t\t/* OF to F */")

        emit(f"{start}:")
        emit(f"\tlea\t{th}(%rip), %rbx")
        emit("\tmovslq\t(%rbx,%r8,4), %r9")
        emit("\tadd\t%rbx, %r9")
        emit("\tmov\t%r9, -8(%rsp)\t\t/* where the first shifts start */")
        emit(f"\tlea\t{ts}(%rip), %rbx")
        emit("\tmovslq\t(%rbx,%r8,4), %r9")
        emit("\tadd\t%rbx, %r9")
        emit("\txor\t%eax, %eax\t\t/* CF's register */")
        emit(f"\tmov\t%rbp, {F}")
        emit(f"\tadd\t$0x7f, {F8}\t\t/* OF = the carry-in, CF = 0 */")
        emit("\tjmp\t*%r9")
        # the first group
        sums(e1)
        emit("\tjmp\t*-8(%rsp)")
        emit(f"{loop}:")
        emit(f"\tadd\t$0x7f, {F8}\t\t/* OF back (0x7f + 1 overflows) */")
        emit("\tclc\t\t\t\t/* CF not to wait on OF's restore */")
        emit(f"\tmov\t$-1, {F}")
        emit(f"\tadcx\t%rax, {F}\t\t/* CF back, OF kept */")
        sums(es)
        for u in range(8):
            emit(f"{eh[u]}:")
            emit(f"\tshrd\t${sh}, {R[u + 1]}, {R[u]}")
            emit(f"\tmov\t{R[u]}, {8 * (u - 1)}(%rdi)")
        emit(f"{hend}:")
        emit(f"\tmov\t{R[8]}, {R[0]}")
        for p_ in ptrs:
            emit(f"\tlea\t64({p_}), {p_}")
        emit("\tcmp\t-16(%rsp), %rdi\t\t/* the flags are saved */")
        emit(f"\tjne\t{loop}")
        # the last sum, at n - 1: shifted down arithmetically
        emit(f"\tsar\t${sh}, {R[0]}")
        emit(f"\tmov\t{R[0]}, -8(%rdi)")
        emit(f"\tjmp\t{last}")
        emit("\t.p2align\t2")
        emit(f"{ts}:")
        for e_ in e1:
            emit(f"\t.long\t{e_} - {ts}")
        emit(f"{th}:")
        for e_ in eh[1:] + [hend]:
            emit(f"\t.long\t{e_} - {th}")
    emit(f"{last}:")
    emit("\txor\t%eax, %eax")
    epilogue(name)


def rsh1add3_flagless(name="gtc_rsh1add3_n", sh=1):
    """The same contract as rsh1add3, for the Zen: shrd is several
    micro-ops there (reciprocal throughput 1.5, uops.info), so the shift is
    shrx, shlx and lea, which leave the flags alone, and the two chains run
    through the whole pass with no save or restore.  Each limb's result is
    stored as soon as the next sum is known (one limb behind, after the
    next limb of every operand has been read, so rp may equal an operand).
    The loop counts an index up to 0 (pointers at their ends, one lea and
    jrcxz per four limbs); the n - 1 limbs after the first are entered
    part-way through the first group by a jump table, the sums rotating
    through four registers.  Per limb: three loads, a store, two flag adds
    (ALU0-2), shrx and shlx (ALU1-2), lea (any ALU): about 1.33 cycles
    from the three AGUs and the three flag-capable ALUs.  With sh > 1
    shifted by sh bits."""
    R = ["%rax", "%rbx", "%r8", "%r9"]
    T1, T2 = "%r11", "%r14"
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\t5")
    emit(f"{name}:")
    for r in ["rbx", "r12", "r13", "r14"]:
        emit(f"\tpush\t%{r}")
    emit("\tmov\t%rcx, %r10\t\t/* c */")
    emit(f"\tmov\t${sh}, %r12d")
    emit(f"\tmov\t${64 - sh}, %r13d")
    emit("\tlea\t2(%r8), %rax")
    emit("\tshr\t$2, %rax")
    emit("\tshl\t$2, %rax\t\t/* 4 g, g = ceil((n - 1) / 4) groups */")
    emit("\tlea\t1(%rax), %r11")
    emit("\tsub\t%r8, %r11\t\t/* u0 = 4 g - n + 1, the step entered */")
    emit("\tmov\t(%rsi), %rbx\t\t/* a0 */")
    emit(f"\tmov\t(%rdx), {T2}\t\t/* b0 */")
    emit("\tmov\t(%r10), %rcx\t\t/* c0 */")
    for p_ in ("%rsi", "%rdx", "%r10", "%rdi"):
        emit(f"\tlea\t({p_},%r8,8), {p_}\t/* at its end */")
    emit("\tneg\t%rax\t\t\t/* the index, -4 g */")
    done1, out_ = lab("hfo1"), lab("hfo")
    comp = lab("hfc")
    emit("\ttest\t%r9d, %r9d")
    emit(f"\tjnz\t{comp}")
    for variant in (0, 1):
        if variant:
            emit(f"{comp}:")
        table, done = lab("hft"), lab("hfd")
        L = [lab("hfl") for _ in range(4)]
        stubs = [lab("hfs") for _ in range(4)]
        emit(f"\tlea\t{table}(%rip), %r8")
        emit(f"\tmovslq\t(%r8,{T1},4), {T1}")
        emit(f"\tadd\t%r8, {T1}")
        if variant:
            emit("\tnot\t%rbx")
            emit("\tmov\t$0x7f, %r8d")
            emit("\tadd\t$1, %r8b\t\t/* OF = 1 (the carry-in), CF = 0 */")
        else:
            emit("\txor\t%r8d, %r8d\t\t/* CF = OF = 0 */")
        emit(f"\tadox\t{T2}, %rbx")
        emit("\tadcx\t%rcx, %rbx\t\t/* s0 */")
        emit("\tmov\t%rax, %rcx")
        z = lab("hfz")
        emit(f"\tjrcxz\t{z}")
        emit(f"\tjmp\t*{T1}")
        emit(f"{z}:")
        emit(f"\tjmp\t{done1}")
        for u in range(4):
            emit(f"{stubs[u]}:")
            prev = R[(u - 1) % 4]
            if prev != "%rbx":
                emit(f"\tmov\t%rbx, {prev}")
            emit(f"\tjmp\t{L[u]}")
        for u in range(4):
            emit(f"{L[u]}:")
            o = 8 * u
            cur, prev = R[u], R[(u - 1) % 4]
            emit(f"\tmov\t{o}(%rsi,%rcx,8), {cur}")
            if variant:
                emit(f"\tnot\t{cur}")
            emit(f"\tadox\t{o}(%rdx,%rcx,8), {cur}")
            emit(f"\tadcx\t{o}(%r10,%rcx,8), {cur}")
            emit(f"\tshrx\t%r12, {prev}, {T1}")
            emit(f"\tshlx\t%r13, {cur}, {T2}")
            emit(f"\tlea\t({T1},{T2}), {T1}")
            emit(f"\tmov\t{T1}, {o - 8}(%rdi,%rcx,8)")
        emit("\tlea\t4(%rcx), %rcx")
        emit(f"\tjrcxz\t{done}")
        emit(f"\tjmp\t{L[0]}")
        emit(f"{done}:")
        emit(f"\tsar\t${sh}, {R[3]}")
        emit(f"\tmov\t{R[3]}, -8(%rdi)")
        emit(f"\tjmp\t{out_}")
        emit("\t.p2align\t2")
        emit(f"{table}:")
        for s_ in stubs:
            emit(f"\t.long\t{s_} - {table}")
    emit(f"{done1}:")
    emit(f"\tsar\t${sh}, %rbx")
    emit("\tmov\t%rbx, -8(%rdi)")
    emit(f"{out_}:")
    emit("\txor\t%eax, %eax")
    for r in ["r14", "r13", "r12", "rbx"]:
        emit(f"\tpop\t%{r}")
    emit("\tret")
    emit(f"\t.size\t{name}, .-{name}")


def aors_n(name, sub):
    """mp_limb_t NAME (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n):
    GMP's mpn_add_n / mpn_sub_n, n >= 1, rp may equal up or vp.  One carry
    chain (adc/sbb), eight limbs a turn, the first turn entered part-way
    through a table (as GMP's coreihwl aors_n), the pointers moved back so
    that the skipped steps would have read below them; the loop counts
    down with dec, which leaves CF alone.  Called directly, not through
    the PLT and the fat binary's CPU vector (gtc.c, GTC_ADD_N)."""
    op = "sbb" if sub else "adc"
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\t5")
    emit(f"{name}:")
    tab, loop = lab("ast"), lab("asl")
    ent = [lab("ase") for _ in range(8)]
    emit("\tlea\t7(%rcx), %r8")
    emit("\tshr\t$3, %r8\t\t\t/* turns, the first one partial */")
    emit("\tneg\t%rcx")
    emit("\tand\t$7, %ecx\t\t/* the steps the first turn skips */")
    emit("\tlea\t(,%rcx,8), %rax")
    emit("\tsub\t%rax, %rdi")
    emit("\tsub\t%rax, %rsi")
    emit("\tsub\t%rax, %rdx")
    emit(f"\tlea\t{tab}(%rip), %rax")
    emit("\tmovslq\t(%rax,%rcx,4), %rcx")
    emit("\tadd\t%rax, %rcx")
    emit("\tclc")
    emit("\tjmp\t*%rcx")
    regs = ["%rax", "%r9", "%r10", "%r11"]
    emit("\t.p2align\t4")
    for u in range(8):
        r = regs[u % 4]
        emit(f"{ent[u]}:")
        emit(f"\tmov\t{8 * u}(%rsi), {r}")
        emit(f"\t{op}\t{8 * u}(%rdx), {r}")
        emit(f"\tmov\t{r}, {8 * u}(%rdi)")
    emit("\tlea\t64(%rsi), %rsi")
    emit("\tlea\t64(%rdx), %rdx")
    emit("\tlea\t64(%rdi), %rdi")
    emit("\tdec\t%r8")
    emit(f"\tjnz\t{ent[0]}")
    emit("\tmov\t$0, %eax")
    emit("\tadc\t%eax, %eax")
    emit("\tret")
    emit("\t.p2align\t2")
    emit(f"{tab}:")
    for e_ in ent:
        emit(f"\t.long\t{e_} - {tab}")
    emit(f"\t.size\t{name}, .-{name}")



def div15_lsh(BLK):
    """mp_limb_t gtc_div15_lsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                                  mp_srcptr zp, mp_size_t n, unsigned signs)

    {rp, n} = (x + 4 y + 8 z) / 15 modulo B^n, the division exact; signs
    & 1 (& 2) subtracts 4 y (8 z) instead.  Blocks of BLK limbs, the
    first one partial (entered through a table, the pointers moved back):
    the sums along CF and OF into rp, the shifted limbs made by shrx and lea
    (which leave the flags alone), the flags parked; then GMP's
    mpn_bdiv_dbm1c over the block, in place, along sub and sbb.  Returns
    the final h of mpn_bdiv_dbm1c.  n >= 0; rp may equal xp.  The flags
    are parked in two instructions, OF by seto into a byte and CF by sbb
    into a mask, and come back in three, an add of 0x7f to the byte (OF,
    and CF 0) and an adcx of the mask to 1 (CF alone), not six and five
    (setc, seto, movzbl and neg twice; adcx and adox of the masks).  The
    sums kept in registers instead, as div5_lsh_reg, were slower: four
    registers are left for them, and blocks of 4 limbs pay for the loop
    and the parking."""
    name = "gtc_div15_lsh_n"
    LG = BLK.bit_length() - 1
    emit(f"/* {name}: see the header.  */")
    prologue(name)
    emit("\tsub\t$8, %rsp\t\t/* the entry of the next block's division */")
    emit("\tmov\t%rdx, %r10")
    emit("\tmov\t%rcx, %r11")
    zero, end = lab("qz"), lab("qe")
    emit(f"\tlea\t{BLK - 1}(%r8), %rcx")
    emit(f"\tshr\t${LG}, %rcx\t\t/* blocks, the first one partial */")
    emit("\ttest\t%rcx, %rcx")
    emit(f"\tjz\t{zero}")
    emit("\tneg\t%r8")
    emit(f"\tand\t${BLK - 1}, %r8d\t\t/* the limbs the first block skips */")
    emit("\tlea\t(,%r8,8), %rax")
    for p_ in ("%rdi", "%rsi", "%r10", "%r11"):
        emit(f"\tsub\t%rax, {p_}")
    emit("\txor\t%r12d, %r12d\t\t/* h */")
    for r in ("%r13d", "%r14d", "%r15d", "%ebp"):
        emit(f"\txor\t{r}, {r}\t\t/* the limbs below y and z: 0 */")
    variants = ((0, 0), (0, 1), (1, 0), (1, 1))
    labels = {v: lab(f"qv{v[0]}{v[1]}_") for v in variants}
    one = lab("qs1_")
    emit("\ttest\t$1, %r9b")
    emit(f"\tjnz\t{one}")
    emit("\ttest\t$2, %r9b")
    emit(f"\tjnz\t{labels[(0, 1)]}")
    emit(f"\tjmp\t{labels[(0, 0)]}")
    emit(f"{one}:")
    emit("\ttest\t$2, %r9b")
    emit(f"\tjnz\t{labels[(1, 1)]}")
    emit(f"\tjmp\t{labels[(1, 0)]}")
    Y, Z = ("%r13", "%r14"), ("%r15", "%rbp")
    for (ny, nz), l in labels.items():
        ea = [lab("qa") for _ in range(BLK)]
        eb = [lab("qb") for _ in range(BLK)]
        ta, tb = lab("qta"), lab("qtb")
        emit(f"{l}:")
        emit(f"\tlea\t{tb}(%rip), %rax")
        emit("\tmovslq\t(%rax,%r8,4), %rbx")
        emit("\tadd\t%rbx, %rax")
        emit("\tmov\t%rax, (%rsp)")
        emit(f"\tlea\t{ta}(%rip), %rax")
        emit("\tmovslq\t(%rax,%r8,4), %rbx")
        emit("\tadd\t%rbx, %rax")
        emit("\tmov\t$62, %edx")
        emit("\tmov\t$61, %r9d")
        emit("\txor\t%ebx, %ebx\t\t/* CF = OF = 0 */")
        if nz:
            emit("\tmov\t$0x7fffffffffffffff, %rbx")
            emit("\tadd\t$1, %rbx\t\t/* OF = 1, CF = 0 */")
        if ny:
            emit("\tstc")
        emit("\tjmp\t*%rax")

        def sums(entries):
            for u in range(BLK):
                o = 8 * u
                yc, yp = Y[u % 2], Y[(u + 1) % 2]
                zc, zp = Z[u % 2], Z[(u + 1) % 2]
                if entries:
                    emit(f"{ea[u]}:")
                emit(f"\tmov\t{o}(%r10), {yc}")
                emit(f"\tshrx\t%rdx, {yp}, %rax")
                emit(f"\tlea\t(%rax,{yc},4), %rax\t/* 4 y, limb {u} */")
                if ny:
                    emit("\tnot\t%rax")
                emit(f"\tmov\t{o}(%rsi), %r8")
                emit("\tadcx\t%rax, %r8")
                emit(f"\tmov\t{o}(%r11), {zc}")
                emit(f"\tshrx\t%r9, {zp}, %rbx")
                emit(f"\tlea\t(%rbx,{zc},8), %rbx\t/* 8 z */")
                if nz:
                    emit("\tnot\t%rbx")
                emit("\tadox\t%rbx, %r8")
                emit(f"\tmov\t%r8, {o}(%rdi)")
            # park the chains: OF as 0 or 1 in bl, CF as 0 or ~0 in r9
            emit("\tseto\t%bl")
            emit("\tsbb\t%r9, %r9")
            emit("\tmovabs\t$0x1111111111111111, %rdx\t/* (B-1)/15 */")

        def divs(entries, keep_entry=False):
            for u in range(BLK):
                o = 8 * u
                if entries:
                    emit(f"{eb[u]}:")
                emit(f"\tmulx\t{o}(%rdi), %rax, %r8")
                emit("\tsub\t%rax, %r12")
                emit(f"\tmov\t%r12, {o}(%rdi)")
                emit("\tsbb\t%r8, %r12")
            if keep_entry:
                emit(f"\tlea\t{eb[0]}(%rip), %rax")
                emit("\tmov\t%rax, (%rsp)")
            for p_ in ("%rdi", "%rsi", "%r10", "%r11"):
                emit(f"\tlea\t{8 * BLK}({p_}), {p_}")

        def restore():
            # the chains back: 0x7f + 1 overflows a byte, and leaves CF 0;
            # then 1 + ~0 carries, 1 + 0 does not (adcx: CF alone)
            emit("\tadd\t$0x7f, %bl")
            emit("\tmov\t$1, %eax")
            emit("\tadcx\t%r9, %rax")
            emit("\tmov\t$62, %edx")
            emit("\tmov\t$61, %r9d")

        # the first, partial block through the tables, then a loop
        # with neither
        steady = lab("qst")
        sums(True)
        emit("\tjmp\t*(%rsp)")
        divs(True)
        emit("\tdec\t%rcx")
        emit(f"\tjz\t{end}")
        emit("\t.p2align\t4")
        emit(f"{steady}:")
        restore()
        sums(False)
        divs(False)
        emit("\tdec\t%rcx")
        emit(f"\tjnz\t{steady}")
        emit(f"\tjmp\t{end}")
        for t_, es in ((ta, ea), (tb, eb)):
            emit("\t.p2align\t2")
            emit(f"{t_}:")
            for e_ in es:
                emit(f"\t.long\t{e_} - {t_}")
    emit(f"{zero}:")
    emit("\txor\t%r12d, %r12d")
    emit(f"{end}:")
    emit("\tmov\t%r12, %rax")
    emit("\tadd\t$8, %rsp")
    epilogue(name)


def div5_lsh_reg():
    """mp_limb_t gtc_div5_lsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                                 mp_size_t n, unsigned sub)

    {rp, n} = (x + 4 y) / 5 modulo B^n, the division exact; sub & 1
    subtracts 4 y instead.  Blocks of 4 limbs, each summed along CF into
    four registers (adc, or sbb with sub & 1, the shifted limbs of 4 y by
    shrx and lea), the chain parked by sbb and restored by neg, then GMP's
    mpn_bdiv_dbm1c step with (B-1)/5 over the four, stored once.  The n
    mod 4 limbs go first, by the same step over one limb.  The loops end
    on rp reaching its end, kept in memory.  Returns the final h of
    mpn_bdiv_dbm1c.  n >= 0; rp may equal xp."""
    name = "gtc_div5_lsh_n"
    emit(f"/* {name}: see the header (the sum in registers).  */")
    prologue(name)
    emit("\tsub\t$16, %rsp\t\t/* (%rsp): blocks of 4, 8(%rsp): the rest */")
    emit("\tmov\t%rdx, %r10\t\t/* yp */")
    # (%rsp): rp + 8n, the end; 8(%rsp): rp + 8 (n mod 4), the rest's
    emit("\tlea\t(%rdi,%rcx,8), %rax")
    emit("\tmov\t%rax, (%rsp)")
    emit("\tand\t$3, %ecx")
    emit("\tlea\t(%rdi,%rcx,8), %rax")
    emit("\tmov\t%rax, 8(%rsp)")
    emit("\tmovabs\t$0x3333333333333333, %rdx\t/* (B-1)/5 */")
    emit("\tmov\t$62, %r15d")
    emit("\txor\t%r12d, %r12d\t\t/* h */")
    emit("\txor\t%r13d, %r13d\t\t/* the limb below y: 0 */")
    emit("\txor\t%r9d, %r9d\t\t/* the sum's chain, parked: 0 or ~0 */")
    end = lab("d5e")
    paths = {0: lab("d5a"), 1: lab("d5s")}
    emit("\ttest\t$1, %r8b")
    emit(f"\tjnz\t{paths[1]}")
    S = ["%rbx", "%rbp", "%r11", "%rcx"]
    for sub, l in paths.items():
        op = "sbb" if sub else "adc"
        one, four, l4 = lab("d5o"), lab("d5f"), lab("d5g")
        emit(f"{l}:")
        emit("\tcmp\t8(%rsp), %rdi")
        emit(f"\tje\t{l4}")
        emit(f"{one}:")
        emit("\tneg\t%r9")
        emit("\tmov\t(%r10), %r14")
        emit("\tshrx\t%r15, %r13, %rax")
        emit("\tlea\t(%rax,%r14,4), %rax\t/* 4 y */")
        emit("\tmov\t(%rsi), %rbx")
        emit(f"\t{op}\t%rax, %rbx")
        emit("\tsbb\t%r9, %r9")
        emit("\tmov\t%r14, %r13")
        emit("\tmulx\t%rbx, %rax, %r8")
        emit("\tsub\t%rax, %r12")
        emit("\tmov\t%r12, (%rdi)")
        emit("\tsbb\t%r8, %r12")
        for p_ in ("%rsi", "%r10", "%rdi"):
            emit(f"\tlea\t8({p_}), {p_}")
        emit("\tcmp\t8(%rsp), %rdi")
        emit(f"\tjne\t{one}")
        emit(f"{l4}:")
        emit("\tcmp\t(%rsp), %rdi")
        emit(f"\tje\t{end}")
        emit(f"{four}:")
        emit("\tneg\t%r9")
        Y = ("%r14", "%r13")
        for u in range(4):
            yc, yp = Y[u % 2], Y[(u + 1) % 2]
            emit(f"\tmov\t{8 * u}(%r10), {yc}")
            emit(f"\tshrx\t%r15, {yp}, %rax")
            emit(f"\tlea\t(%rax,{yc},4), %rax\t/* 4 y, limb {u} */")
            emit(f"\tmov\t{8 * u}(%rsi), {S[u]}")
            emit(f"\t{op}\t%rax, {S[u]}")
        emit("\tsbb\t%r9, %r9")
        for u in range(4):
            emit(f"\tmulx\t{S[u]}, %rax, %r8")
            emit("\tsub\t%rax, %r12")
            emit(f"\tmov\t%r12, {8 * u}(%rdi)")
            emit("\tsbb\t%r8, %r12")
        for p_ in ("%rsi", "%r10", "%rdi"):
            emit(f"\tlea\t32({p_}), {p_}")
        emit("\tcmp\t(%rsp), %rdi")
        emit(f"\tjne\t{four}")
        emit(f"\tjmp\t{end}")
    emit(f"{end}:")
    emit("\tmov\t%r12, %rax")
    emit("\tadd\t$16, %rsp")
    epilogue(name)



def div5_rlsh_shld():
    """mp_limb_t gtc_div5_rlsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                                  mp_size_t n, unsigned rsb)

    {rp, n} = (x + 4 y) / 5 modulo B^n, the division exact; rsb & 1:
    (4 y - x) / 5.  div5_lsh_reg with the limbs of 4 y made by shld $2
    before the chain's CF is restored (shld writes the flags), x then
    added (adc) or subtracted from them (sbb) as a memory operand: one
    µop a limb fewer.  The loops end on rp reaching its end, kept in a
    register (r14 for the n mod 4 limbs, r15 for the blocks).  SHLD is
    slow on the Zen (4 µops): for the Intel cores."""
    name = "gtc_div5_rlsh_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name)
    emit("	mov	%rdx, %r10		/* yp */")
    emit("	lea\t(%rdi,%rcx,8), %r15\t/* rp's end */")
    emit("	and\t$3, %ecx")
    emit("	lea\t(%rdi,%rcx,8), %r14\t/* the end of the n mod 4 limbs */")
    emit("	movabs\t$0x3333333333333333, %rdx\t/* (B-1)/5 */")
    emit("	xor\t%r12d, %r12d\t\t/* h */")
    emit("	xor\t%r13d, %r13d\t\t/* the limb below y: 0 */")
    emit("	xor\t%r9d, %r9d\t\t/* the sum's chain, parked: 0 or ~0 */")
    end = lab("r5e")
    paths = {0: lab("r5a"), 1: lab("r5s")}
    emit("	test\t$1, %r8b")
    emit(f"\tjnz\t{paths[1]}")
    S = ["%rbx", "%rbp", "%r11", "%rcx"]
    for sub, l in paths.items():
        op = "sbb" if sub else "adc"
        one, four, l4 = lab("r5o"), lab("r5f"), lab("r5g")
        emit(f"{l}:")
        emit("\tcmp\t%r14, %rdi")
        emit(f"\tje\t{l4}")
        emit(f"{one}:")
        emit("\tmov\t(%r10), %rbx")
        emit("\tmov\t%rbx, %rbp")
        emit("\tshld\t$2, %r13, %rbx\t/* 4 y */")
        emit("\tmov\t%rbp, %r13")
        emit("\tneg\t%r9")
        emit(f"\t{op}\t(%rsi), %rbx")
        emit("\tsbb\t%r9, %r9")
        emit("\tmulx\t%rbx, %rax, %r8")
        emit("\tsub\t%rax, %r12")
        emit("\tmov\t%r12, (%rdi)")
        emit("\tsbb\t%r8, %r12")
        for p_ in ("%rsi", "%r10", "%rdi"):
            emit(f"\tlea\t8({p_}), {p_}")
        emit("\tcmp\t%r14, %rdi")
        emit(f"\tjne\t{one}")
        emit(f"{l4}:")
        emit("\tcmp\t%r15, %rdi")
        emit(f"\tje\t{end}")
        emit(f"{four}:")
        for u in range(4):
            emit(f"\tmov\t{8 * u}(%r10), {S[u]}")
        emit("\tmov\t%rcx, %rax\t\t/* y3, the next block's limb below */")
        for u in (3, 2, 1):
            emit(f"\tshld\t$2, {S[u - 1]}, {S[u]}")
        emit("\tshld\t$2, %r13, %rbx")
        emit("\tmov\t%rax, %r13")
        emit("\tneg\t%r9")
        for u in range(4):
            emit(f"\t{op}\t{8 * u}(%rsi), {S[u]}")
        emit("\tsbb\t%r9, %r9")
        for u in range(4):
            emit(f"\tmulx\t{S[u]}, %rax, %r8")
            emit("\tsub\t%rax, %r12")
            emit(f"\tmov\t%r12, {8 * u}(%rdi)")
            emit("\tsbb\t%r8, %r12")
        for p_ in ("%rsi", "%r10", "%rdi"):
            emit(f"\tlea\t32({p_}), {p_}")
        emit("\tcmp\t%r15, %rdi")
        emit(f"\tjne\t{four}")
        emit(f"\tjmp\t{end}")
    emit(f"{end}:")
    emit("\tmov\t%r12, %rax")
    epilogue(name)

def div5_add_reg():
    """mp_limb_t gtc_div5_add_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                                 mp_size_t n, unsigned mode)

    {rp, n} = (x + y) / 5 modulo B^n, exact; mode 1: (x - y) / 5, mode 2:
    (y - x) / 5.  div5_lsh_reg without the shift of y: blocks of 4 limbs
    summed along CF (adc, or sbb with x or y first) into four registers
    with the second operand from memory, the chain parked by sbb and
    restored by neg, then GMP's mpn_bdiv_dbm1c step with (B-1)/5 over the
    four.  The n mod 4 limbs go first, one at a time.  Returns the final
    h.  n >= 0; rp may equal xp or yp.  The Gaussian Toom-5's q5, q6, q7
    from x already divided by 4 (gtc.c, reg_div5_add)."""
    name = "gtc_div5_add_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name)
    emit("\tmov\t%rcx, %r13")
    emit("\tshr\t$2, %r13\t\t/* blocks of 4 */")
    emit("\tmov\t%rcx, %r15")
    emit("\tand\t$3, %r15d\t\t/* the rest */")
    emit("\tmovabs\t$0x3333333333333333, %r10\t/* (B-1)/5 */")
    emit("\txor\t%r12d, %r12d\t\t/* h */")
    emit("\txor\t%r9d, %r9d\t\t/* the sum's chain, parked: 0 or ~0 */")
    emit("\tmov\t%rdx, %r14\t\t/* yp */")
    emit("\tmov\t%r10, %rdx")
    end = lab("a5e")
    paths = {0: lab("a5a"), 1: lab("a5s"), 2: lab("a5r")}
    emit("\tcmp\t$1, %r8d")
    emit(f"\tje\t{paths[1]}")
    emit(f"\tja\t{paths[2]}")
    S = ["%rbx", "%rbp", "%r11", "%rcx"]
    for mode, l in paths.items():
        op = "adc" if mode == 0 else "sbb"
        first, second = ("%r14", "%rsi") if mode == 2 else ("%rsi", "%r14")
        one, four, l4 = lab("a5o"), lab("a5f"), lab("a5g")
        emit(f"{l}:")
        emit("\ttest\t%r15, %r15")
        emit(f"\tje\t{l4}")
        emit(f"{one}:")
        emit("\tneg\t%r9")
        emit(f"\tmov\t({first}), %rbx")
        emit(f"\t{op}\t({second}), %rbx")
        emit("\tsbb\t%r9, %r9")
        emit("\tmulx\t%rbx, %rax, %r8")
        emit("\tsub\t%rax, %r12")
        emit("\tmov\t%r12, (%rdi)")
        emit("\tsbb\t%r8, %r12")
        for p_ in ("%rsi", "%r14", "%rdi"):
            emit(f"\tlea\t8({p_}), {p_}")
        emit("\tdec\t%r15")
        emit(f"\tjnz\t{one}")
        emit(f"{l4}:")
        emit("\ttest\t%r13, %r13")
        emit(f"\tje\t{end}")
        emit("\t.p2align\t4")
        emit(f"{four}:")
        emit("\tneg\t%r9")
        for u in range(4):
            emit(f"\tmov\t{8 * u}({first}), {S[u]}")
            emit(f"\t{op}\t{8 * u}({second}), {S[u]}")
        emit("\tsbb\t%r9, %r9")
        for u in range(4):
            emit(f"\tmulx\t{S[u]}, %rax, %r8")
            emit("\tsub\t%rax, %r12")
            emit(f"\tmov\t%r12, {8 * u}(%rdi)")
            emit("\tsbb\t%r8, %r12")
        for p_ in ("%rsi", "%r14", "%rdi"):
            emit(f"\tlea\t32({p_}), {p_}")
        emit("\tdec\t%r13")
        emit(f"\tjnz\t{four}")
        emit(f"\tjmp\t{end}")
    emit(f"{end}:")
    emit("\tmov\t%r12, %rax")
    epilogue(name)


def addlsh123(modes):
    """mp_limb_t gtc_addlsh123_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
                                  mp_srcptr cp, mp_srcptr dp, mp_size_t n)

    {rp, n} = a + 2 b + 4 c + 8 d; returns the limb above (at most 14).
    By Horner's rule, a + 2 (b + 2 (c + 2 d)), in blocks of eight limbs
    (the n mod 8 first, by n's bits, in blocks of 1, 2 and 4), each held in
    registers through the three steps t <- x + 2 t.  A step doubles t in
    one of two ways, modes[k - 1] for step k:
      'a': along CF, adcx t, t (no shift: the bit shifted out is the
           carry), x added along OF (adox): two carry chains;
      's': first, flag-free, shld $1 from the limb below, the limbs from
           the top down (the block's top limb parked for the next block's
           first, in memory), then x added along CF: one chain, the shift
           on another port than adcx and adox.
    Each step restores its chains at its start and parks them at its end:
    CF alone by setnc to a byte, back by cmpb $1; CF and OF by seto to a
    byte and sbb to a mask, back by cmpb $0x81 (OF) and adcx of the mask
    into 0 (CF).  n >= 1; rp may equal ap.  The Gaussian Toom-4's
    evaluation at 2.  'ass' on the Intel cores, 'aaa' on the Zen, where
    shld is several micro-ops."""
    name = "gtc_addlsh123_n"
    emit(f"/* {name}: see the header ({modes}).  */")
    prologue(name)
    T = ["%rax", "%rbx", "%rbp", "%r10", "%r11", "%r12", "%r13", "%r14"]
    TMP, N = "%r15", "%r9"
    RP, A0, A1, A2, A3 = "%rdi", "%rsi", "%rdx", "%rcx", "%r8"
    SRC = {1: A2, 2: A1, 3: A0}
    # stack: the parked chains, CF of step k at 16 (k - 1), OF at 16 k - 8;
    # the limb below the block of a step that shifts, at 40 + 8 k
    CF = {1: 0, 2: 16, 3: 32}
    OF = {1: 8, 2: 24, 3: 40}
    PRV = {1: 48, 2: 56, 3: 64}

    def save(k, of):			# leaves CF = OF = 0
        if of:
            emit(f"\tseto\t{OF[k]}(%rsp)")
            emit(f"\tsbb\t{TMP}, {TMP}")
            emit(f"\tmov\t{TMP}, {CF[k]}(%rsp)\t/* -CF */")
        else:
            emit(f"\tsetnc\t{CF[k]}(%rsp)\t/* 1 - CF */")

    def restore(k, of):
        if of:
            # OF = [b - (-127) overflows] = b, CF = 1; then CF = [0 +
            # mask + 1 carries] = [mask = ~0], OF left alone
            emit(f"\tcmpb\t$0x81, {OF[k]}(%rsp)")
            emit(f"\tmov\t$0, {TMP}")
            emit(f"\tadcx\t{CF[k]}(%rsp), {TMP}")
        else:
            emit(f"\tcmpb\t$1, {CF[k]}(%rsp)\t/* CF = [1 - CF < 1] */")

    def block(m):
        for u in range(m):
            emit(f"\tmov\t{8 * u}({A3}), {T[u]}")
        for k in (1, 2, 3):
            if modes[k - 1] == 'a':
                restore(k, True)
                for u in range(m):
                    emit(f"\tadcx\t{T[u]}, {T[u]}")
                    emit(f"\tadox\t{8 * u}({SRC[k]}), {T[u]}")
                save(k, True)
            else:
                emit(f"\tmov\t{PRV[k]}(%rsp), {TMP}\t/* the limb below */")
                emit(f"\tmov\t{T[m - 1]}, {PRV[k]}(%rsp)")
                for u in range(m - 1, -1, -1):
                    emit(f"\tshld\t$1, {T[u - 1] if u else TMP}, {T[u]}")
                restore(k, False)
                for u in range(m):
                    emit(f"\tadcx\t{8 * u}({SRC[k]}), {T[u]}")
                save(k, False)
        for u in range(m):
            emit(f"\tmov\t{T[u]}, {8 * u}({RP})")
        for p_ in (RP, A0, A1, A2, A3):
            emit(f"\tlea\t{8 * m}({p_}), {p_}")

    emit("\tsub\t$72, %rsp")
    for o in range(0, 72, 8):
        emit(f"\tmovq\t$0, {o}(%rsp)")
    for k in (1, 2, 3):
        if modes[k - 1] == 's':
            emit(f"\tmovq\t$1, {CF[k]}(%rsp)\t/* 1 - CF */")
    for bit in (0, 1, 2):
        skip = lab("hs")
        emit(f"\ttest\t${1 << bit}, %r9b")
        emit(f"\tjz\t{skip}")
        block(1 << bit)
        emit(f"{skip}:")
    loop, done = lab("hll"), lab("hle")
    emit(f"\tshr\t$3, {N}")
    emit(f"\tjz\t{done}")
    emit("\t.p2align\t4")
    emit(f"{loop}:")
    block(8)
    emit(f"\tsub\t$1, {N}")
    emit(f"\tjnz\t{loop}")
    emit(f"{done}:")
    # the limb above: 2 (2 top1 + dbl2 + add2) + dbl3 + add3, where a
    # step's doubling carried CF ('a') or the top bit of its input's top
    # limb ('s'), and its sum CF ('s') or OF ('a')
    emit("\txor\t%eax, %eax")
    for k in (1, 2, 3):
        emit("\tadd\t%rax, %rax")
        if modes[k - 1] == 'a':
            emit(f"\tsub\t{CF[k]}(%rsp), %rax")
        else:
            emit(f"\tmovzbl\t{CF[k]}(%rsp), {TMP}d")
            emit("\tadd\t$1, %rax")
            emit(f"\tsub\t{TMP}, %rax")
        if modes[k - 1] == 'a':
            emit(f"\tmovzbl\t{OF[k]}(%rsp), {TMP}d")
            emit(f"\tadd\t{TMP}, %rax")
        else:
            emit(f"\tmov\t{PRV[k]}(%rsp), {TMP}")
            emit(f"\tshr\t$63, {TMP}")
            emit(f"\tadd\t{TMP}, %rax")
    emit("\tadd\t$72, %rsp")
    epilogue(name)


def t5r():
    """gtc_t5r_n: the Gaussian Toom-5's t5 pass (gtc.c, eval5_t5r) from its
    second limb: d = x0 - 4 x4 (or 4 x4 - x0) along CF and dm = 4 y - d
    along OF, both chains live through the whole pass: each a subtraction
    as the addition of the complement (adcx, adox; sbb would write OF
    too), the carry-in 1 for no borrow.  The shifted limbs 4 x4 and 4 y
    are made by shrx of the limb before and lea, which leave the flags
    alone; the limbs of x4 and y rotate through two registers each, and
    the loop is lea and jrcxz.  m (a multiple of 4) limbs; the arguments'
    pointers at the first limb taken, whose predecessors x4[-1] and y[-1]
    are read.  r9 = m << 3 | neg << 2 | c << 1 | b, the chains' borrows
    in; returns b | c << 1."""
    name = "gtc_t5r_n"
    Z, Y = ["%r12", "%r13"], ["%r14", "%r15"]
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\t5")
    emit(f"{name}:")
    for r in ["rbx", "rbp", "r12", "r13", "r14", "r15"]:
        emit(f"\tpush\t%{r}")
    emit("\tmov\t%rcx, %r10\t\t/* x4 */")
    emit(f"\tmov\t-8(%r10), {Z[1]}\t/* x4[-1] */")
    emit(f"\tmov\t-8(%r8), {Y[1]}\t/* y[-1] */")
    emit("\tmov\t%r9, %rcx")
    emit("\tshr\t$3, %rcx\t\t/* m */")
    for p_ in ("%rdi", "%rsi", "%rdx", "%r10", "%r8"):
        emit(f"\tlea\t({p_},%rcx,8), {p_}\t/* at its end */")
    emit("\tneg\t%rcx\t\t\t/* the index, -m */")
    emit("\tmov\t$62, %r11d")
    emit("\tmov\t%r9, %rbx")
    emit("\tshr\t%rbx")
    emit("\tand\t$1, %ebx")
    emit("\tsub\t$1, %rbx\t\t/* c - 1: all ones for c = 0 */")
    emit("\tmov\t%r9, %rax")
    emit("\tand\t$1, %eax")
    emit("\txor\t$1, %eax\t\t/* 1 - b */")
    neg_l, out_ = lab("t5n"), lab("t5o")
    emit("\ttest\t$4, %r9b")
    emit(f"\tjnz\t{neg_l}")
    for neg in (0, 1):
        if neg:
            emit(f"{neg_l}:")
        top, done = lab("t5l"), lab("t5d")
        emit("\tmov\t$1, %r9d")
        emit("\tneg\t%rax\t\t\t/* CF = 1 - b, OF = 0 */")
        emit("\tadox\t%r9, %rbx\t\t/* OF = 1 - c */")
        emit("\t.p2align\t4")
        emit(f"{top}:")
        for u in range(4):
            o = 8 * u
            zc, zp, yc, yp = Z[u % 2], Z[(u + 1) % 2], Y[u % 2], Y[(u + 1) % 2]
            emit(f"\tmov\t{o}(%r10,%rcx,8), {zc}")
            emit(f"\tshrx\t%r11, {zp}, %rbx")
            emit(f"\tlea\t(%rbx,{zc},4), %r9\t/* 4 x4 */")
            emit(f"\tmov\t{o}(%rdx,%rcx,8), %rax")
            if neg:
                emit("\tnot\t%rax")
                emit("\tadcx\t%rax, %r9")
                emit(f"\tmov\t%r9, {o}(%rdi,%rcx,8)")
                dr = "%r9"
            else:
                emit("\tnot\t%r9")
                emit("\tadcx\t%r9, %rax")
                emit(f"\tmov\t%rax, {o}(%rdi,%rcx,8)")
                dr = "%rax"
            emit(f"\tmov\t{o}(%r8,%rcx,8), {yc}")
            emit(f"\tshrx\t%r11, {yp}, %rbx")
            emit(f"\tlea\t(%rbx,{yc},4), %rbp\t/* 4 y */")
            emit(f"\tnot\t{dr}")
            emit(f"\tadox\t{dr}, %rbp")
            emit(f"\tmov\t%rbp, {o}(%rsi,%rcx,8)")
        emit("\tlea\t4(%rcx), %rcx")
        emit(f"\tjrcxz\t{done}")
        emit(f"\tjmp\t{top}")
        emit(f"{done}:")
        if not neg:
            emit(f"\tjmp\t{out_}")
    emit(f"{out_}:")
    emit("\tsetc\t%al")
    emit("\tseto\t%bl")
    emit("\tmovzbl\t%al, %eax")
    emit("\tmovzbl\t%bl, %ebx")
    emit("\txor\t$1, %eax\t\t/* b = 1 - CF */")
    emit("\txor\t$1, %ebx\t\t/* c = 1 - OF */")
    emit("\tlea\t(%rax,%rbx,2), %rax")
    for r in ["r15", "r14", "r13", "r12", "rbp", "rbx"]:
        emit(f"\tpop\t%{r}")
    emit("\tret")
    emit(f"\t.size\t{name}, .-{name}")


def div5_add_fused(kind):
    """mp_limb_t gtc_div5_add_sub_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                                     mp_size_t n, unsigned mode, mp_ptr sp)

    gtc_div5_add_n's {rp, n} = q (mode & 3 as there), and in the same pass
    {sp, n} = z - q modulo B^n, z = y with mode & 4, else x: a third
    chain per block of 4 (sbb of the quotient limbs just stored), parked
    in r10.  The block count in r13, the rest's on the stack, sp in r15.
    Returns the final h.  n >= 0; rp and sp overlap neither xp nor yp nor
    each other.  The Toom-5s' q6 and q2 = c2 - q6 (gtc.c, tc5_q6 and
    r2gtc_55_interp)."""
    if kind == "acc":
        # gtc_div5_add_acc_n (rp, xp, yp, n, mode, ap): q into rp, and
        # {ap, n} += q (mode & 4: -= q) along the third chain; returns
        # that chain's carry (or borrow), 0 or 1.  The Toom-5 style
        # reduction's q5 and q7 (gtc.c, r2gtc_55_interp).
        pass
    name = "gtc_div5_add_sub_n" if kind == "sub" else "gtc_div5_add_acc_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name)
    emit("\tsub\t$8, %rsp\t\t/* (%rsp): the rest */")
    emit("\tmov\t%rcx, %r13")
    emit("\tshr\t$2, %r13\t\t/* blocks of 4 */")
    emit("\tand\t$3, %ecx")
    emit("\tmov\t%rcx, (%rsp)")
    emit("\tmov\t%r9, %r15\t\t/* sp */")
    emit("\txor\t%r12d, %r12d\t\t/* h */")
    emit("\txor\t%r9d, %r9d\t\t/* the sum's chain, parked: 0 or ~0 */")
    emit("\txor\t%r10d, %r10d\t\t/* z - q's chain, parked */")
    emit("\tmov\t%rdx, %r14\t\t/* yp */")
    emit("\tmovabs\t$0x3333333333333333, %rdx\t/* (B-1)/5 */")
    end = lab("a5se")
    paths = {(m, zy): lab("a5sp") for m in (0, 1, 2) for zy in (0, 1)}
    emit("\tmov\t%r8d, %eax")
    emit("\tand\t$3, %eax")
    emit("\ttest\t$4, %r8b")
    zy1 = lab("a5sz")
    emit(f"\tjnz\t{zy1}")
    for zy in (0, 1):
        if zy:
            emit(f"{zy1}:")
        emit("\tcmp\t$1, %eax")
        emit(f"\tje\t{paths[(1, zy)]}")
        emit(f"\tja\t{paths[(2, zy)]}")
        emit(f"\tjmp\t{paths[(0, zy)]}")
    S = ["%rbx", "%rbp", "%r11", "%rcx"]
    for (mode, zy), l in paths.items():
        op = "adc" if mode == 0 else "sbb"
        first, second = ("%r14", "%rsi") if mode == 2 else ("%rsi", "%r14")
        z = "%r14" if zy else "%rsi"
        aop = "sbb" if zy else "adc"
        one, four, l4 = lab("a5so"), lab("a5sf"), lab("a5sg")
        emit(f"{l}:")
        emit("\tcmpq\t$0, (%rsp)")
        emit(f"\tje\t{l4}")
        emit(f"{one}:")
        emit("\tneg\t%r9")
        emit(f"\tmov\t({first}), %rbx")
        emit(f"\t{op}\t({second}), %rbx")
        emit("\tsbb\t%r9, %r9")
        emit("\tmulx\t%rbx, %rax, %r8")
        emit("\tsub\t%rax, %r12")
        emit("\tmov\t%r12, (%rdi)")
        emit("\tsbb\t%r8, %r12")
        emit("\tneg\t%r10")
        if kind == "sub":
            emit(f"\tmov\t({z}), %rbx")
            emit("\tsbb\t(%rdi), %rbx")
        else:
            emit("\tmov\t(%r15), %rbx")
            emit(f"\t{aop}\t(%rdi), %rbx")
        emit("\tmov\t%rbx, (%r15)")
        emit("\tsbb\t%r10, %r10")
        for p_ in ("%rsi", "%r14", "%rdi", "%r15"):
            emit(f"\tlea\t8({p_}), {p_}")
        emit("\tsubq\t$1, (%rsp)")
        emit(f"\tjnz\t{one}")
        emit(f"{l4}:")
        emit("\ttest\t%r13, %r13")
        emit(f"\tje\t{end}")
        emit("\t.p2align\t4")
        emit(f"{four}:")
        emit("\tneg\t%r9")
        for u in range(4):
            emit(f"\tmov\t{8 * u}({first}), {S[u]}")
            emit(f"\t{op}\t{8 * u}({second}), {S[u]}")
        emit("\tsbb\t%r9, %r9")
        for u in range(4):
            emit(f"\tmulx\t{S[u]}, %rax, %r8")
            emit("\tsub\t%rax, %r12")
            emit(f"\tmov\t%r12, {8 * u}(%rdi)")
            emit("\tsbb\t%r8, %r12")
        emit("\tneg\t%r10")
        for u in range(4):
            if kind == "sub":
                emit(f"\tmov\t{8 * u}({z}), {S[u]}")
                emit(f"\tsbb\t{8 * u}(%rdi), {S[u]}")
            else:
                emit(f"\tmov\t{8 * u}(%r15), {S[u]}")
                emit(f"\t{aop}\t{8 * u}(%rdi), {S[u]}")
            emit(f"\tmov\t{S[u]}, {8 * u}(%r15)")
        emit("\tsbb\t%r10, %r10")
        for p_ in ("%rsi", "%r14", "%rdi", "%r15"):
            emit(f"\tlea\t32({p_}), {p_}")
        emit("\tdec\t%r13")
        emit(f"\tjnz\t{four}")
        emit(f"\tjmp\t{end}")
    emit(f"{end}:")
    if kind == "sub":
        emit("\tmov\t%r12, %rax")
    else:
        emit("\tmov\t%r10, %rax")
        emit("\tneg\t%rax\t\t\t/* the third chain's carry */")
    emit("\tadd\t$8, %rsp")
    epilogue(name)


def aorslsh():
    """mp_limb_t gtc_aorslsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                                mp_size_t n, unsigned s, unsigned mode)

    {rp, n} = x + 2^s y (mode 0), x - 2^s y (mode 1) or 2^s y - x (mode
    2); returns the limb above, or what was borrowed from it, as GMP's
    mpn_addmul_1 and mpn_submul_1 by 2^s do, and for mode 2 the bits
    shifted out minus the borrow, as GMP's mpn_rsblsh_n.  1 <= s <= 63;
    rp may equal xp.  A limb of 2^s y is made flag-free, so one chain
    along CF (adc, sbb) runs through the pass: for s <= 3 by shrx of the
    limb below and lea with the scale 2^s, else by shlx, shrx and lea.
    The loop control is dec, which leaves CF alone, and jrcxz.  The limbs
    of y rotate through two registers; the n mod 4 limbs first, then
    blocks of 4."""
    name = "gtc_aorslsh_n"
    Y = ["%rbx", "%r11"]
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\t5")
    emit(f"{name}:")
    emit("\tpush\t%rbx")
    emit("\tpush\t%rbp")
    emit("\tmov\t$64, %r10d")
    emit("\tsub\t%r8d, %r10d\t\t/* 64 - s */")
    emit("\txor\t%r11d, %r11d\t\t/* the limb below */")
    entries = {}
    for sh in (1, 2, 3, 0):
        for mode in (0, 1, 2):
            entries[(sh, mode)] = lab(f"lsh{sh}m{mode}_")
    # dispatch: s (1, 2, 3 or the general form), then the mode
    for sh in (1, 2, 3):
        emit(f"\tcmp\t${sh}, %r8d")
        emit(f"\tje\t{lab('d')}_{sh}")
    disp = {sh: f".Ld{label_no[0] - 2 + sh - 1}_{sh}" for sh in (1, 2, 3)}
    for sh in (0, 1, 2, 3):
        if sh:
            emit(f"{disp[sh]}:")
        emit("\tcmp\t$1, %r9d")
        emit(f"\tjb\t{entries[(sh, 0)]}")
        emit(f"\tje\t{entries[(sh, 1)]}")
        emit(f"\tjmp\t{entries[(sh, 2)]}")
    for (sh, mode), entry in entries.items():
        tag = f"{sh}{mode}"

        def limb(o, cur, prv):
            emit(f"\tmov\t{o}(%rdx), {cur}")
            emit(f"\tshrx\t%r10, {prv}, %rax")
            if sh:
                emit(f"\tlea\t(%rax,{cur},{1 << sh}), %rax")
            else:
                emit(f"\tshlx\t%r8, {cur}, %rbp")
                emit("\tlea\t(%rax,%rbp), %rax")
            if mode == 1:
                emit(f"\tmov\t{o}(%rsi), %rbp")
                emit("\tsbb\t%rax, %rbp")
                emit(f"\tmov\t%rbp, {o}(%rdi)")
            elif mode == 2:
                emit(f"\tsbb\t{o}(%rsi), %rax")
                emit(f"\tmov\t%rax, {o}(%rdi)")
            else:
                emit(f"\tadc\t{o}(%rsi), %rax")
                emit(f"\tmov\t%rax, {o}(%rdi)")
        ls, lb, ll, ld = (lab("ls" + tag), lab("lb" + tag), lab("ll" + tag),
                          lab("ld" + tag))
        emit(f"{entry}:")
        emit("\tmov\t%rcx, %r9")
        emit("\tshr\t$2, %r9\t\t/* blocks */")
        emit("\tand\t$3, %ecx")
        emit("\tclc")
        emit(f"\tjrcxz\t{lb}")
        emit(f"{ls}:")
        limb(0, Y[0], Y[1])
        emit(f"\tmov\t{Y[0]}, {Y[1]}")
        for p_ in ("%rdi", "%rsi", "%rdx"):
            emit(f"\tlea\t8({p_}), {p_}")
        emit("\tdec\t%ecx")
        emit(f"\tjnz\t{ls}")
        emit(f"{lb}:")
        lz = lab("lz" + tag)
        emit("\tmov\t%r9, %rcx")
        emit(f"\tjrcxz\t{lz}\t\t/* jrcxz reaches 127 bytes only */")
        emit(f"\tjmp\t{ll}")
        emit(f"{lz}:")
        emit(f"\tjmp\t{ld}")
        emit("\t.p2align\t4")
        emit(f"{ll}:")
        for u in range(4):
            limb(8 * u, Y[u % 2], Y[1 - u % 2])
        for p_ in ("%rdi", "%rsi", "%rdx"):
            emit(f"\tlea\t32({p_}), {p_}")
        emit("\tdec\t%rcx")
        emit(f"\tjnz\t{ll}")
        emit(f"{ld}:")
        emit("\tsetc\t%al")
        emit("\tmovzbl\t%al, %eax")
        emit(f"\tshrx\t%r10, {Y[1]}, %rbp")
        if mode == 2:
            emit("\tneg\t%rax")
        emit("\tadd\t%rbp, %rax")
        emit("\tpop\t%rbp")
        emit("\tpop\t%rbx")
        emit("\tret")
    emit(f"\t.size\t{name}, .-{name}")


def main():
    emit("/* Copyright (c) 2026 Huawei Technologies Co., Ltd.")
    emit("   SPDX-License-Identifier: BSD-3-Clause-Clear  */")
    emit("/* Generated by gen_linear.py; do not edit.")
    emit("")
    emit("   void gtc_kara_recomp2 (mp_ptr P, mp_ptr Q, mp_srcptr uP,")
    emit("                          mp_srcptr uQ, mp_size_t k, mp_limb_t *io)")
    emit("")
    emit("   The passes of the Karatsuba recomposition (gtc_22) for both")
    emit("   components C = P and C = Q, each holding a0, a1 (of V0) at C, C+k")
    emit("   and b0, b1 (of V2) at C+2k, C+3k:")
    emit("     A: t  = a1 + b0     over k limbs, into C+2k;")
    emit("     B: X1 = t + a0      over k limbs, into C+k;")
    emit("     C: X2 = t + b1      over k limbs, into C+2k, where b1 has")
    emit("                         io[0] = nb limbs (0 <= nb <= k), zero above;")
    emit("     D: C[k..3k) += u XOR mask, with carry-in mask & 1: minus vm1")
    emit("        when mask = ~0, plus vm1 when mask = 0 (u = uP or uQ).")
    emit("   io[1], io[2]: the masks of P and Q.  The carries out, 0 or 1, go")
    emit("   to io[3..10]: A of P, A of Q, B of P, B of Q, C of P, C of Q, D of")
    emit("   P, D of Q.  io[11..18] are scratch (loop counts); io has 19")
    emit("   limbs.  k >= 1.")
    emit("")
    emit("   mp_limb_t gtc_aors2_n (mp_ptr P, mp_ptr Q, mp_srcptr xP,")
    emit("                          mp_srcptr xQ, mp_size_t n, unsigned signs)")
    emit("")
    emit("   {P, n} += {xP, n} and {Q, n} += {xQ, n} in one loop, P along CF")
    emit("   and Q along OF; signs & 1 (& 2) subtracts xP (xQ) instead, as the")
    emit("   complement with a carry-in of 1.  Returns cP + 2 cQ, the carries")
    emit("   out of those complement additions: a subtraction borrowed iff its")
    emit("   carry is 0.  n >= 0.  The accumulation of Toom-3's odd")
    emit("   coefficients into both components (gtc.c, acc2) on a Zen; a Core")
    emit("   takes GMP's two passes there instead (GTC_ACC2_SPLIT).")
    emit("")
    emit("   mp_limb_t gtc_add3_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp,")
    emit("                         mp_srcptr wp, mp_size_t n, unsigned signs)")
    emit("")
    emit("   {rp, n} = {up, n} + {vp, n} + {wp, n}, v along CF and w along OF;")
    emit("   signs & 1 (& 2) subtracts vp (wp) instead, as above.  Returns the")
    emit("   sum of the two carries out, 0, 1 or 2; the signed carry of the")
    emit("   whole is that minus the number of subtractions.  rp may equal up.")
    emit("   n >= 0.  An accumulation fused with the last step of the")
    emit("   interpolation before it (gtc.c, acc3).")
    emit("")
    emit("   void gtc_k2_recomp (mp_ptr rp, mp_srcptr Q, mp_size_t k,")
    emit("                       mp_size_t s)")
    emit("")
    emit("   The recomposition of the Karatsuba-style reduction (gtc.c,")
    emit("   r2gtc_22), n = k + s, s = k or k - 1, s >= 1: {rp, 2k} = c0 + c2")
    emit("   modulo B^(2k), c0 in {rp, 2k} and c2 the 2s limbs at rp + 2k,")
    emit("   along CF; then {rp + k, k + 2s} += {Q, n + 1} (c1) along OF, in")
    emit("   the same pass over limbs k .. 2k.  The carry out of c1 is")
    emit("   propagated; the product absorbs it.")
    emit("")
    emit("   mp_limb_t gtc_rsh1add3_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp,")
    emit("                             mp_srcptr cp, mp_size_t n, unsigned comp)")
    emit("")
    emit("   {rp, n} = (a + b + c mod B^n) >> 1, arithmetic, or with comp = 1")
    emit("   (b + c - a mod B^n) >> 1: the interpolations' halvings fused with")
    emit("   the three-operand sum before them (gtc.c, reg_add3_half).  n >= 1;")
    emit("   rp may equal ap, bp or cp.  Returns 0.")
    emit("")
    emit("   mp_limb_t gtc_add_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp,")
    emit("                        mp_size_t n)")
    emit("   mp_limb_t gtc_sub_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp,")
    emit("                        mp_size_t n)")
    emit("")
    emit("   GMP's mpn_add_n and mpn_sub_n, n >= 1, rp may equal up or vp")
    emit("   (gtc.c, GTC_ADD_N, GTC_SUB_N).")
    emit("")
    emit("   mp_limb_t gtc_div15_lsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,")
    emit("                              mp_srcptr zp, mp_size_t n, unsigned signs)")
    emit("")
    emit("   {rp, n} = (x + 4 y + 8 z) / 15 modulo B^n, exact; signs & 1 (& 2)")
    emit("   subtracts 4 y (8 z) instead.  Returns the final h of GMP's")
    emit("   mpn_bdiv_dbm1c, whose step it is.  n >= 0; rp may equal xp.  The")
    emit("   Gaussian Toom-4's q5 (gtc.c, gtc_44_interp_coeffs).")
    emit("")
    emit("   mp_limb_t gtc_addlsh123_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp,")
    emit("                              mp_srcptr cp, mp_srcptr dp, mp_size_t n)")
    emit("")
    emit("   {rp, n} = a + 2 b + 4 c + 8 d by Horner's rule; returns the limb")
    emit("   above (at most 14).  n >= 1; rp may equal ap.  The Gaussian")
    emit("   Toom-4's evaluation at 2 (gtc.c, gauss_eval4).")
    emit("")
    emit("   mp_limb_t gtc_aorslsh1_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,")
    emit("                             mp_size_t n, unsigned sub)")
    emit("")
    emit("   {rp, n} = x + 2 y, or x - 2 y with sub; returns the carry or the")
    emit("   borrow (0 to 2), as mpn_addlsh1_n and mpn_sublsh1_n.  n >= 0; rp")
    emit("   may equal xp.  The Gaussian Toom-4's q5 - 2 q6 (gtc.c,")
    emit("   reg_aorslsh1).")
    emit("")
    emit("   mp_limb_t gtc_div5_lsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,")
    emit("                             mp_size_t n, unsigned sub)")
    emit("")
    emit("   {rp, n} = (x + 4 y) / 5 modulo B^n, exact; sub & 1 subtracts 4 y")
    emit("   instead.  Returns the final h of GMP's mpn_bdiv_dbm1c.  n >= 0; rp")
    emit("   may equal xp.  The Toom-4 style reduction's q5 (gtc.c,")
    emit("   r2gtc_44_interp).")
    emit("")
    emit("   mp_limb_t gtc_aorsmul_1 (mp_ptr rp, mp_srcptr up, mp_size_t n,")
    emit("                            mp_limb_t v, unsigned sub)")
    emit("")
    emit("   {rp, n} += v {up, n}, or -= with sub & 1; returns the high limb,")
    emit("   or the borrow, as mpn_addmul_1 and mpn_submul_1.  n >= 0.  The")
    emit("   Toom-4 style reduction's 10 q6 (gtc.c, reg_aorsmul_u).")
    emit("")
    emit("   mp_limb_t gtc_rsh2add_n, gtc_rsh2sub_n (mp_ptr rp, mp_srcptr ap,")
    emit("                                          mp_srcptr bp, mp_size_t n)")
    emit("   mp_limb_t gtc_rsh2add3_n, gtc_rsh3add3_n (mp_ptr rp, mp_srcptr ap,")
    emit("                       mp_srcptr bp, mp_srcptr cp, mp_size_t n,")
    emit("                       unsigned comp)")
    emit("")
    emit("   gtc_rsh1add_n, gtc_rsh1sub_n and gtc_rsh1add3_n shifted by 2 or 3")
    emit("   bits instead of 1 (the first two return the bits shifted out).")
    emit("")
    emit("   mp_limb_t gtc_div5_add_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,")
    emit("                             mp_size_t n, unsigned mode)")
    emit("")
    emit("   {rp, n} = (x + y) / 5 modulo B^n, exact; mode 1: (x - y) / 5,")
    emit("   mode 2: (y - x) / 5.  Returns the final h of GMP's")
    emit("   mpn_bdiv_dbm1c.  n >= 0; rp may equal xp or yp.  The Gaussian")
    emit("   Toom-5's q5, q6 and q7 from sums already divided by 4 (gtc.c,")
    emit("   reg_div5_add).")
    emit("")
    emit("   mp_limb_t gtc_t5r_n (mp_ptr d, mp_ptr dm, mp_srcptr x0,")
    emit("                        mp_srcptr x4, mp_srcptr y, mp_limb_t mf)")
    emit("")
    emit("   {d, m} = x0 - 4 x4 (or 4 x4 - x0) and {dm, m} = 4 y - d, the")
    emit("   shifted limbs including the top bits of x4[-1] and y[-1], with")
    emit("   the borrows b and c in and out; mf = m << 3 | neg << 2 | c << 1")
    emit("   | b, m a multiple of 4, m >= 4; returns b | c << 1.  The Gaussian")
    emit("   Toom-5's t5 pass (gtc.c, eval5_t5r).")
    emit("")
    emit("   mp_limb_t gtc_div5_add_sub_n (mp_ptr rp, mp_srcptr xp,")
    emit("                     mp_srcptr yp, mp_size_t n, unsigned mode, mp_ptr sp)")
    emit("")
    emit("   gtc_div5_add_n's {rp, n} = q (mode & 3), and {sp, n} = z - q")
    emit("   modulo B^n, z = y with mode & 4, else x, in the same pass.  rp")
    emit("   and sp overlap neither xp nor yp nor each other.  The Toom-5s'")
    emit("   q6 and q2 = c2 - q6 (gtc.c).")
    emit("")
    emit("   mp_limb_t gtc_div5_add_acc_n (mp_ptr rp, mp_srcptr xp,")
    emit("                     mp_srcptr yp, mp_size_t n, unsigned mode, mp_ptr ap)")
    emit("")
    emit("   gtc_div5_add_n's {rp, n} = q (mode & 3), and {ap, n} += q, or -=")
    emit("   q with mode & 4, in the same pass; returns that addition's carry")
    emit("   (or borrow).  rp may equal xp or yp; ap overlaps none of")
    emit("   rp, xp, yp.  The Toom-5 style reduction's q5 and q7 (gtc.c).")
    emit("")
    emit("   mp_limb_t gtc_rsh3add_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp,")
    emit("                            mp_size_t n)")
    emit("   mp_limb_t gtc_rsh3sub_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp,")
    emit("                            mp_size_t n)")
    emit("")
    emit("   (a +- b mod B^n) >> 3, arithmetic, as gtc_rsh1add_n by 3 bits:")
    emit("   the Gaussian Toom-5's q7 from a shared sum (gtc.c, GTC_TC5_AB).")
    emit("")
    emit("   mp_limb_t gtc_aorslsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,")
    emit("                            mp_size_t n, unsigned s, unsigned mode)")
    emit("")
    emit("   {rp, n} = x + 2^s y (mode 0), x - 2^s y (1) or 2^s y - x (2),")
    emit("   1 <= s <= 63; returns the limb above (or the borrow from it) as")
    emit("   mpn_addmul_1 and mpn_submul_1 by 2^s, and as mpn_rsblsh_n with")
    emit("   mode 2 (the shifted-out bits minus the borrow).  n >= 0; rp may")
    emit("   equal xp.  The Gaussian Toom-5's 8 q4 and 32 q8 (gtc.c,")
    emit("   reg_aorsmul5), and in place of GMP's rsblsh1, rsblsh2 and addlsh2")
    emit("   (GTC_RSBLSH1_N, GTC_RSBLSH2_N, GTC_ADDLSH2_N).")
    emit("")
    emit("   mp_limb_t gtc_div5_rlsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,")
    emit("                              mp_size_t n, unsigned rsb)")
    emit("")
    emit("   {rp, n} = (x + 4 y) / 5, or (4 y - x) / 5 with rsb & 1, modulo")
    emit("   B^n, the division exact: gtc_div5_lsh_n with the limbs of 4 y")
    emit("   made by shld.  rp may equal xp.  r2gtc_44's q5 on the Intel")
    emit("   cores (GTC_RTC4_SHLD).  */")
    emit("")
    emit("#if defined (__x86_64__) && defined (__ELF__) && defined (__BMI2__) "
         "&& defined (__ADX__) && !defined (GTC_NO_ASM)")
    # The section aligned to 64 bytes, so that the kernels keep their place
    # in the 64-byte fetch windows whatever the size of the code before them
    # (gtc.o): with the 32 of their own .p2align, a change of gtc.c moved
    # them by 32 mod 64 at random, 0.1-0.2% of whole products.
    emit("\t.text")
    emit("\t.p2align\t6")
    kara_recomp2()
    # shrd is one micro-op on the Intel cores and several on the Zen, where
    # the halvings keep the flags-free shift
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__)")
    rsh1_flagless("gtc_rsh1add_n", False)
    rsh1_flagless("gtc_rsh1sub_n", True)
    emit("#else")
    rsh1("gtc_rsh1add_n", False)
    rsh1("gtc_rsh1sub_n", True)
    emit("#endif")
    aors2()
    add3()
    k2_recomp()
    # the fused halving likewise: flags-free on the Zen
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__)")
    rsh1add3_flagless()
    emit("#else")
    rsh1add3()
    emit("#endif")
    aors_n("gtc_add_n", False)
    aors_n("gtc_sub_n", True)
    # the division by 15 in blocks of 16 limbs on the Zen and with AVX-512,
    # of 8 on the other Intel cores
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__) || defined (__AVX512F__)")
    div15_lsh(16)
    emit("#else")
    div15_lsh(8)
    emit("#endif")
    # the value at 2: shld for steps 2 and 3 on the Intel cores, the
    # doublings along CF on the Zen, where shld is several micro-ops
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__)")
    addlsh123("aaa")
    emit("#else")
    addlsh123("ass")
    emit("#endif")
    aorslsh1()
    div5_lsh_reg()
    div5_rlsh_shld()
    aorsmul1()
    # 32 bytes where a kernel no longer used was, so that the kernels below
    # keep their offsets modulo 64 (see above)
    emit("\t.p2align\t5")
    emit("\t.skip\t32, 0xcc")
    # the Gaussian Toom-5's shifted sums and its division by 5 of a sum
    # already divided by 4 (the forms of the halvings above, shifted by 2
    # or 3 bits)
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__)")
    rsh1_flagless("gtc_rsh2add_n", False, 2)
    rsh1_flagless("gtc_rsh2sub_n", True, 2)
    rsh1add3_flagless("gtc_rsh2add3_n", 2)
    rsh1add3_flagless("gtc_rsh3add3_n", 3)
    emit("#else")
    rsh1("gtc_rsh2add_n", False, 2)
    rsh1("gtc_rsh2sub_n", True, 2)
    rsh1add3("gtc_rsh2add3_n", 2)
    rsh1add3("gtc_rsh3add3_n", 3)
    emit("#endif")
    div5_add_reg()
    t5r()
    div5_add_fused("sub")
    div5_add_fused("acc")
    # (x +- y)/8 for the Gaussian Toom-5's q7 from a shared sum
    # (gtc.c, GTC_TC5_AB), as the shifts by 2 above
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__)")
    rsh1_flagless("gtc_rsh3add_n", False, 3)
    rsh1_flagless("gtc_rsh3sub_n", True, 3)
    emit("#else")
    rsh1("gtc_rsh3add_n", False, 3)
    rsh1("gtc_rsh3sub_n", True, 3)
    emit("#endif")
    aorslsh()
    emit('\t.section\t.note.GNU-stack,"",@progbits')
    emit("#endif")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
