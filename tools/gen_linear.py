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


# Set, the kernels shift by mov, shl, shr and or (the flags are dead
# there) instead of shrd, which is several micro-ops on the Zen and one on
# the Intel cores.
FLAGFREE_SHR = [False]


def shr_pair(s, hi, lo, tmp):
    """lo = (lo >> s) | (hi << (64 - s)), as shrd $s, hi, lo."""
    if FLAGFREE_SHR[0]:
        emit(f"\tmov\t{hi}, {tmp}")
        emit(f"\tshl\t${64 - s}, {tmp}")
        emit(f"\tshr\t${s}, {lo}")
        emit(f"\tor\t{tmp}, {lo}")
    else:
        emit(f"\tshrd\t${s}, {hi}, {lo}")


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


_trees = {}


def mode_dispatch(reg, table):
    """The entry of a kernel's mode copies on reg's low bits (a 32-bit
    register, already masked): a tree of test and jnz, the highest bit
    first, put in by mode_table once the copies' labels are known."""
    _trees[table] = (len(out), reg)
    out.append(None)


def mode_table(table, entries):
    """mode_dispatch's tree for table, entries[m] the copy of mode m."""
    at, reg = _trees.pop(table)
    nb = len(entries).bit_length() - 1
    assert len(entries) == 1 << nb
    tree = []

    def rec(bit, base):
        if bit < 0:
            tree.append(f"\tjmp\t{entries[base]}")
            return
        hi = lab("td")
        tree.append(f"\ttest\t${1 << bit}, {reg}")
        tree.append(f"\tjnz\t{hi}")
        rec(bit - 1, base)
        tree.append(f"{hi}:")
        rec(bit - 1, base | 1 << bit)
    rec(nb - 1, 0)
    out[at:at + 1] = tree


def prologue(name, align=5):
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit(f"\t.p2align\t{align}")
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


def rsh1add3():
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
    shifts part-way; the loop's groups run straight through."""
    name = "gtc_rsh1add3_n"
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
            emit(f"\tshrd\t$1, {R[u + 1]}, {R[u]}")
            emit(f"\tmov\t{R[u]}, {8 * (u - 1)}(%rdi)")
        emit(f"{hend}:")
        emit(f"\tmov\t{R[8]}, {R[0]}")
        for p_ in ptrs:
            emit(f"\tlea\t64({p_}), {p_}")
        emit("\tcmp\t-16(%rsp), %rdi\t\t/* the flags are saved */")
        emit(f"\tjne\t{loop}")
        # the last sum, at n - 1: shifted down arithmetically
        emit(f"\tsar\t$1, {R[0]}")
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


def rsh1add3_flagless():
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
    from the three AGUs and the three flag-capable ALUs."""
    name = "gtc_rsh1add3_n"
    R = ["%rax", "%rbx", "%r8", "%r9"]
    T1, T2 = "%r11", "%r14"
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\t5")
    emit(f"{name}:")
    for r in ["rbx", "r12", "r13", "r14"]:
        emit(f"\tpush\t%{r}")
    emit("\tmov\t%rcx, %r10\t\t/* c */")
    emit("\tmov\t$1, %r12d")
    emit("\tmov\t$63, %r13d")
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
        emit(f"\tsar\t$1, {R[3]}")
        emit(f"\tmov\t{R[3]}, -8(%rdi)")
        emit(f"\tjmp\t{out_}")
        emit("\t.p2align\t2")
        emit(f"{table}:")
        for s_ in stubs:
            emit(f"\t.long\t{s_} - {table}")
    emit(f"{done1}:")
    emit("\tsar\t$1, %rbx")
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



def div15_lsh(neg=False):
    """mp_limb_t gtc_div15_lsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                                  mp_srcptr zp, mp_size_t n, unsigned signs)

    {rp, n} = (x + 4 y + 8 z) / 15 modulo B^n, the division exact; signs
    & 1 (& 2) subtracts 4 y (8 z) instead.  Blocks of 16 limbs, the
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
    and the parking.

    With neg, gtc_div15n_lsh_n: {rp, n} = -(x + 4 y + 8 z) / 15, the
    division along two chains of one cycle a limb: R = R B + s m (m =
    (B-1)/15) is -s/15, the column sums of s m (mulx, adcx of the high
    limb before) along CF and R along OF (adox); both carries go into the
    block's last high limb (at most m + 1), from which, with R's last
    limb, the next block starts.  The loop ends on rp reaching its end,
    kept on the stack.  Returns R's last limb."""
    name = "gtc_div15n_lsh_n" if neg else "gtc_div15_lsh_n"
    BLK = 16
    LG = BLK.bit_length() - 1
    emit(f"/* {name}: see the header.  */")
    prologue(name)
    if neg:
        # (%rsp) the entry of the next block's division, 8(%rsp) rp's end
        emit("\tsub\t$16, %rsp")
    else:
        emit("\tsub\t$8, %rsp\t\t/* the entry of the next block's division */")
    emit("\tmov\t%rdx, %r10")
    emit("\tmov\t%rcx, %r11")
    zero, end = lab("qz"), lab("qe")
    if neg:
        emit("\ttest\t%r8, %r8")
        emit(f"\tjz\t{zero}")
        emit("\tlea\t(%rdi,%r8,8), %rax")
        emit("\tmov\t%rax, 8(%rsp)\t\t/* rp's end */")
    else:
        emit(f"\tlea\t{BLK - 1}(%r8), %rcx")
        emit(f"\tshr\t${LG}, %rcx\t\t/* blocks, the first one partial */")
        emit("\ttest\t%rcx, %rcx")
        emit(f"\tjz\t{zero}")
    emit("\tneg\t%r8")
    emit(f"\tand\t${BLK - 1}, %r8d\t\t/* the limbs the first block skips */")
    emit("\tlea\t(,%r8,8), %rax")
    for p_ in ("%rdi", "%rsi", "%r10", "%r11"):
        emit(f"\tsub\t%rax, {p_}")
    if neg:
        emit("\txor\t%r12d, %r12d\t\t/* R, the running sum */")
        emit("\txor\t%ecx, %ecx\t\t/* the high limb before: 0 */")
    else:
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

        def divs(entries):
            if neg and not entries:
                emit("\ttest\t%r12, %r12\t\t/* CF = OF = 0 */")
            HH = ("%r8", "%rcx")
            for u in range(BLK):
                o = 8 * u
                if entries:
                    emit(f"{eb[u]}:")
                if neg:
                    # -q: R = R B + s m, the column sums of s m along CF,
                    # R along OF; the high limbs alternate, the last in rcx
                    hc, hp = HH[u % 2], HH[(u + 1) % 2]
                    emit(f"\tmulx\t{o}(%rdi), %rax, {hc}")
                    emit(f"\tadcx\t{hp}, %rax")
                    emit("\tadox\t%rax, %r12")
                    emit(f"\tmov\t%r12, {o}(%rdi)")
                    continue
                emit(f"\tmulx\t{o}(%rdi), %rax, %r8")
                emit("\tsub\t%rax, %r12")
                emit(f"\tmov\t%r12, {o}(%rdi)")
                emit("\tsbb\t%r8, %r12")
            if neg:
                # both carries into the last high limb, which leaves CF =
                # OF = 0
                emit("\tmov\t$0, %eax")
                emit("\tadcx\t%rax, %rcx")
                emit("\tadox\t%rax, %rcx")
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

        def last(jz):
            # the loop's end: the block count, or (neg, where rcx holds a
            # high limb) rp against its end
            if neg:
                emit("\tcmp\t8(%rsp), %rdi")
                emit(f"\t{'je' if jz else 'jne'}\t{end if jz else steady}")
            else:
                emit("\tdec\t%rcx")
                emit(f"\t{'jz' if jz else 'jnz'}\t{end if jz else steady}")

        # the first, partial block through the tables, then a loop
        # with neither
        steady = lab("qst")
        sums(True)
        if neg:
            # CF = OF = 0, and the high limb before a first block entered
            # at an odd limb 0
            emit("\txor\t%r8d, %r8d")
        emit("\tjmp\t*(%rsp)")
        divs(True)
        last(True)
        emit("\t.p2align\t4")
        emit(f"{steady}:")
        restore()
        sums(False)
        divs(False)
        last(False)
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
    emit(f"\tadd\t${16 if neg else 8}, %rsp")
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



def div5n_rlsh():
    """mp_limb_t gtc_div5n_rlsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                                   mp_size_t n, unsigned sub)

    {rp, n} = -(x + 4 y) / 5, or (x - 4 y) / 5 with sub & 1, modulo B^n,
    the division exact; n >= 0, rp may equal xp.  The n mod 4 limbs one
    at a time, then blocks of 4: the limbs of 4 y made by shld $2 before
    the chain's CF is restored (shld writes the flags), x then added (adc)
    or subtracted from them (sbb) as a memory operand.  The division along
    two chains of one cycle a limb: R = R B + s m (m = (B-1)/5) is -s/5,
    the column sums (adcx) along CF, R along OF (adox); the high limbs
    alternate in rbp and r8 over the n mod 4 limbs, in r14 (their end,
    dead by then) and r8 over the blocks, and both carries go into the
    last.  The loops end on rp reaching its end, kept in r14 for the n mod
    4 limbs, r15 for the blocks.  Returns R's last limb.  shld is 4 µops
    on the Zen: for the Intel cores."""
    name = "gtc_div5n_rlsh_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name)
    emit("	mov	%rdx, %r10		/* yp */")
    emit("	lea\t(%rdi,%rcx,8), %r15\t/* rp's end */")
    emit("	and\t$3, %ecx")
    emit("	lea\t(%rdi,%rcx,8), %r14\t/* the end of the n mod 4 limbs */")
    emit("	movabs\t$0x3333333333333333, %rdx\t/* (B-1)/5 */")
    emit("	xor\t%r12d, %r12d\t\t/* R */")
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
        # after the test of sub, which came in r8
        emit("\txor\t%r8d, %r8d\t\t/* the high limb before: 0 */")
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
        emit("\txor\t%eax, %eax\t\t/* CF = OF = 0 */")
        emit("\tmulx\t%rbx, %rbx, %rbp")
        emit("\tadcx\t%r8, %rbx")
        emit("\tadox\t%rbx, %r12\t\t/* -q */")
        emit("\tmov\t%r12, (%rdi)")
        emit("\tadcx\t%rax, %rbp\t\t/* both carries in */")
        emit("\tadox\t%rax, %rbp")
        emit("\tmov\t%rbp, %r8")
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
        emit("\txor\t%eax, %eax\t\t/* CF = OF = 0 */")
        for u in range(4):
            hc, hp = ("%r14", "%r8") if u % 2 == 0 else ("%r8", "%r14")
            emit(f"\tmulx\t{S[u]}, {S[u]}, {hc}")
            emit(f"\tadcx\t{hp}, {S[u]}")
            emit(f"\tadox\t{S[u]}, %r12\t\t/* -q */")
            emit(f"\tmov\t%r12, {8 * u}(%rdi)")
        emit("\tadcx\t%rax, %r8\t\t/* both carries in */")
        emit("\tadox\t%rax, %r8")
        for p_ in ("%rsi", "%r10", "%rdi"):
            emit(f"\tlea\t32({p_}), {p_}")
        emit("\tcmp\t%r15, %rdi")
        emit(f"\tjne\t{four}")
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


def aors2c():
    """mp_limb_t gtc_aors2c_n (mp_ptr d, mp_ptr zp, mp_srcptr xp,
                               mp_srcptr yp, mp_size_t n, unsigned signs)

    {d, n} = x +- y along CF and then {zp, n} = d +- z along OF in the same
    pass, d's limb still in its register (signs & 1, & 2 subtract y, z, as
    complements with a carry-in of 1); returns CF + 2 OF.  n >= 0; d may
    equal xp or yp."""
    def step(off, nv, nw):
        o = 8 * off
        emit(f"\tmov\t{o}(%rdx), %rax")
        if nv:
            emit(f"\tmov\t{o}(%r10), %r9")
            emit("\tnot\t%r9")
            emit("\tadcx\t%r9, %rax")
        else:
            emit(f"\tadcx\t{o}(%r10), %rax")
        emit(f"\tmov\t%rax, {o}(%rdi)")
        if nw:
            emit(f"\tmov\t{o}(%rsi), %r9")
            emit("\tnot\t%r9")
            emit("\tadox\t%r9, %rax")
        else:
            emit(f"\tadox\t{o}(%rsi), %rax")
        emit(f"\tmov\t%rax, {o}(%rsi)")
    chains("gtc_aors2c_n",
           "{d, n} = x +- y (CF), {zp, n} = d +- z (OF); returns CF + 2 OF.\n"
           "   See the header.",
           ["%rdi", "%rsi", "%rdx", "%r10"], step,
           [(0, 0), (0, 1), (1, 0), (1, 1)], zero_aors2)
    emit("\tlea\t(%rax,%rcx,2), %eax")
    emit("\tret")
    emit("\t.size\tgtc_aors2c_n, .-gtc_aors2c_n")


def tc5q():
    """mp_limb_t gtc_tc5q_n (mp_ptr a1, mp_ptr a2, mp_srcptr xp,
                            mp_srcptr yp, mp_srcptr cp, mp_size_t n,
                            unsigned mode)

    In one pass over n >= 1 limbs, all read as signed n-limb numbers:
      t = (x + y mod B^n) >> s, arithmetic and exact, s = 2 + (mode >> 3
          & 1), y subtracted with mode & 1;
      q = (t + c) / 5 modulo B^n, exact, c subtracted with mode & 2;
      {a1, n} += q and {a2, n} += c - q, c's sign as in q, both
          subtracted instead with mode & 4.
    q is never stored.  Returns the net carries out of a1 and a2 (what the
    limbs above each change by, -2 to 2) as two 32-bit signed halves, a1's
    low.  a1 and a2 overlap neither each other nor the operands.

    Five carry chains: x +- y (CF; adc or sbb, y from memory), t +- c
    (CF), a1 +- q (CF), and a2 +- c and a2 -+ q in one loop along CF and
    OF (adcx, adox; c subtracted as its complement with a carry-in of 1).
    The sums run one limb ahead of the rest (t needs the next one), in
    blocks of 4 (the n - 1 mod 4 limbs first, in blocks of 1 and 2 by its
    bits), each chain restored at its block's start and parked at its end:
    the first in n's register (sbb, bt), the loop closed on x's end; the
    others on the stack (CF alone: setnc, cmpb $1; CF and OF: seto and sbb
    to a mask, cmpb $0x81 and adcx of the mask into 0).  t by shl, shr and
    or (shrd is several micro-ops on the Zen), q by GMP's mpn_bdiv_dbm1c
    step with (B-1)/5, each limb into the register its t was in.  Where q
    is added (mode & 4 clear), the step makes ~q instead, carrying g = ~h
    (~q = lo + g, the low half straight into t's register, and g' = ~q +
    hi + carry: four µops a limb, the two registers trading places each
    limb), so that a1 + q is a1 - ~q - 1 by sbb and a2 - q is a2 + ~q + 1
    by adox: q needs no complement in any mode.  The last limb's t is the
    last sum shifted arithmetically.  One copy per mode, entered through a
    tree on its bits.  For the Zen."""
    name = "gtc_tc5q_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name, "GTC_TC5_FA")
    A1, A2, X, Y, C, N = "%rdi", "%rsi", "%r10", "%rcx", "%r8", "%r9"
    H, LO, HI = "%r11", "%rax", "%rbx"
    R = ["%rbp", "%r12", "%r13", "%r14", "%r15"]
    # stack: 1 - CF of chains 2 and 3 at 8 and 16 (chain 1 in n's
    # register; 0 unused); the mask -CF4 at 24, OF5's byte at 32; n - 1 at
    # 40, x's end at 48
    P1, P2, P3, P4, P5, CNT, END, FRAME = 0, 8, 16, 24, 32, 40, 48, 56
    MODE = FRAME + 56
    emit(f"\tsub\t${FRAME}, %rsp")
    emit(f"\tmov\t%rdx, {X}")
    emit("\tmovabs\t$0x3333333333333333, %rdx\t/* (B-1)/5 */")
    emit(f"\tmov\t{MODE}(%rsp), %eax")
    emit("\tand\t$15, %eax")
    table = lab("q5t")
    mode_dispatch("%eax", table)
    entries = []
    end = lab("q5e")

    def chain_restore(slot):
        if slot == P1:
            emit(f"\tbt\t$0, {N}\t\t/* CF back */")
        else:
            emit(f"\tcmpb\t$1, {slot}(%rsp)\t/* CF back */")

    def chain_park(slot):
        if slot == P1:
            emit(f"\tsbb\t{N}, {N}\t\t/* -CF */")
        else:
            emit(f"\tsetnc\t{slot}(%rsp)\t/* 1 - CF */")

    for mode in range(16):
        ysub, csub, asub, sh = (mode & 1, mode >> 1 & 1, mode >> 2 & 1,
                                2 + (mode >> 3 & 1))
        nq = 1 - asub			# the registers hold ~q
        ccmp = csub ^ asub		# c complemented on a2's CF chain
        l = lab("q5m")
        entries.append(l)
        emit(f"{l}:")
        emit(f"\tlea\t-1({N}), {N}")
        emit(f"\tmov\t{N}, {CNT}(%rsp)")
        emit(f"\tlea\t8({X},{N},8), {LO}")
        emit(f"\tmov\t{LO}, {END}(%rsp)\t/* x's end */")
        emit(f"\tmov\t({X}), {R[0]}")
        emit(f"\t{'sub' if ysub else 'add'}\t({Y}), {R[0]}\t/* s0 */")
        chain_park(P1)
        emit(f"\tlea\t8({X}), {X}")
        emit(f"\tlea\t8({Y}), {Y}")
        emit(f"\tmovq\t$1, {P2}(%rsp)")
        emit(f"\tmovq\t${1 - nq}, {P3}(%rsp)\t/* a1's borrow-in {nq} */")
        emit(f"\tmovq\t${-ccmp}, {P4}(%rsp)\t/* -CF4 */")
        emit(f"\tmovq\t${nq}, {P5}(%rsp)\t/* OF5 */")
        emit(f"\tmov\t${-nq}, {H}\t\t/* h = 0{', as ~h' if nq else ''} */")

        def rest(m):
            """t in R[0..m-1]: + c, / 5, the accumulations."""
            chain_restore(P2)
            for u in range(m):
                emit(f"\t{'sbb' if csub else 'adc'}\t{8 * u}({C}), {R[u]}")
            chain_park(P2)
            g, lo = H, (HI if nq else LO)
            for u in range(m):
                if nq:
                    # the low half into t's register, the high one into
                    # the register g' then takes
                    emit(f"\tmulx\t{R[u]}, {R[u]}, {lo}")
                    emit(f"\tadd\t{g}, {R[u]}\t\t/* ~q */")
                    emit(f"\tadc\t{R[u]}, {lo}\t\t/* g' */")
                    g, lo = lo, g
                else:
                    emit(f"\tmulx\t{R[u]}, {lo}, {HI}")
                    emit(f"\tsub\t{lo}, {g}")
                    emit(f"\tmov\t{g}, {R[u]}")
                    emit(f"\tsbb\t{HI}, {g}")
            if g != H:
                emit(f"\tmov\t{g}, {H}")
            chain_restore(P3)
            for u in range(m):
                emit(f"\tmov\t{8 * u}({A1}), {LO}")
                emit(f"\tsbb\t{R[u]}, {LO}")
                emit(f"\tmov\t{LO}, {8 * u}({A1})")
            chain_park(P3)
            emit(f"\tcmpb\t$0x81, {P5}(%rsp)\t/* OF back, CF = 1 */")
            emit(f"\tmov\t$0, {LO}")
            emit(f"\tadcx\t{P4}(%rsp), {LO}\t/* CF back */")
            for u in range(m):
                emit(f"\tmov\t{8 * u}({A2}), {LO}")
                if ccmp:
                    emit(f"\tmov\t{8 * u}({C}), {HI}")
                    emit(f"\tnot\t{HI}")
                    emit(f"\tadcx\t{HI}, {LO}")
                else:
                    emit(f"\tadcx\t{8 * u}({C}), {LO}")
                emit(f"\tadox\t{R[u]}, {LO}")
                emit(f"\tmov\t{LO}, {8 * u}({A2})")
            emit(f"\tseto\t{P5}(%rsp)")
            emit(f"\tsbb\t{HI}, {HI}")
            emit(f"\tmov\t{HI}, {P4}(%rsp)\t/* -CF4 */")
            for p_ in (C, A1, A2):
                emit(f"\tlea\t{8 * m}({p_}), {p_}")

        def block(m):
            """The sums of the next m limbs into R[1..m], t of the m before
            them into R[0..m-1] (R[0] the last sum before), the rest (which
            leaves R[m] alone); the last sum back to R[0]."""
            chain_restore(P1)
            for u in range(m):
                emit(f"\tmov\t{8 * u}({X}), {R[u + 1]}")
                emit(f"\t{'sbb' if ysub else 'adc'}\t{8 * u}({Y}), {R[u + 1]}")
            chain_park(P1)
            for u in range(m):
                shr_pair(sh, R[u + 1], R[u], LO)
            for p_ in (X, Y):
                emit(f"\tlea\t{8 * m}({p_}), {p_}")
            rest(m)
            emit(f"\tmov\t{R[m]}, {R[0]}")

        for bit in (0, 1):
            skip = lab("q5b")
            emit(f"\ttestb\t${1 << bit}, {CNT}(%rsp)")
            emit(f"\tjz\t{skip}")
            block(1 << bit)
            emit(f"{skip}:")
        loop, ldone = lab("q5l"), lab("q5d")
        emit(f"\tcmp\t{END}(%rsp), {X}")
        emit(f"\tje\t{ldone}")
        emit("\t.p2align\t4")
        emit(f"{loop}:")
        block(4)
        emit(f"\tcmp\t{END}(%rsp), {X}")
        emit(f"\tjne\t{loop}")
        emit(f"{ldone}:")
        # the last limb: the last sum shifted arithmetically
        emit(f"\tsar\t${sh}, {R[0]}")
        rest(1)
        # the net carries: q's sign bit tq (R[0] its top limb, or ~ of it),
        # c's tc; a1: (1 - b3) - tq adding (a1 - ~q - 1 = a1 + q - B^n),
        # tq - b3 subtracting; a2: CF4 + OF5 less, for each of c and q as
        # added, its sign bit, or 1 - it where complemented
        emit(f"\tshr\t$63, {R[0]}")
        if nq:
            emit(f"\txor\t$1, {R[0]}\t\t/* tq */")
        emit(f"\tmovzbl\t{P3}(%rsp), %eax\t/* 1 - b3 */")
        if nq:
            emit(f"\tsub\t{R[0]}, %rax")
        else:
            emit(f"\tlea\t-1(%rax,{R[0]}), %rax")
        emit("\tmov\t%eax, %eax")
        emit(f"\tmov\t-8({C}), {HI}")
        emit(f"\tshr\t$63, {HI}\t\t/* tc */")
        emit(f"\tmovzbl\t{P5}(%rsp), %ecx")
        emit(f"\tsub\t{P4}(%rsp), %rcx\t/* CF4 + OF5 */")
        if ccmp:
            emit(f"\tlea\t-1(%rcx,{HI}), %rcx")
        else:
            emit(f"\tsub\t{HI}, %rcx")
        if nq:
            emit(f"\tlea\t-1(%rcx,{R[0]}), %rcx")
        else:
            emit(f"\tsub\t{R[0]}, %rcx")
        emit("\tshl\t$32, %rcx")
        emit("\tor\t%rcx, %rax")
        emit(f"\tjmp\t{end}")
    mode_table(table, entries)
    emit(f"{end}:")
    emit(f"\tadd\t${FRAME}, %rsp")
    epilogue(name)


def tc5q8():
    """mp_limb_t gtc_tc5q8_n (mp_ptr a1, mp_ptr a2, mp_srcptr xp,
                             mp_srcptr yp, mp_srcptr cp, mp_size_t n,
                             unsigned mode)

    In one pass over n >= 1 limbs, all read as signed n-limb numbers:
      t = (x + y mod B^n) >> s, arithmetic and exact, s = 2 + (mode >> 3
          & 1), y subtracted with mode & 1;
      q = (t + c) / 5 modulo B^n, exact, c subtracted with mode & 2;
      {a1, n} += q and {a2, n} += c - q, c's sign as in q, both
          subtracted instead with mode & 4, where q != -B^n/2.
    q is never stored.  Returns the net carries out of a1 and a2 (what the
    limbs above each change by, -2 to 2) as two 32-bit signed halves, a1's
    low.  a1 and a2 overlap neither each other nor the operands.

    By blocks of 8 limbs (the n - 1 mod 8 limbs first, in blocks of 1, 2
    and 4), the sums in eight registers and the ninth limb (the next
    block's first) on the stack; the chains x +- y, t +- c and a1's along
    CF, a2's two along CF and OF, each parked on the stack between blocks,
    the loop closed on x's end there.  Every memory operand of an adc, sbb
    or adcx and every store is base plus displacement, and no pointer
    moves by a memory operand: three pointers, c's; x's, y's from it and
    back by y - x; a1's, a2's likewise by a2 - a1; the differences in rdx,
    free outside the division ((B-1)/5 is set again before it).  Where q
    is subtracted, R = q by GMP's mpn_bdiv_dbm1c step with (B-1)/5; where
    it is added, R = -q along two chains of one cycle a limb: R[u] = lo +
    the high limb before (adcx) + R[u-1] (adox), each R in its register,
    both carries into the block's last high limb, R's last limb through
    the stack.  Then a1 - R (sbb) and a2 + R (adox) in every mode; -R is q
    but for q = -B^n/2, its own negative.  One copy per mode, entered
    through a tree on its bits.  For the Intel cores."""
    name = "gtc_tc5q8_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name, "GTC_TC5_FA")
    PX, PC, A = "%r10", "%rsi", "%rdi"
    H, LO, HI = "%r11", "%rax", "%rbx"
    R = ["%rbp", "%r12", "%r13", "%r14", "%r15", "%r9", "%rcx", "%r8"]
    # stack: 1 - CF of chains 1 to 3 at 0, 8 and 16; the mask -CF4 at 24,
    # OF5's byte at 32; n - 1 at 40, x's end at 48; y - x at 56, a2 - a1
    # at 64; the ninth sum at 72, R's last limb at 80
    P1, P2, P3, P4, P5, CNT, END, DXY, DA, S8, RL = (0, 8, 16, 24, 32, 40,
                                                      48, 56, 64, 72, 80)
    FRAME = 88
    MODE = FRAME + 56
    emit(f"\tsub\t${FRAME}, %rsp")
    emit("\tmov\t%rcx, %rax")
    emit("\tsub\t%rdx, %rax")
    emit(f"\tmov\t%rax, {DXY}(%rsp)\t/* y - x */")
    emit("\tmov\t%rsi, %rax")
    emit("\tsub\t%rdi, %rax")
    emit(f"\tmov\t%rax, {DA}(%rsp)\t/* a2 - a1 */")
    emit(f"\tmov\t%r8, {PC}")
    emit(f"\tmov\t%rdx, {PX}")
    emit(f"\tmov\t{MODE}(%rsp), %eax")
    emit("\tand\t$15, %eax")
    table = lab("q8t")
    mode_dispatch("%eax", table)
    entries = []
    end = lab("q8e")
    for mode in range(16):
        ysub, csub, asub, sh = (mode & 1, mode >> 1 & 1, mode >> 2 & 1,
                                2 + (mode >> 3 & 1))
        ccmp = csub ^ asub		# c complemented on a2's CF chain
        q2c = not asub			# -q by the two chains
        l = lab("q8m")
        entries.append(l)
        emit(f"{l}:")
        emit("\tlea\t-1(%r9), %r9")
        emit(f"\tmov\t%r9, {CNT}(%rsp)")
        emit(f"\tlea\t8({PX},%r9,8), {LO}")
        emit(f"\tmov\t{LO}, {END}(%rsp)\t/* x's end */")
        emit(f"\tmov\t({PX}), {LO}")
        emit(f"\t{'sub' if ysub else 'add'}\t(%rcx), {LO}\t/* s0 */")
        emit(f"\tsetnc\t{P1}(%rsp)")
        emit(f"\tmov\t{LO}, {S8}(%rsp)")
        emit(f"\tlea\t8({PX}), {PX}")
        emit(f"\tmovq\t$1, {P2}(%rsp)")
        emit(f"\tmovq\t$1, {P3}(%rsp)")
        emit(f"\tmovq\t${-ccmp}, {P4}(%rsp)")
        emit(f"\tmovq\t$0, {P5}(%rsp)")
        emit(f"\tmov\t$0, {H}")
        if q2c:
            emit(f"\tmovq\t$0, {RL}(%rsp)\t/* R before: 0 */")
        emit(f"\tmov\t{DXY}(%rsp), %rdx")

        def front(m):
            """The sums of m limbs, t in R[0..m-1], + c, / 5: q (-q where
            q is added) in R; rdx left with a2 - a1."""
            S = R[1:m + 1] if m < 8 else R[1:] + [HI]
            emit(f"\tmov\t{S8}(%rsp), {R[0]}")
            for u in range(m):
                emit(f"\tmov\t{8 * u}({PX}), {S[u]}")
            emit(f"\tadd\t%rdx, {PX}\t\t/* y */")
            emit(f"\tcmpb\t$1, {P1}(%rsp)")
            for u in range(m):
                emit(f"\t{'sbb' if ysub else 'adc'}\t{8 * u}({PX}), {S[u]}")
            emit(f"\tsetnc\t{P1}(%rsp)")
            emit(f"\tsub\t%rdx, {PX}\t\t/* x */")
            emit(f"\tmov\t{S[m - 1]}, {S8}(%rsp)")
            for u in range(m):
                shr_pair(sh, S[u], R[u], LO)
            emit(f"\tcmpb\t$1, {P2}(%rsp)")
            for u in range(m):
                emit(f"\t{'sbb' if csub else 'adc'}\t{8 * u}({PC}), {R[u]}")
            emit(f"\tsetnc\t{P2}(%rsp)")
            emit("\tmovabs\t$0x3333333333333333, %rdx\t/* (B-1)/5 */")
            if q2c:
                # R[u] = lo + the high limb before (CF) + R[u-1] (OF); the
                # high limbs alternate HI, H, ..., the one before the
                # block in H with both carries; R's last limb in RL
                emit("\txor\t%eax, %eax\t\t/* CF = OF = 0, rax 0 */")
                hp = H
                for u in range(m):
                    hc = HI if u % 2 == 0 else H
                    emit(f"\tmulx\t{R[u]}, {R[u]}, {hc}")
                    emit(f"\tadcx\t{hp}, {R[u]}")
                    prev = f"{RL}(%rsp)" if u == 0 else R[u - 1]
                    emit(f"\tadox\t{prev}, {R[u]}\t/* -q */")
                    hp = hc
                if hp != H:
                    emit(f"\tmov\t{hp}, {H}")
                emit(f"\tadcx\t%rax, {H}\t\t/* both carries in */")
                emit(f"\tadox\t%rax, {H}")
                emit(f"\tmov\t{R[m - 1]}, {RL}(%rsp)")
            else:
                for u in range(m):
                    emit(f"\tmulx\t{R[u]}, {LO}, {HI}")
                    emit(f"\tsub\t{LO}, {H}")
                    emit(f"\tmov\t{H}, {R[u]}")
                    emit(f"\tsbb\t{HI}, {H}")
            emit(f"\tmov\t{DA}(%rsp), %rdx\t/* a2 - a1 */")

        def back(m):
            """The accumulations of m limbs of q in R; rdx left with
            y - x."""
            emit(f"\tcmpb\t$1, {P3}(%rsp)")
            for u in range(m):
                emit(f"\tmov\t{8 * u}({A}), {LO}")
                emit(f"\tsbb\t{R[u]}, {LO}")
                emit(f"\tmov\t{LO}, {8 * u}({A})")
            emit(f"\tsetnc\t{P3}(%rsp)")
            emit(f"\tadd\t%rdx, {A}\t\t/* a2 */")
            emit(f"\tcmpb\t$0x81, {P5}(%rsp)\t/* OF back, CF = 1 */")
            emit(f"\tmov\t$0, {LO}")
            emit(f"\tadcx\t{P4}(%rsp), {LO}\t/* CF back */")
            for u in range(m):
                emit(f"\tmov\t{8 * u}({A}), {LO}")
                cu = f"{8 * u}({PC})"
                if ccmp:
                    emit(f"\tmov\t{cu}, {HI}")
                    emit(f"\tnot\t{HI}")
                    emit(f"\tadcx\t{HI}, {LO}")
                else:
                    emit(f"\tadcx\t{cu}, {LO}")
                emit(f"\tadox\t{R[u]}, {LO}")
                emit(f"\tmov\t{LO}, {8 * u}({A})")
            emit(f"\tseto\t{P5}(%rsp)")
            emit(f"\tsbb\t{HI}, {HI}")
            emit(f"\tmov\t{HI}, {P4}(%rsp)\t/* -CF4 */")
            emit(f"\tsub\t%rdx, {A}\t\t/* a1 */")
            emit(f"\tlea\t{8 * m}({A}), {A}")
            emit(f"\tmov\t{DXY}(%rsp), %rdx\t/* y - x */")

        def block(m):
            front(m)
            back(m)
            emit(f"\tlea\t{8 * m}({PX}), {PX}")
            emit(f"\tlea\t{8 * m}({PC}), {PC}")

        for bit in (0, 1, 2):
            skip = lab("q8b")
            emit(f"\ttestb\t${1 << bit}, {CNT}(%rsp)")
            emit(f"\tjz\t{skip}")
            block(1 << bit)
            emit(f"{skip}:")
        loop, ldone = lab("q8l"), lab("q8d")
        emit(f"\tcmp\t{END}(%rsp), {PX}")
        emit(f"\tje\t{ldone}")
        emit("\t.p2align\t4")
        emit(f"{loop}:")
        block(8)
        emit(f"\tcmp\t{END}(%rsp), {PX}")
        emit(f"\tjne\t{loop}")
        emit(f"{ldone}:")
        # the last limb: the last sum shifted arithmetically, then as
        # front (1) from its shift on and back (1)
        emit(f"\tmov\t{S8}(%rsp), {R[0]}")
        emit(f"\tsar\t${sh}, {R[0]}")
        emit(f"\tcmpb\t$1, {P2}(%rsp)")
        emit(f"\t{'sbb' if csub else 'adc'}\t({PC}), {R[0]}")
        emit(f"\tsetnc\t{P2}(%rsp)")
        emit("\tmovabs\t$0x3333333333333333, %rdx\t/* (B-1)/5 */")
        if q2c:
            emit(f"\tmulx\t{R[0]}, {R[0]}, {HI}")
            emit(f"\tadd\t{H}, {R[0]}")
            emit(f"\tadd\t{RL}(%rsp), {R[0]}\t/* -q */")
        else:
            emit(f"\tmulx\t{R[0]}, {LO}, {HI}")
            emit(f"\tsub\t{LO}, {H}")
            emit(f"\tmov\t{H}, {R[0]}")
        emit(f"\tmov\t{DA}(%rsp), %rdx\t/* a2 - a1 */")
        back(1)
        # the net carries, as where q is subtracted
        emit(f"\tshr\t$63, {R[0]}")
        emit(f"\tmovzbl\t{P3}(%rsp), %eax\t/* 1 - b3 */")
        emit(f"\tlea\t-1(%rax,{R[0]}), %rax")
        emit("\tmov\t%eax, %eax")
        emit(f"\tmov\t({PC}), {HI}")
        emit(f"\tshr\t$63, {HI}\t\t/* tc */")
        emit(f"\tmovzbl\t{P5}(%rsp), %ecx")
        emit(f"\tsub\t{P4}(%rsp), %rcx\t/* CF4 + OF5 */")
        if ccmp:
            emit(f"\tlea\t-1(%rcx,{HI}), %rcx")
        else:
            emit(f"\tsub\t{HI}, %rcx")
        emit(f"\tsub\t{R[0]}, %rcx")
        emit("\tshl\t$32, %rcx")
        emit("\tor\t%rcx, %rax")
        emit(f"\tjmp\t{end}")
    mode_table(table, entries)
    emit(f"{end}:")
    emit(f"\tadd\t${FRAME}, %rsp")
    epilogue(name)


def bfly():
    """mp_limb_t gtc_bfly_n (mp_ptr sp, mp_ptr dp, mp_srcptr xp,
                            mp_srcptr yp, mp_size_t n)

    {sp, n} = x + y and {dp, n} = x - y modulo B^n in one pass, the sum
    along CF (adcx), the difference along OF (adox of y's complement, the
    carry-in 1); n >= 0.  sp may equal xp and dp yp (or the other way
    round).  Returns 0.  Unrolled eight times, entered through a table at
    the step that leaves a multiple of eight behind it, the loop control
    lea and jrcxz."""
    name = "gtc_bfly_n"
    UN = 8
    emit(f"/* {name}: see the header.  */")
    emit(f"\t.globl\t{name}")
    emit(f"\t.type\t{name}, @function")
    emit("\t.p2align\tGTC_TC5_FA")
    emit(f"{name}:")
    ptrs = ["%rdi", "%rsi", "%rdx", "%r10"]
    zero = lab("bz")
    emit("\tmov\t%rcx, %r10\t\t/* yp: rcx counts */")
    emit(f"\tlea\t{UN - 1}(%r8), %rcx")
    emit(f"\tshr\t${UN.bit_length() - 1}, %rcx\t\t/* groups, the first one partial */")
    emit(f"\tjz\t{zero}")
    emit("\tneg\t%r8")
    emit(f"\tand\t${UN - 1}, %r8d\t\t/* the steps the first group skips */")
    emit("\tlea\t(,%r8,8), %r11")
    for p in ptrs:
        emit(f"\tsub\t%r11, {p}")
    entries = [lab("be") for _ in range(UN)]
    table, done = lab("bt"), lab("bd")
    emit(f"\tlea\t{table}(%rip), %rax")
    emit("\tmovslq\t(%rax,%r8,4), %r11")
    emit("\tadd\t%r11, %rax")
    emit("\tmov\t$0x7fffffffffffffff, %r11")
    emit("\tadd\t$1, %r11\t\t/* OF = 1, CF = 0 */")
    emit("\tjmp\t*%rax")
    for u in range(UN):
        o = 8 * u
        emit(f"{entries[u]}:")
        emit(f"\tmov\t{o}(%rdx), %rax")
        emit(f"\tmov\t{o}(%r10), %r9")
        emit("\tmov\t%rax, %r11")
        emit("\tadcx\t%r9, %rax")
        emit("\tnot\t%r9")
        emit("\tadox\t%r9, %r11")
        emit(f"\tmov\t%rax, {o}(%rdi)")
        emit(f"\tmov\t%r11, {o}(%rsi)")
    for p in ptrs:
        emit(f"\tlea\t{8 * UN}({p}), {p}")
    emit("\tlea\t-1(%rcx), %rcx")
    emit(f"\tjrcxz\t{done}")
    emit(f"\tjmp\t{entries[0]}")
    emit(f"{done}:")
    emit(f"{zero}:")
    emit("\txor\t%eax, %eax")
    emit("\tret")
    emit("\t.p2align\t2")
    emit(f"{table}:")
    for e in entries:
        emit(f"\t.long\t{e} - {table}")
    emit(f"\t.size\t{name}, .-{name}")


def tc5s(two_chain=False):
    """mp_limb_t gtc_tc5s_n (mp_ptr r1, mp_ptr r2, mp_srcptr xp,
                            mp_srcptr yp, mp_srcptr cp, mp_size_t n,
                            unsigned mode)

    In one pass over n >= 1 limbs:
      t = (x + y mod B^n) >> 2, arithmetic and exact, y subtracted with
          mode & 1;
      {r1, n} = q = (t + c) / 5 modulo B^n, exact; mode >> 1 = 1: (t -
          c) / 5, 2: (c - t) / 5;
      {r2, n} = c - q modulo B^n.
    Returns 0.  r1 and r2 overlap neither each other nor the operands.
    Three carry chains (x +- y, the numerator, c - q), each along CF and
    parked between blocks of 4 (the n - 1 mod 4 limbs first), the first in
    n's register, the others on the stack; t by shrd (on the Zen by shl,
    shr and or) one limb behind the sums, q by GMP's mpn_bdiv_dbm1c step
    with (B-1)/5.  One copy per mode, entered through a tree on its bits.

    With two_chain, (c - t)/5 along two chains of one cycle a limb: s = t
    - c, the negated numerator, is one sbb in place, and R = R B + s m (m
    = (B-1)/5) is q; the column sums along CF (adcx, the high limbs
    alternating in rax and rbx, the block's last in rbx with both carries)
    and q_u = that + q_(u-1) along OF (adox), each q in its register for c
    - q, the block's last in r11."""
    name = "gtc_tc5s_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name, "GTC_TC5_FA")
    R1, R2, X, Y, C, N = "%rdi", "%rsi", "%r10", "%rcx", "%r8", "%r9"
    H, LO, HI = "%r11", "%rax", "%rbx"
    R = ["%rbp", "%r12", "%r13", "%r14", "%r15"]
    # stack: the parked chains 2 and 3 at 8 and 16 (chain 1 in n's
    # register; 0 unused), n - 1 at 24, x's end at 32
    P1, P2, P3, CNT, END, FRAME = 0, 8, 16, 24, 32, 40
    MODE = FRAME + 56
    emit(f"\tsub\t${FRAME}, %rsp")
    emit(f"\tmov\t%rdx, {X}")
    emit("\tmovabs\t$0x3333333333333333, %rdx\t/* (B-1)/5 */")
    emit(f"\tmov\t{MODE}(%rsp), %eax")
    emit("\tand\t$7, %eax")
    table = lab("q6t")
    mode_dispatch("%eax", table)
    entries = []
    end = lab("q6e")

    def restore(slot):
        if slot == P1:
            emit(f"\tbt\t$0, {N}\t\t/* CF back */")
        else:
            emit(f"\tcmpb\t$1, {slot}(%rsp)\t/* CF back */")

    def park(slot):
        if slot == P1:
            emit(f"\tsbb\t{N}, {N}\t\t/* -CF */")
        else:
            emit(f"\tsetnc\t{slot}(%rsp)\t/* 1 - CF */")

    for mode in range(8):
        ysub, cm = mode & 1, mode >> 1
        if cm == 3:
            entries.append(entries[mode & 1])	# unused
            continue
        l = lab("q6m")
        entries.append(l)
        emit(f"{l}:")
        emit(f"\tlea\t-1({N}), {N}")
        emit(f"\tmov\t{N}, {CNT}(%rsp)")
        emit(f"\tlea\t8({X},{N},8), {LO}")
        emit(f"\tmov\t{LO}, {END}(%rsp)\t/* x's end */")
        emit(f"\tmov\t({X}), {R[0]}")
        emit(f"\t{'sub' if ysub else 'add'}\t({Y}), {R[0]}\t/* s0 */")
        park(P1)
        emit(f"\tlea\t8({X}), {X}")
        emit(f"\tlea\t8({Y}), {Y}")
        emit(f"\tmovq\t$1, {P2}(%rsp)")
        emit(f"\tmovq\t$1, {P3}(%rsp)")
        emit(f"\txor\t{H}d, {H}d")
        q2c = two_chain and cm == 2
        if q2c:
            emit("\txor\t%ebx, %ebx\t\t/* the high limb before: 0 */")

        def rest2c(m):
            """rest's steps for (c - t)/5 by the two chains."""
            restore(P2)
            for u in range(m):
                emit(f"\tsbb\t{8 * u}({C}), {R[u]}\t/* t - c */")
            park(P2)
            emit(f"\ttest\t{LO}, {LO}\t\t/* CF = OF = 0 */")
            hp = HI
            for u in range(m):
                hc = LO if u % 2 == 0 else HI
                emit(f"\tmulx\t{R[u]}, {R[u]}, {hc}")
                emit(f"\tadcx\t{hp}, {R[u]}")
                emit(f"\tadox\t{H if u == 0 else R[u - 1]}, {R[u]}\t/* q */")
                emit(f"\tmov\t{R[u]}, {8 * u}({R1})")
                hp = hc
            emit(f"\tmov\t{R[m - 1]}, {H}")
            if hp != HI:
                emit(f"\tmov\t{hp}, {HI}")
            emit("\tmov\t$0, %eax")
            emit(f"\tadcx\t{LO}, {HI}\t\t/* both carries in */")
            emit(f"\tadox\t{LO}, {HI}")
            restore(P3)
            for u in range(m):
                emit(f"\tmov\t{8 * u}({C}), {LO}")
                emit(f"\tsbb\t{R[u]}, {LO}")
                emit(f"\tmov\t{LO}, {8 * u}({R2})")
            park(P3)
            for p_ in (C, R1, R2):
                emit(f"\tlea\t{8 * m}({p_}), {p_}")

        def rest(m):
            if q2c:
                return rest2c(m)
            restore(P2)
            for u in range(m):
                if cm == 2:
                    emit(f"\tmov\t{8 * u}({C}), {HI}")
                    emit(f"\tsbb\t{R[u]}, {HI}")
                    emit(f"\tmov\t{HI}, {R[u]}")
                else:
                    emit(f"\t{'sbb' if cm else 'adc'}\t{8 * u}({C}), {R[u]}")
            park(P2)
            for u in range(m):
                emit(f"\tmulx\t{R[u]}, {LO}, {HI}")
                emit(f"\tsub\t{LO}, {H}")
                emit(f"\tmov\t{H}, {8 * u}({R1})")
                emit(f"\tmov\t{H}, {R[u]}")
                emit(f"\tsbb\t{HI}, {H}")
            restore(P3)
            for u in range(m):
                emit(f"\tmov\t{8 * u}({C}), {LO}")
                emit(f"\tsbb\t{R[u]}, {LO}")
                emit(f"\tmov\t{LO}, {8 * u}({R2})")
            park(P3)
            for p_ in (C, R1, R2):
                emit(f"\tlea\t{8 * m}({p_}), {p_}")

        def block(m):
            restore(P1)
            for u in range(m):
                emit(f"\tmov\t{8 * u}({X}), {R[u + 1]}")
                emit(f"\t{'sbb' if ysub else 'adc'}\t{8 * u}({Y}), {R[u + 1]}")
            park(P1)
            for u in range(m):
                shr_pair(2, R[u + 1], R[u], LO)
            for p_ in (X, Y):
                emit(f"\tlea\t{8 * m}({p_}), {p_}")
            rest(m)
            emit(f"\tmov\t{R[m]}, {R[0]}")

        for bit in (0, 1):
            skip = lab("q6b")
            emit(f"\ttestb\t${1 << bit}, {CNT}(%rsp)")
            emit(f"\tjz\t{skip}")
            block(1 << bit)
            emit(f"{skip}:")
        loop, ldone = lab("q6l"), lab("q6d")
        emit(f"\tcmp\t{END}(%rsp), {X}")
        emit(f"\tje\t{ldone}")
        emit("\t.p2align\t4")
        emit(f"{loop}:")
        block(4)
        emit(f"\tcmp\t{END}(%rsp), {X}")
        emit(f"\tjne\t{loop}")
        emit(f"{ldone}:")
        emit(f"\tsar\t$2, {R[0]}")
        rest(1)
        emit(f"\tjmp\t{end}")
    mode_table(table, entries)
    emit(f"{end}:")
    emit("\txor\t%eax, %eax")
    emit(f"\tadd\t${FRAME}, %rsp")
    epilogue(name)


def tc5q4():
    """void gtc_tc5q4_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,
                         mp_srcptr q0, mp_srcptr q8, mp_size_t n,
                         mp_size_t n8y)

    {rp, n} = ((x + y mod B^n) >> 2) - q0 - q8 modulo B^n, y subtracted
    with n8y & 1, the shift arithmetic and exact, q8 of n8 = n8y >> 1
    limbs (1 <= n8 <= n) sign-extended.  n >= 1; rp overlaps nothing.
    Three carry chains: x +- y along CF (adc or sbb, parked by setnc,
    back by cmpb $1), the sums one limb ahead and t by shrd (on the Zen by
    shl, shr and or); then q0 + q8 along CF (adcx) and t + ~(q0 + q8)
    along OF (adox, the carry-in 1), parked together (seto and sbb to a
    mask, back by cmpb $0x81 and adcx of the mask into 0).  The limbs of t
    below min (n8, n - 1) first, by the bits of their count in blocks of
    1, 2 and 4, then of 8; then those of q8's sign extension below n - 1
    the same way, q8's pointer at the sign limb on the stack and not
    advanced; then the last limb (the last sum shifted arithmetically),
    q8's pointer at its limb or the sign limb."""
    name = "gtc_tc5q4_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name, "GTC_TC5_FA")
    RP, X, Y, Q0, Q8 = "%rdi", "%rsi", "%rdx", "%rcx", "%r8"
    T = "%rax"
    R = ["%rbx", "%rbp", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15",
         "%r9"]
    # stack: 1 - CF of the sums at 0; the pair's -CF mask at 8 and OF's
    # byte at 16; q8's sign limb at 24; the count of the sign extension's
    # part at 32, of the first part at 40; q8's pointer for the last limb
    # at 48
    P1, P4, P5, SIGN, CB, CA, LASTQ, FRAME = 0, 8, 16, 24, 32, 40, 48, 56
    N8Y = FRAME + 56
    emit(f"\tsub\t${FRAME}, %rsp")
    emit(f"\tmov\t{N8Y}(%rsp), %rax")
    emit("\tmov\t%rax, %r10")
    emit("\tshr\t%rax\t\t\t/* n8 */")
    emit(f"\tmov\t-8({Q8},%rax,8), %r11")
    emit("\tsar\t$63, %r11")
    emit(f"\tmov\t%r11, {SIGN}(%rsp)\t/* q8's sign limb */")
    emit(f"\tlea\t-8({Q8},%r9,8), %r11\t/* q8's limb n - 1, */")
    emit(f"\tlea\t{SIGN}(%rsp), %rbx")
    emit("\tcmp\t%r9, %rax")
    emit("\tcmovne\t%rbx, %r11\t\t/* or the sign limb where n8 < n */")
    emit(f"\tmov\t%r11, {LASTQ}(%rsp)")
    emit("\tlea\t-1(%r9), %r11\t\t/* n - 1 */")
    emit("\tcmp\t%r11, %rax")
    emit("\tcmova\t%r11, %rax\t\t/* the first part: min (n8, n - 1) */")
    emit("\tsub\t%rax, %r11")
    emit(f"\tmov\t%r11, {CB}(%rsp)\t/* the second: n - 1 - that */")
    emit(f"\tmov\t%rax, {CA}(%rsp)")
    emit(f"\tmovq\t$0, {P4}(%rsp)\t/* q0 + q8: CF = 0 */")
    emit(f"\tmovq\t$1, {P5}(%rsp)\t/* the complement's carry-in */")
    minus = lab("q4m")
    end = lab("q4e")
    emit("\ttest\t$1, %r10b")
    emit(f"\tjnz\t{minus}")

    def restore1():
        emit(f"\tcmpb\t$1, {P1}(%rsp)\t/* CF back */")

    def park1():
        emit(f"\tsetnc\t{P1}(%rsp)\t/* 1 - CF */")

    def rest(m, tail):
        """t in R[0..m-1]: - q0 - q8, stored."""
        emit(f"\tcmpb\t$0x81, {P5}(%rsp)\t/* OF back, CF = 1 */")
        emit(f"\tmov\t$0, {T}")
        emit(f"\tadcx\t{P4}(%rsp), {T}\t/* CF back */")
        for u in range(m):
            emit(f"\tmov\t{8 * u}({Q0}), {T}")
            emit(f"\tadcx\t{0 if tail else 8 * u}({Q8}), {T}")
            emit(f"\tnot\t{T}")
            emit(f"\tadox\t{R[u]}, {T}")
            emit(f"\tmov\t{T}, {8 * u}({RP})")
        emit(f"\tseto\t{P5}(%rsp)")
        emit(f"\tsbb\t{T}, {T}")
        emit(f"\tmov\t{T}, {P4}(%rsp)\t/* -CF */")
        for p_ in (Q0, RP) + (() if tail else (Q8,)):
            emit(f"\tlea\t{8 * m}({p_}), {p_}")

    def block(m, ysub, tail):
        restore1()
        for u in range(m):
            emit(f"\tmov\t{8 * u}({X}), {R[u + 1]}")
            emit(f"\t{'sbb' if ysub else 'adc'}\t{8 * u}({Y}), {R[u + 1]}")
        park1()
        for u in range(m):
            shr_pair(2, R[u + 1], R[u], T)
        for p_ in (X, Y):
            emit(f"\tlea\t{8 * m}({p_}), {p_}")
        rest(m, tail)
        emit(f"\tmov\t{R[m]}, {R[0]}")

    for ysub, start in ((0, None), (1, minus)):
        if start:
            emit(f"{start}:")
        emit(f"\tmov\t({X}), {R[0]}")
        emit(f"\t{'sub' if ysub else 'add'}\t({Y}), {R[0]}\t/* s0 */")
        park1()
        emit(f"\tlea\t8({X}), {X}")
        emit(f"\tlea\t8({Y}), {Y}")
        # the first part (q8 from memory); R[8] is n's register, so the
        # counts live on the stack
        for bit in (0, 1, 2):
            skip = lab("q4b")
            emit(f"\ttestb\t${1 << bit}, {CA}(%rsp)")
            emit(f"\tjz\t{skip}")
            block(1 << bit, ysub, False)
            emit(f"{skip}:")
        loop, ldone = lab("q4l"), lab("q4d")
        emit(f"\tshrq\t$3, {CA}(%rsp)")
        emit(f"\tjz\t{ldone}")
        emit("\t.p2align\t4")
        emit(f"{loop}:")
        block(8, ysub, False)
        emit(f"\tsubq\t$1, {CA}(%rsp)")
        emit(f"\tjnz\t{loop}")
        emit(f"{ldone}:")
        # q8's sign extension below n - 1: the pointer at the sign limb
        emit(f"\tlea\t{SIGN}(%rsp), {Q8}")
        for bit in (0, 1, 2):
            skip = lab("q4b")
            emit(f"\ttestb\t${1 << bit}, {CB}(%rsp)")
            emit(f"\tjz\t{skip}")
            block(1 << bit, ysub, True)
            emit(f"{skip}:")
        loop, ldone = lab("q4l"), lab("q4d")
        emit(f"\tshrq\t$3, {CB}(%rsp)")
        emit(f"\tjz\t{ldone}")
        emit(f"{loop}:")
        block(8, ysub, True)
        emit(f"\tsubq\t$1, {CB}(%rsp)")
        emit(f"\tjnz\t{loop}")
        emit(f"{ldone}:")
        # the last limb
        emit(f"\tmov\t{LASTQ}(%rsp), {Q8}")
        emit(f"\tsar\t$2, {R[0]}")
        rest(1, True)
        emit(f"\tjmp\t{end}")
    emit(f"{end}:")
    emit(f"\tadd\t${FRAME}, %rsp")
    epilogue(name)


def bflyq():
    """mp_limb_t gtc_bflyq_n (mp_ptr sp, mp_ptr dp, mp_srcptr xp,
                             mp_srcptr yp, mp_size_t n)

    {sp, n} = (x + y mod B^n) >> 2 and {dp, n} = (x - y mod B^n) >> 2,
    the shifts arithmetic and exact, in one pass; n >= 1.  sp may equal xp
    and dp yp (or the other way round).  Returns 0.  Per block of 4 limbs
    (the n - 1 mod 4 first, in blocks of 1 and 2 by its bits): x into
    S[1..m] and D[1..m], the sums along CF (adc), the chain parked (setnc,
    cmpb $1), the differences likewise (sbb), the results one limb behind
    the sums (shrd), stored."""
    name = "gtc_bflyq_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name, "GTC_TC5_FA")
    SP, DP, X, Y, N = "%rdi", "%rsi", "%rdx", "%rcx", "%r8"
    S = ["%rax", "%rbx", "%rbp", "%r9", "%r10"]
    D = ["%r11", "%r12", "%r13", "%r14", "%r15"]
    # stack: the two parked chains (16 unused)
    PS, PD, FRAME = 0, 8, 24
    emit(f"\tsub\t${FRAME}, %rsp")
    emit(f"\tmov\t({X}), {S[0]}")
    emit(f"\tmov\t{S[0]}, {D[0]}")
    emit(f"\tadd\t({Y}), {S[0]}")
    emit(f"\tsetnc\t{PS}(%rsp)")
    emit(f"\tsub\t({Y}), {D[0]}")
    emit(f"\tsetnc\t{PD}(%rsp)")
    emit(f"\tlea\t8({X}), {X}")
    emit(f"\tlea\t8({Y}), {Y}")
    emit(f"\tlea\t-1({N}), {N}")

    def block(m):
        emit(f"\tcmpb\t$1, {PS}(%rsp)\t/* CF back */")
        for u in range(m):
            emit(f"\tmov\t{8 * u}({X}), {S[u + 1]}")
            emit(f"\tmov\t{S[u + 1]}, {D[u + 1]}")
            emit(f"\tadc\t{8 * u}({Y}), {S[u + 1]}")
        emit(f"\tsetnc\t{PS}(%rsp)")
        emit(f"\tcmpb\t$1, {PD}(%rsp)")
        for u in range(m):
            emit(f"\tsbb\t{8 * u}({Y}), {D[u + 1]}")
        emit(f"\tsetnc\t{PD}(%rsp)")
        for u in range(m):
            emit(f"\tshrd\t$2, {S[u + 1]}, {S[u]}")
            emit(f"\tmov\t{S[u]}, {8 * u}({SP})")
        for u in range(m):
            emit(f"\tshrd\t$2, {D[u + 1]}, {D[u]}")
            emit(f"\tmov\t{D[u]}, {8 * u}({DP})")
        emit(f"\tmov\t{S[m]}, {S[0]}")
        emit(f"\tmov\t{D[m]}, {D[0]}")
        for p_ in (SP, DP, X, Y):
            emit(f"\tlea\t{8 * m}({p_}), {p_}")

    for bit in (0, 1):
        skip = lab("bqb")
        emit(f"\ttest\t${1 << bit}, {N}b")
        emit(f"\tjz\t{skip}")
        block(1 << bit)
        emit(f"{skip}:")
    loop, done = lab("bql"), lab("bqd")
    emit(f"\tshr\t$2, {N}")
    emit(f"\tjz\t{done}")
    emit("\t.p2align\t4")
    emit(f"{loop}:")
    block(4)
    emit(f"\tdec\t{N}")
    emit(f"\tjnz\t{loop}")
    emit(f"{done}:")
    emit(f"\tsar\t$2, {S[0]}")
    emit(f"\tmov\t{S[0]}, ({SP})")
    emit(f"\tsar\t$2, {D[0]}")
    emit(f"\tmov\t{D[0]}, ({DP})")
    emit("\txor\t%eax, %eax")
    emit(f"\tadd\t${FRAME}, %rsp")
    epilogue(name)


def tc5a():
    """void gtc_tc5a_n (mp_ptr rp, mp_srcptr q0, mp_srcptr q4,
                        mp_srcptr q8, mp_size_t n, mp_size_t n8)

    {rp, n} = 2 (q0 + 4 (4 q8 - q4)) modulo B^n, q8 of n8 limbs (1 <= n8
    <= n) sign-extended: t = 4 q8 - q4 along CF (q4's complement, the
    carry-in 1), u = q0 + 4 t along OF, and 2 u, each shifted limb by shrx
    of the limb below and lea with the scale, which leave the flags alone,
    so that neither chain is ever parked.  Unrolled four times over q8's
    limbs, entered through a table, then a limb a turn over q8's sign
    extension; the loop control lea and jrcxz."""
    name = "gtc_tc5a_n"
    UN = 4
    emit(f"/* {name}: see the header.  */")
    prologue(name, "GTC_TC5_FA")
    RP, Q0, Q4, Q8 = "%rdi", "%rsi", "%rdx", "%r10"
    PQ8, PT, PU, LO, S, X = "%r11", "%rbx", "%rbp", "%r12", "%r13", "%r14"
    C62, C63 = "%r15", "%rax"
    emit(f"\tmov\t%rcx, {Q8}")
    emit("\tsub\t%r9, %r8\t\t/* the limbs above q8 */")
    emit("\tlea\t3(%r9), %rcx")
    emit("\tshr\t$2, %rcx\t\t/* turns over q8, the first partial */")
    emit("\tneg\t%r9")
    emit(f"\tand\t${UN - 1}, %r9d\t\t/* the steps the first one skips */")
    emit(f"\tlea\t(,%r9,8), {LO}")
    for p_ in (RP, Q0, Q4, Q8):
        emit(f"\tsub\t{LO}, {p_}")
    entries = [lab("ae") for _ in range(UN)]
    table, tail, tl, done = lab("at"), lab("aw"), lab("al"), lab("ad")
    emit(f"\tlea\t{table}(%rip), {S}")
    emit(f"\tmovslq\t({S},%r9,4), {X}")
    emit(f"\tadd\t{X}, {S}")
    emit(f"\tmov\t$62, {C62}d")
    emit("\tmov\t$63, %eax")
    emit(f"\txor\t{PQ8}d, {PQ8}d")
    emit("\txor\t%ebx, %ebx")
    emit("\txor\t%ebp, %ebp\t\t/* the limbs below: 0; CF = OF = 0 */")
    emit("\tstc")
    emit(f"\tjmp\t*{S}")

    def step(o, q8):
        """One limb at offset o, q8's from memory or the sign (q8 a
        register)."""
        emit(f"\tshrx\t{C62}, {PQ8}, {LO}")
        emit(f"\tmov\t{q8}, {PQ8}")
        emit(f"\tlea\t({LO},{PQ8},4), {S}\t/* 4 q8 */")
        emit(f"\tmov\t{o}({Q4}), {X}")
        emit(f"\tnot\t{X}")
        emit(f"\tadcx\t{X}, {S}\t\t/* t */")
        emit(f"\tshrx\t{C62}, {PT}, {LO}")
        emit(f"\tmov\t{S}, {PT}")
        emit(f"\tlea\t({LO},{S},4), {S}\t/* 4 t */")
        emit(f"\tadox\t{o}({Q0}), {S}\t/* u */")
        emit(f"\tshrx\t{C63}, {PU}, {LO}")
        emit(f"\tmov\t{S}, {PU}")
        emit(f"\tlea\t({LO},{S},2), {S}\t/* 2 u */")
        emit(f"\tmov\t{S}, {o}({RP})")

    for u in range(UN):
        emit(f"{entries[u]}:")
        step(8 * u, f"{8 * u}({Q8})")
    for p_ in (RP, Q0, Q4, Q8):
        emit(f"\tlea\t{8 * UN}({p_}), {p_}")
    emit("\tlea\t-1(%rcx), %rcx")
    emit(f"\tjrcxz\t{tail}")
    emit(f"\tjmp\t{entries[0]}")
    emit(f"{tail}:")
    emit("\tmov\t%r8, %rcx")
    emit(f"\tsarx\t{C63}, {PQ8}, {Q8}\t/* q8's sign */")
    emit(f"\tjrcxz\t{done}")
    emit(f"{tl}:")
    step(0, Q8)
    for p_ in (RP, Q0, Q4):
        emit(f"\tlea\t8({p_}), {p_}")
    emit("\tlea\t-1(%rcx), %rcx")
    emit(f"\tjrcxz\t{done}")
    emit(f"\tjmp\t{tl}")
    emit(f"{done}:")
    for r in ["r15", "r14", "r13", "r12", "rbp", "rbx"]:
        emit(f"\tpop\t%{r}")
    emit("\tret")
    emit("\t.p2align\t2")
    emit(f"{table}:")
    for e in entries:
        emit(f"\t.long\t{e} - {table}")
    emit(f"\t.size\t{name}, .-{name}")


def tc5uys():
    """void gtc_tc5uys_n (mp_ptr rp, mp_ptr zp, mp_srcptr q0, mp_srcptr q4,
                          mp_srcptr q8, mp_size_t n, mp_size_t n8m,
                          mp_ptr up, mp_srcptr vp)

    In one pass, all modulo B^n, q8 of n8 = n8m >> 3 limbs (1 <= n8 <= n)
    sign-extended: {rp, n} = X = 2 (q0 + 4 (4 q8 - q4)) - z, or + z with
    n8m & 1; {up, n} = Y = u - X, or u + X with n8m & 2; {zp, n} = Y + v,
    or Y - v with n8m & 4, minus z where n8m's bits 0 and 1 are equal,
    else plus z.  tc5uy's blocks with zp's two chains after u's (v from
    the pointer on the stack, z again), along CF, each parked as the
    others.
    One copy per mode, entered through a tree on its bits."""
    tc5uy(with_s=True)


def tc5uy(with_s=False):
    """void gtc_tc5uy_n (mp_ptr rp, mp_srcptr zp, mp_srcptr q0, mp_srcptr q4,
                         mp_srcptr q8, mp_size_t n, mp_size_t n8zu,
                         mp_ptr up)

    {rp, n} = 2 (q0 + 4 (4 q8 - q4)) - z, or + z with n8zu & 1, modulo
    B^n, and in the same pass {up, n} = u - {rp, n} over u, or u + {rp,
    n} with n8zu & 2, q8 of n8 = n8zu >> 2 limbs (1 <= n8 <= n)
    sign-extended.  By Horner's rule in blocks of eight limbs held in
    registers through the three steps t <- x +- 2^s t, s = 2, 2, 1, the
    n8 mod 8 limbs first (blocks of 1, 2 and 4 by n8's bits), then the
    n - n8 limbs of q8's sign extension the same way, the sign in q8's
    pointer register.  A step shifts t by shld $s from the limb below, the
    limbs from the top down (the block's top limb kept on the stack for
    the next block's first), then adds or subtracts x along CF (adc, sbb),
    the chain restored at the step's start (cmpb $1) and parked at its end
    (setnc).  u's is a fourth chain per block after rp's stores (sbb from
    u's limb, or adc of it), parked as the others; the loop counts on the
    stack, so that u's pointer has n's register.  rp may equal zp.  With
    with_s, gtc_tc5uys_n."""
    name = "gtc_tc5uys_n" if with_s else "gtc_tc5uy_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name, "GTC_TC5_FA")
    T = ["%rax", "%rbx", "%rbp", "%r10", "%r11", "%r12", "%r13", "%r14"]
    TMP = "%r15"
    RP, Z, Q0, Q4, Q8 = "%rdi", "%rsi", "%rdx", "%rcx", "%r8"
    BLK = 8
    SH = {1: 2, 2: 2, 3: 1}
    SRC = {1: Q4, 2: Q0, 3: Z}
    # stack: the parked chains (1 - CF) at 8 (k - 1); the limb below a
    # step's block at 24 + 8 (k - 1); the tail's count at 48; the sign of
    # q8 at 56 (64 unused); Y's parked chain at 72, the loop count at 80;
    # with S its chains at 88 and 96, v's pointer at 104; u's pointer the
    # eighth argument, v's the ninth
    CF = {1: 0, 2: 8, 3: 16}
    PRV = {1: 24, 2: 32, 3: 40}
    TAIL, SIGN, FRAME = 48, 56, 88
    CFY, CNTM, UP = 72, 80, "%r9"
    CFS1, CFS2, VPS = 88, 96, 104
    if with_s:
        FRAME = 120
    N8Z, UPARG, VPARG = FRAME + 56, FRAME + 64, FRAME + 72

    def block(m, ops, tail, yadd=0, vop=None, zsop=None):
        for u in range(m):
            if tail:
                emit(f"\tmov\t{Q8}, {T[u]}")
            else:
                emit(f"\tmov\t{8 * u}({Q8}), {T[u]}")
        for k in (1, 2, 3):
            s = SH[k]
            emit(f"\tmov\t{PRV[k]}(%rsp), {TMP}\t/* the limb below */")
            emit(f"\tmov\t{T[m - 1]}, {PRV[k]}(%rsp)")
            for u in range(m - 1, -1, -1):
                below = T[u - 1] if u else TMP
                emit(f"\tshld\t${s}, {below}, {T[u]}")
            emit(f"\tcmpb\t$1, {CF[k]}(%rsp)\t/* CF back */")
            for u in range(m):
                emit(f"\t{ops[k]}\t{8 * u}({SRC[k]}), {T[u]}")
            emit(f"\tsetnc\t{CF[k]}(%rsp)\t/* 1 - CF */")
        for u in range(m):
            emit(f"\tmov\t{T[u]}, {8 * u}({RP})")
        emit(f"\tcmpb\t$1, {CFY}(%rsp)\t/* CF back */")
        if with_s:
            # Y over -U's registers, then S = Y +- v -+ z, over z
            for u in range(m):
                if not yadd:
                    emit(f"\tnot\t{T[u]}")
                emit(f"\tadc\t{8 * u}({UP}), {T[u]}")
                emit(f"\tmov\t{T[u]}, {8 * u}({UP})")
            emit(f"\tsetnc\t{CFY}(%rsp)\t/* 1 - CF */")
            emit(f"\tmov\t{VPS}(%rsp), {TMP}")
            emit(f"\tcmpb\t$1, {CFS1}(%rsp)\t/* CF back */")
            for u in range(m):
                emit(f"\t{vop}\t{8 * u}({TMP}), {T[u]}")
            emit(f"\tsetnc\t{CFS1}(%rsp)\t/* 1 - CF */")
            emit(f"\tcmpb\t$1, {CFS2}(%rsp)\t/* CF back */")
            for u in range(m):
                emit(f"\t{zsop}\t{8 * u}({Z}), {T[u]}")
            emit(f"\tsetnc\t{CFS2}(%rsp)\t/* 1 - CF */")
            for u in range(m):
                emit(f"\tmov\t{T[u]}, {8 * u}({Z})")
            emit(f"\tlea\t{8 * m}({TMP}), {TMP}")
            emit(f"\tmov\t{TMP}, {VPS}(%rsp)")
            for p_ in (RP, Z, Q0, Q4) + (() if tail else (Q8,)) + (UP,):
                emit(f"\tlea\t{8 * m}({p_}), {p_}")
            return
        for u in range(m):
            if yadd:
                emit(f"\tadc\t{8 * u}({UP}), {T[u]}")
                emit(f"\tmov\t{T[u]}, {8 * u}({UP})")
            else:
                emit(f"\tmov\t{8 * u}({UP}), {TMP}")
                emit(f"\tsbb\t{T[u]}, {TMP}")
                emit(f"\tmov\t{TMP}, {8 * u}({UP})")
        emit(f"\tsetnc\t{CFY}(%rsp)\t/* 1 - CF */")
        for p_ in (RP, Z, Q0, Q4) + (() if tail else (Q8,)) + (UP,):
            emit(f"\tlea\t{8 * m}({p_}), {p_}")

    emit(f"\tsub\t${FRAME}, %rsp")
    emit(f"\tmov\t{N8Z}(%rsp), %rax")
    emit("\tmov\t%rax, %r10")
    emit(f"\tshr\t${3 if with_s else 2}, %rax\t\t/* n8 */")
    emit("\tsub\t%rax, %r9")
    emit(f"\tmov\t%r9, {TAIL}(%rsp)\t/* n - n8 */")
    emit(f"\tmov\t-8({Q8},%rax,8), %rbx")
    emit("\tsar\t$63, %rbx")
    emit(f"\tmov\t%rbx, {SIGN}(%rsp)\t/* q8's sign limb */")
    emit(f"\tmov\t%rax, {CNTM}(%rsp)")
    emit(f"\tmovq\t$1, {CFY}(%rsp)\t/* CF = 0 */")
    emit(f"\tmov\t{UPARG}(%rsp), {UP}")
    for k in (1, 2, 3):
        emit(f"\tmovq\t$1, {CF[k]}(%rsp)\t/* CF = 0 */")
        emit(f"\tmovq\t$0, {PRV[k]}(%rsp)")
    done = lab("u5e")
    if with_s:
        # mode m: z added (bit 0), -U' added (bit 1), v subtracted (bit 2)
        copies = [("adc" if m & 1 else "sbb", m >> 1 & 1,
                   "sbb" if m & 4 else "adc",
                   "sbb" if (m & 1) == (m >> 1 & 1) else "adc")
                  for m in range(8)]
        starts = [lab("u5p") for _ in copies]
        emit(f"\tmov\t{VPARG}(%rsp), {TMP}")
        emit(f"\tmov\t{TMP}, {VPS}(%rsp)\t/* v's pointer */")
        emit(f"\tmovq\t$1, {CFS1}(%rsp)\t/* CF = 0 */")
        emit(f"\tmovq\t$1, {CFS2}(%rsp)\t/* CF = 0 */")
        emit("\tmov\t%r10d, %eax")
        emit("\tand\t$7, %eax")
        table = lab("u5t")
        mode_dispatch("%eax", table)
    else:
        copies = [("sbb", 0, None, None), ("adc", 0, None, None),
                  ("sbb", 1, None, None), ("adc", 1, None, None)]
        starts = [None] + [lab("u5p") for _ in copies[1:]]
        emit("\ttest\t$2, %r10b")
        emit(f"\tjnz\t{starts[2]}")
        emit("\ttest\t$1, %r10b")
        emit(f"\tjnz\t{starts[1]}")
    for (zop, yadd, vop, zsop), start in zip(copies, starts):
        if start:
            emit(f"{start}:")
        if with_s:
            # Y's carry-in: 1 where -U' is subtracted (its complement)
            emit(f"\tmovq\t${1 if yadd else 0}, {CFY}(%rsp)")
        elif start and yadd and zop == "sbb":
            emit("\ttest\t$1, %r10b")
            emit(f"\tjnz\t{starts[3]}")
        ops = {1: "sbb", 2: "adc", 3: zop}
        for tail in (False, True):
            if tail:
                emit(f"\tmov\t{TAIL}(%rsp), {TMP}")
                emit(f"\tmov\t{TMP}, {CNTM}(%rsp)")
                emit(f"\ttest\t{TMP}, {TMP}")
                emit(f"\tjz\t{done}")
                emit(f"\tmov\t{SIGN}(%rsp), {Q8}")
            for bit in range(BLK.bit_length() - 1):
                skip = lab("u5b")
                emit(f"\ttestb\t${1 << bit}, {CNTM}(%rsp)")
                emit(f"\tjz\t{skip}")
                block(1 << bit, ops, tail, yadd, vop, zsop)
                emit(f"{skip}:")
            loop, ldone = lab("u5l"), lab("u5d")
            emit(f"\tshrq\t${BLK.bit_length() - 1}, {CNTM}(%rsp)")
            emit(f"\tjz\t{ldone}")
            emit("\t.p2align\t4")
            emit(f"{loop}:")
            block(BLK, ops, tail, yadd, vop, zsop)
            emit(f"\tsubq\t$1, {CNTM}(%rsp)")
            emit(f"\tjnz\t{loop}")
            emit(f"{ldone}:")
        emit(f"\tjmp\t{done}")
    emit(f"{done}:")
    emit(f"\tadd\t${FRAME}, %rsp")
    if with_s:
        for r in ["r15", "r14", "r13", "r12", "rbp", "rbx"]:
            emit(f"\tpop\t%{r}")
        emit("\tret")
        mode_table(table, starts)
        emit(f"\t.size\t{name}, .-{name}")
        return
    epilogue(name)


def tc5quad():
    """void gtc_tc5quad_n (mp_ptr ap, mp_ptr bp, mp_ptr cp, mp_ptr dp,
                           mp_size_t n, unsigned mode)

    With P = a + b, M = a - b, Q = c + d, N = c - d (mod B^n): G = P and
    H1 = M with mode & 1 (else G = M, H1 = P), D = Q and H2 = N with mode
    & 2 (else D = N, H2 = Q); then a = H1, c = H2, b = (G - D) >> 2 and d
    = (G + D) >> 2 with mode & 4 (else b = (G + D) >> 2, d = (G - D) >>
    2), the shifts arithmetic, all in place.  n >= 1.  Per block of 3 limbs
    (11 registers of data): c into two sets, Q along CF (adc) and N (sbb),
    each chain parked (setnc, cmpb $1); H2 stored, D kept; a and b
    likewise, H1 stored, G kept; G + D into the free set (adc), G - D over
    G (sbb); each shifted (shrd, on the Zen shl, shr and or) and stored a
    limb behind.  The first limb before the blocks, n - 1 by blocks of 3
    and the rest by its bits after them, the last limb's shifts at the end;
    the count on the stack.  One copy per mode, entered through a tree on
    its bits."""
    name = "gtc_tc5quad_n"
    emit(f"/* {name}: see the header.  */")
    prologue(name, "GTC_TC5_FA")
    A, B, C, D = "%rdi", "%rsi", "%rdx", "%rcx"
    SETS = [["%rax", "%rbx", "%rbp"], ["%r8", "%r9", "%r10"],
            ["%r11", "%r12", "%r13"]]
    XP, YP = "%r14", "%r15"
    KQ, KN, KP, KM, KS, KD, CNT, FRAME = 0, 8, 16, 24, 32, 40, 48, 56
    emit(f"\tsub\t${FRAME}, %rsp")
    emit("\tlea\t-1(%r8), %rax")
    emit(f"\tmov\t%rax, {CNT}(%rsp)\t/* n - 1 */")
    table, end = lab("qdt"), lab("qde")
    emit("\tand\t$7, %r9d")
    mode_dispatch("%r9d", table)

    def chain(park, first, ops):
        if not first:
            emit(f"\tcmpb\t$1, {park}(%rsp)\t/* CF back */")
        for o in ops:
            emit(o)
        emit(f"\tsetnc\t{park}(%rsp)")

    def block(m, mode, first=False):
        """m limbs; first: limb 0, the sum and difference of G and D kept
        unshifted."""
        m0, m1, m2 = mode & 1, mode >> 1 & 1, mode >> 2 & 1
        s1, s2, s3 = SETS
        ad, sb = ("add", "sub") if first else ("adc", "sbb")
        for u in range(m):
            emit(f"\tmov\t{8 * u}({C}), {s1[u]}")
            emit(f"\tmov\t{s1[u]}, {s2[u]}")
        chain(KQ, first, [f"\t{ad}\t{8 * u}({D}), {s1[u]}" for u in range(m)])
        chain(KN, first, [f"\t{sb}\t{8 * u}({D}), {s2[u]}" for u in range(m)])
        dd, h2 = (s1, s2) if m1 else (s2, s1)
        for u in range(m):
            emit(f"\tmov\t{h2[u]}, {8 * u}({C})")
        # a into the free set (P) and s3 (M)
        for u in range(m):
            emit(f"\tmov\t{8 * u}({A}), {h2[u]}")
            emit(f"\tmov\t{h2[u]}, {s3[u]}")
        chain(KP, first, [f"\t{ad}\t{8 * u}({B}), {h2[u]}" for u in range(m)])
        chain(KM, first, [f"\t{sb}\t{8 * u}({B}), {s3[u]}" for u in range(m)])
        gg, h1 = (h2, s3) if m0 else (s3, h2)
        for u in range(m):
            emit(f"\tmov\t{h1[u]}, {8 * u}({A})")
        # G + D into H1's set, G - D over G
        xs = h1
        for u in range(m):
            emit(f"\tmov\t{gg[u]}, {xs[u]}")
        chain(KS, first, [f"\t{ad}\t{dd[u]}, {xs[u]}" for u in range(m)])
        chain(KD, first, [f"\t{sb}\t{dd[u]}, {gg[u]}" for u in range(m)])
        bs, ds = (gg, xs) if m2 else (xs, gg)
        if first:
            emit(f"\tmov\t{bs[0]}, {XP}")
            emit(f"\tmov\t{ds[0]}, {YP}")
        else:
            for prev, cur, P_ in ((XP, bs, B), (YP, ds, D)):
                shr_pair(2, cur[0], prev, dd[0])
                emit(f"\tmov\t{prev}, -8({P_})")
                for u in range(m - 1):
                    shr_pair(2, cur[u + 1], cur[u], dd[0])
                    emit(f"\tmov\t{cur[u]}, {8 * u}({P_})")
                emit(f"\tmov\t{cur[m - 1]}, {prev}")
        for p_ in (A, B, C, D):
            emit(f"\tlea\t{8 * m}({p_}), {p_}")

    entries = []
    for mode in range(8):
        l = lab("qdm")
        entries.append(l)
        emit(f"{l}:")
        block(1, mode, first=True)
        loop, tail = lab("qdl"), lab("qdr")
        emit(f"\tsubq\t$3, {CNT}(%rsp)")
        emit(f"\tjb\t{tail}")
        emit("\t.p2align\t4")
        emit(f"{loop}:")
        block(3, mode)
        emit(f"\tsubq\t$3, {CNT}(%rsp)")
        emit(f"\tjae\t{loop}")
        emit(f"{tail}:")
        emit(f"\taddq\t$3, {CNT}(%rsp)\t/* the rest, 0 to 2 */")
        for b in (1, 0):
            skip = lab("qdb")
            emit(f"\ttestb\t${1 << b}, {CNT}(%rsp)")
            emit(f"\tjz\t{skip}")
            block(1 << b, mode)
            emit(f"{skip}:")
        emit(f"\tsar\t$2, {XP}")
        emit(f"\tmov\t{XP}, -8({B})")
        emit(f"\tsar\t$2, {YP}")
        emit(f"\tmov\t{YP}, -8({D})")
        emit(f"\tjmp\t{end}")
    emit(f"{end}:")
    emit(f"\tadd\t${FRAME}, %rsp")
    for r in ["r15", "r14", "r13", "r12", "rbp", "rbx"]:
        emit(f"\tpop\t%{r}")
    emit("\tret")
    mode_table(table, entries)
    emit(f"\t.size\t{name}, .-{name}")


def gap():
    """32 bytes of padding, which keep the kernels after it at their
    offsets modulo 64 (their places in the 64-byte fetch windows)."""
    emit("\t.p2align\t5")
    emit("\t.skip\t32, 0xcc")


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
    emit("   mp_limb_t gtc_div15n_lsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,")
    emit("                               mp_srcptr zp, mp_size_t n, unsigned signs)")
    emit("")
    emit("   {rp, n} = -(x + 4 y + 8 z) / 15 modulo B^n, exact, signs as")
    emit("   above, the division along two carry chains.  Returns the result's")
    emit("   top limb (0 for n = 0).  n >= 0; rp may equal xp.")
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
    emit("   mp_limb_t gtc_div5n_rlsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,")
    emit("                               mp_size_t n, unsigned sub)")
    emit("")
    emit("   {rp, n} = -(x + 4 y) / 5, or (x - 4 y) / 5 with sub & 1, modulo")
    emit("   B^n, the division exact, along two carry chains; the limbs of 4 y")
    emit("   made by shld.  Returns the result's top limb (0 for n = 0).")
    emit("   n >= 0; rp may equal xp.")
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
    emit("")
    emit("   gtc_rsh1add_n and gtc_rsh1sub_n shifted by 2 bits instead of 1;")
    emit("   they return the bits shifted out.")
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
    emit("   mp_limb_t gtc_aorslsh_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,")
    emit("                            mp_size_t n, unsigned s, unsigned mode)")
    emit("")
    emit("   {rp, n} = x + 2^s y (mode 0), x - 2^s y (1) or 2^s y - x (2),")
    emit("   1 <= s <= 63; returns the limb above (or the borrow from it) as")
    emit("   mpn_addmul_1 and mpn_submul_1 by 2^s, and as mpn_rsblsh_n with")
    emit("   mode 2 (the shifted-out bits minus the borrow).  n >= 0; rp may")
    emit("   equal xp.  In place of GMP's rsblsh1, rsblsh2 and addlsh2 (gtc.c,")
    emit("   GTC_RSBLSH1_N, GTC_RSBLSH2_N, GTC_ADDLSH2_N).")
    emit("")
    emit("   mp_limb_t gtc_aors2c_n (mp_ptr d, mp_ptr zp, mp_srcptr xp,")
    emit("                           mp_srcptr yp, mp_size_t n, unsigned signs)")
    emit("")
    emit("   {d, n} = x +- y along CF and {zp, n} = d +- z along OF in one")
    emit("   pass (signs & 1, & 2 subtract); returns CF + 2 OF.  n >= 0; d")
    emit("   may equal xp or yp.")
    emit("")
    emit("   mp_limb_t gtc_tc5q_n (mp_ptr a1, mp_ptr a2, mp_srcptr xp,")
    emit("                         mp_srcptr yp, mp_srcptr cp, mp_size_t n,")
    emit("                         unsigned mode)")
    emit("")
    emit("   t = (x +- y mod B^n) >> s (arithmetic, exact; s = 2 + (mode >> 3")
    emit("   & 1), y subtracted with mode & 1), q = (t +- c) / 5 (exact, c")
    emit("   subtracted with mode & 2), {a1, n} += q and {a2, n} += c - q")
    emit("   (both subtracted with mode & 4), q never stored.  Returns the net")
    emit("   signed carries out of a1 and a2 as two 32-bit halves, a1's low.")
    emit("   n >= 1.")
    emit("")
    emit("   mp_limb_t gtc_tc5q8_n (mp_ptr a1, mp_ptr a2, mp_srcptr xp,")
    emit("                          mp_srcptr yp, mp_srcptr cp, mp_size_t n,")
    emit("                          unsigned mode)")
    emit("")
    emit("   As gtc_tc5q_n, by blocks of 8 limbs; where q is added, q !=")
    emit("   -B^n/2.")
    emit("")
    emit("   mp_limb_t gtc_tc5s_n (mp_ptr r1, mp_ptr r2, mp_srcptr xp,")
    emit("                         mp_srcptr yp, mp_srcptr cp, mp_size_t n,")
    emit("                         unsigned mode)")
    emit("")
    emit("   t = (x +- y mod B^n) >> 2 (y subtracted with mode & 1), {r1, n} =")
    emit("   q = (t + c) / 5, (t - c) / 5 or (c - t) / 5 by mode >> 1, {r2, n}")
    emit("   = c - q.  Returns 0.  n >= 1.")
    emit("")
    emit("   mp_limb_t gtc_bfly_n (mp_ptr sp, mp_ptr dp, mp_srcptr xp,")
    emit("                         mp_srcptr yp, mp_size_t n)")
    emit("")
    emit("   {sp, n} = x + y and {dp, n} = x - y modulo B^n in one pass; sp")
    emit("   may equal xp and dp yp, or the other way round.  Returns 0.")
    emit("   n >= 0.")
    emit("")
    emit("   void gtc_tc5q4_n (mp_ptr rp, mp_srcptr xp, mp_srcptr yp,")
    emit("                     mp_srcptr q0, mp_srcptr q8, mp_size_t n,")
    emit("                     mp_size_t n8y)")
    emit("")
    emit("   {rp, n} = ((x +- y mod B^n) >> 2) - q0 - q8 modulo B^n, y")
    emit("   subtracted with n8y & 1, q8 of n8 = n8y >> 1 limbs (1 <= n8 <= n)")
    emit("   sign-extended.  n >= 1.")
    emit("")
    emit("   mp_limb_t gtc_bflyq_n (mp_ptr sp, mp_ptr dp, mp_srcptr xp,")
    emit("                          mp_srcptr yp, mp_size_t n)")
    emit("")
    emit("   {sp, n} = (x + y mod B^n) >> 2 and {dp, n} = (x - y mod B^n) >> 2,")
    emit("   arithmetic and exact, in one pass; sp may equal xp and dp yp, or")
    emit("   the other way round.  Returns 0.  n >= 1.")
    emit("")
    emit("   void gtc_tc5a_n (mp_ptr rp, mp_srcptr q0, mp_srcptr q4,")
    emit("                    mp_srcptr q8, mp_size_t n, mp_size_t n8)")
    emit("")
    emit("   {rp, n} = A = 2 (q0 + 4 (4 q8 - q4)) modulo B^n, q8 of n8 limbs")
    emit("   (1 <= n8 <= n) sign-extended.")
    emit("")
    emit("   void gtc_tc5uy_n (mp_ptr rp, mp_srcptr zp, mp_srcptr q0,")
    emit("                     mp_srcptr q4, mp_srcptr q8, mp_size_t n,")
    emit("                     mp_size_t n8zu, mp_ptr up)")
    emit("")
    emit("   {rp, n} = 2 (q0 + 4 (4 q8 - q4)) - z, or + z with n8zu & 1,")
    emit("   modulo B^n, and {up, n} = u - {rp, n}, or u + {rp, n} with")
    emit("   n8zu & 2, q8 of n8 = n8zu >> 2 limbs (1 <= n8 <= n)")
    emit("   sign-extended.  rp may equal zp.")
    emit("")
    emit("   void gtc_tc5uys_n (mp_ptr rp, mp_ptr zp, mp_srcptr q0,")
    emit("                      mp_srcptr q4, mp_srcptr q8, mp_size_t n,")
    emit("                      mp_size_t n8m, mp_ptr up, mp_srcptr vp)")
    emit("")
    emit("   {rp, n} and {up, n} as gtc_tc5uy_n's (n8m & 1, & 2), and in the")
    emit("   same pass {zp, n} = {up, n} + v, or - v with n8m & 4, minus z")
    emit("   where n8m's bits 0 and 1 are equal, else plus z, q8 of n8 = n8m")
    emit("   >> 3 limbs sign-extended.")
    emit("")
    emit("   void gtc_tc5quad_n (mp_ptr ap, mp_ptr bp, mp_ptr cp, mp_ptr dp,")
    emit("                       mp_size_t n, unsigned mode)")
    emit("")
    emit("   In place, from P = a + b, M = a - b, Q = c + d, N = c - d modulo")
    emit("   B^n: a = P, or M with mode & 1, and G the other; c = Q, or N")
    emit("   with mode & 2, and D the other; b = (G + D) >> 2 and d = (G - D)")
    emit("   >> 2, swapped with mode & 4, arithmetic and exact.  n >= 1.")
    emit("")
    emit("   Each kernel's comment in gen_linear.py says how.  */")
    emit("")
    emit("#if defined (__x86_64__) && defined (__ELF__) && defined (__BMI2__) "
         "&& defined (__ADX__) && !defined (GTC_NO_ASM)")
    # The section aligned to 64 bytes, so that the kernels keep their place
    # in the 64-byte fetch windows whatever the size of the code before them
    # (gtc.o): with the 32 of their own .p2align, a change of gtc.c moved
    # them by 32 mod 64 at random, 0.1-0.2% of whole products.  The
    # Gaussian Toom-5's kernels start on a 64-byte boundary on the Zen.
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__)")
    emit("#define GTC_TC5_FA 6")
    emit("#else")
    emit("#define GTC_TC5_FA 5")
    emit("#endif")
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
    # the division by 15, and on the Zen its -q by two chains
    div15_lsh()
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__)")
    div15_lsh(True)
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
    # -(x + 4 y)/5 by shld, for the Intel cores
    emit("#if !(defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__))")
    div5n_rlsh()
    emit("#endif")
    aorsmul1()
    gap()
    # the Gaussian Toom-5's quarter butterflies (the forms of the halvings
    # above, shifted by 2 bits)
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__)")
    rsh1_flagless("gtc_rsh2add_n", False, 2)
    rsh1_flagless("gtc_rsh2sub_n", True, 2)
    emit("#else")
    rsh1("gtc_rsh2add_n", False, 2)
    rsh1("gtc_rsh2sub_n", True, 2)
    emit("#endif")
    gap()
    t5r()
    gap()
    aorslsh()
    aors2c()
    # the Gaussian Toom-5's kernels: the shifts flag-free on the Zen; the
    # Intel cores' gtc_tc5s_n with (c - t)/5 along two chains, their q
    # pass gtc_tc5q8_n (below)
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__)")
    FLAGFREE_SHR[0] = True
    tc5q()
    bfly()
    tc5s()
    tc5q4()
    FLAGFREE_SHR[0] = False
    emit("#else")
    gap()
    bfly()
    tc5s(two_chain=True)
    tc5q4()
    emit("#endif")
    bflyq()
    tc5a()
    tc5uy()
    emit("#if defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__)")
    FLAGFREE_SHR[0] = True
    tc5quad()
    FLAGFREE_SHR[0] = False
    emit("#else")
    tc5quad()
    emit("#endif")
    tc5uys()
    emit("#if !(defined (__znver1__) || defined (__znver2__) "
         "|| defined (__znver3__) || defined (__znver4__) "
         "|| defined (__znver5__))")
    tc5q8()
    emit("#endif")
    emit('\t.section\t.note.GNU-stack,"",@progbits')
    emit("#endif")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
