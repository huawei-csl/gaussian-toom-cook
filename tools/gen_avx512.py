#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Generate gtc_avx512_x86.S: gtc's linear passes by AVX-512VL on 256-bit
vectors, for x86-64 with AVX-512VL and DQ (the Cascade Lake).

  mp_limb_t gtc_add3v_n (rp, up, vp, wp, n, signs)   gtc_add3_n's contract

rp may equal any source exactly; n >= 0.

Only ymm16..ymm31 are used: their EVEX encodings need no vzeroupper and
cause no SSE/AVX transition with the textbook kernels' SSE copies, and
256-bit vectors keep a Skylake-SP / Cascade Lake core at turbo licence 0
(a 512-bit version ran at licence 1: 3.37 -> 2.70 GHz on a turbo node).

Speculative carries.  A block of 8 limbs (two vectors) is summed lane by
lane; each lane's carry out (0..2: x = u + v', s = x + w') comes from its
operands alone, as the MSB of vpternlogq 0xd4 = (a & b) | ((a | b) & ~s),
brought down by vpsrlq 63; valignq moves the carries up a lane, the block's top lane into the
next block, and they are added.  That is exact unless a lane wraps as it
takes its carry in (all-ones, or 0 for a borrow), which vpcmpuq finds; such
a block is redone in scalar code (rare for random limbs, common for sign
extensions, so it must stay cheap: one block).  So nothing is serial
between blocks.

The n mod 8 lowest limbs go first, in scalar code along the flag chains as
the scalar kernels do (CF for v, OF for w; the complements' carry-ins nv,
nw start in the flags), entered part-way into seven unrolled steps; their
carry starts the vector blocks.  No masked stores: a masked store cannot
forward to a later load of its 32 bytes, which the callers do (acc_limb_bf
at rp + n, mpn_add_1 after mpn_add_n).  The blocks go two a loop turn, an
odd count entering at the second.

The Makefile assembles this file with the JCC-erratum padding
(-mbranches-within-32B-boundaries): each block ends in a jnz.

Run:  ./tools/gen_avx512.py > src/gtc_avx512_x86.S  (or: make asm)"""
import sys

out = []
e = out.append
lab_no = [0]


def lab(s):
    lab_no[0] += 1
    return f".Lv{s}{lab_no[0]}"


Y = lambda i: f"%ymm{i}"
X = lambda i: f"%xmm{i}"


def block3(nv, nw, off, cin, cout, fix):
    """8 limbs of u + v' + w' at off + r10; cin: the carries into the
    lanes from below (lane 3 is the one into this block's lane 0), cout:
    this block's; ymm16 / ymm17 the complement masks."""
    e(f"\tvmovdqu64\t{off}(%rsi,%r10), {Y(20)}")
    e(f"\tvmovdqu64\t{off + 32}(%rsi,%r10), {Y(21)}")
    e(f"\tvmovdqu64\t{off}(%rdx,%r10), {Y(22)}")
    e(f"\tvmovdqu64\t{off + 32}(%rdx,%r10), {Y(23)}")
    if nv:
        e(f"\tvpxorq\t{Y(16)}, {Y(22)}, {Y(22)}")
        e(f"\tvpxorq\t{Y(16)}, {Y(23)}, {Y(23)}")
    e(f"\tvpaddq\t{Y(22)}, {Y(20)}, {Y(24)}")
    e(f"\tvpaddq\t{Y(23)}, {Y(21)}, {Y(25)}")
    e(f"\tvpternlogq\t$0xd4, {Y(24)}, {Y(22)}, {Y(20)}")
    e(f"\tvpternlogq\t$0xd4, {Y(25)}, {Y(23)}, {Y(21)}")
    e(f"\tvmovdqu64\t{off}(%rcx,%r10), {Y(22)}")
    e(f"\tvmovdqu64\t{off + 32}(%rcx,%r10), {Y(23)}")
    if nw:
        e(f"\tvpxorq\t{Y(17)}, {Y(22)}, {Y(22)}")
        e(f"\tvpxorq\t{Y(17)}, {Y(23)}, {Y(23)}")
    e(f"\tvpaddq\t{Y(22)}, {Y(24)}, {Y(26)}")
    e(f"\tvpaddq\t{Y(23)}, {Y(25)}, {Y(27)}")
    e(f"\tvpternlogq\t$0xd4, {Y(26)}, {Y(22)}, {Y(24)}")
    e(f"\tvpternlogq\t$0xd4, {Y(27)}, {Y(23)}, {Y(25)}")
    e(f"\tvpsrlq\t$63, {Y(20)}, {Y(20)}")
    e(f"\tvpsrlq\t$63, {Y(21)}, {Y(21)}")
    e(f"\tvpsrlq\t$63, {Y(24)}, {Y(24)}")
    e(f"\tvpsrlq\t$63, {Y(25)}, {Y(25)}")
    e(f"\tvpaddq\t{Y(24)}, {Y(20)}, {Y(20)}")
    e(f"\tvpaddq\t{Y(25)}, {Y(21)}, {cout}")
    e(f"\tvalignq\t$3, {cin}, {Y(20)}, {Y(22)}")
    e(f"\tvalignq\t$3, {Y(20)}, {cout}, {Y(23)}")
    e(f"\tvpaddq\t{Y(22)}, {Y(26)}, {Y(26)}")
    e(f"\tvpaddq\t{Y(23)}, {Y(27)}, {Y(27)}")
    e(f"\tvpcmpuq\t$1, {Y(22)}, {Y(26)}, %k1")
    e(f"\tvpcmpuq\t$1, {Y(23)}, {Y(27)}, %k2")
    e("\tkortestb\t%k1, %k2")
    e(f"\tjnz\t{fix}")
    e(f"\tvmovdqu64\t{Y(26)}, {off}(%rdi,%r10)")
    e(f"\tvmovdqu64\t{Y(27)}, {off + 32}(%rdi,%r10)")


def lane3(v, reg):
    e(f"\tvalignq\t$3, {v}, {v}, {Y(28)}")
    e(f"\tvmovq\t{X(28)}, {reg}")


def redo(kind, off):
    """The 8 limbs at off + r10 in scalar code, the carry in rax (0..2) to
    rax.  kind: (nv, nw).  Saves rbx, rbp."""
    e("\tpush\t%rbx")
    e("\tpush\t%rbp")
    e("\tlea\t(%r10), %r11")
    e("\tlea\t64(%r10), %rbp")
    e("\tpush\t%rbp")
    loop = lab("sr")
    e(f"{loop}:")
    e(f"\tmov\t{off}(%rsi,%r11), %rbx")
    e("\txor\t%ebp, %ebp")
    nv, nw = kind
    e("\tadd\t%rax, %rbx")
    e("\tadc\t$0, %rbp")
    e(f"\tmov\t{off}(%rdx,%r11), %rax")
    if nv:
        e("\tnot\t%rax")
    e("\tadd\t%rax, %rbx")
    e("\tadc\t$0, %rbp")
    e(f"\tmov\t{off}(%rcx,%r11), %rax")
    if nw:
        e("\tnot\t%rax")
    e("\tadd\t%rax, %rbx")
    e("\tadc\t$0, %rbp")
    e(f"\tmov\t%rbx, {off}(%rdi,%r11)")
    e("\tmov\t%rbp, %rax")
    e("\tlea\t8(%r11), %r11")
    e("\tcmp\t(%rsp), %r11")
    e(f"\tjne\t{loop}")
    e("\tpop\t%rbp")
    e("\tpop\t%rbp")
    e("\tpop\t%rbx")


def head(kind):
    """The m = n mod 8 (r9) lowest limbs along the flags, entered part-way
    into seven steps; r11 = 8 (m - 7) biases the offsets.  Flags in: the
    carry-ins.  Leaves the carry in rax; clobbers r10, r11."""
    steps = [lab("h") for _ in range(8)]
    table = lab("ht")
    e(f"\tlea\t{table}(%rip), %rax")
    e("\tmovslq\t(%rax,%r9,4), %r11")
    e("\tlea\t(%rax,%r11), %rax\t/* lea, movslq: the flags stay */")
    e("\tlea\t-56(,%r9,8), %r11")      # 8 (m - 7), lea leaves the flags
    e("\tjmp\t*%rax")
    for s in range(7):
        e(f"{steps[s]}:")
        o = 8 * s
        e(f"\tmov\t{o}(%rsi,%r11), %rax")
        nv, nw = kind
        if nv:
            e(f"\tmov\t{o}(%rdx,%r11), %r10")
            e("\tnot\t%r10")
            e("\tadcx\t%r10, %rax")
        else:
            e(f"\tadcx\t{o}(%rdx,%r11), %rax")
        if nw:
            e(f"\tmov\t{o}(%rcx,%r11), %r10")
            e("\tnot\t%r10")
            e("\tadox\t%r10, %rax")
        else:
            e(f"\tadox\t{o}(%rcx,%r11), %rax")
        e(f"\tmov\t%rax, {o}(%rdi,%r11)")
    e(f"{steps[7]}:")
    e("\tsetc\t%al")
    e("\tseto\t%r11b")
    e("\tmovzbl\t%al, %eax")
    e("\tmovzbl\t%r11b, %r11d")
    e("\tadd\t%r11, %rax")
    # the table: m -> the step that leaves m steps behind it
    e("\t.section\t.rodata")
    e("\t.p2align\t2")
    e(f"{table}:")
    for m in range(8):
        e(f"\t.long\t{steps[7 - m]} - {table}")
    e("\t.text")


def body(kind, blk):
    """After the prologue: r9 = n mod 8, r8 = n - n mod 8 (limbs), flags =
    carry-ins.  The head, then the blocks; returns the carry in rax."""
    A_fix, B_fix, A_back, B_back = lab("fa"), lab("fb"), lab("ba"), lab("bb")
    loop, ent_b, oddb, done = lab("lp"), lab("eb"), lab("ob"), lab("dn")
    head(kind)
    e("\ttest\t%r8, %r8")
    e(f"\tjz\t{done}")
    e(f"\tvpbroadcastq\t%rax, {Y(18)}")
    e(f"\tvmovdqa64\t{Y(18)}, {Y(19)}")
    # r10: byte offset of the blocks, from 8m; r8: where they end
    e("\tlea\t(,%r9,8), %r10")
    e("\tlea\t(%r10,%r8,8), %r8")
    e("\tmov\t%r8, %rax")
    e("\tsub\t%r10, %rax")
    e("\ttest\t$64, %al")
    e(f"\tjnz\t{oddb}")
    e("\t.p2align\t5")
    e(f"{loop}:")
    blk(0, Y(18), Y(19), A_fix)
    e(f"{A_back}:")
    e(f"{ent_b}:")
    blk(64, Y(19), Y(18), B_fix)
    e(f"{B_back}:")
    e("\tsub\t$-128, %r10")
    e("\tcmp\t%r8, %r10")
    e(f"\tjne\t{loop}")
    lane3(Y(18), "%rax")
    e(f"{done}:")
    e("\tret")
    e(f"{oddb}:")
    e("\tsub\t$64, %r10")
    e(f"\tjmp\t{ent_b}")
    e(f"{A_fix}:")
    lane3(Y(18), "%rax")
    redo(kind, 0)
    e(f"\tvpbroadcastq\t%rax, {Y(19)}")
    e(f"\tjmp\t{A_back}")
    e(f"{B_fix}:")
    lane3(Y(19), "%rax")
    redo(kind, 64)
    e(f"\tvpbroadcastq\t%rax, {Y(18)}")
    e(f"\tjmp\t{B_back}")


def func(name):
    e(f"\t.globl\t{name}")
    e(f"\t.type\t{name}, @function")
    e("\t.p2align\t5")
    e(f"{name}:")


def add3v():
    func("gtc_add3v_n")
    e("\tmov\t%r9d, %eax")
    e("\tmov\t%r8, %r9")
    e("\tand\t$7, %r9d\t\t/* n mod 8 */")
    e("\tand\t$-8, %r8\t\t/* limbs in full blocks */")
    v = {(a, b): lab(f"v{a}{b}_") for a in (0, 1) for b in (0, 1)}
    table = lab("vt")
    e("\tand\t$3, %eax")
    e(f"\tlea\t{table}(%rip), %r11")
    e("\tmovslq\t(%r11,%rax,4), %rax")
    e("\tadd\t%r11, %rax")
    e("\tjmp\t*%rax")
    e("\t.section\t.rodata")
    e("\t.p2align\t2")
    e(f"{table}:")
    for s in range(4):
        e(f"\t.long\t{v[(s & 1, s >> 1)]} - {table}")
    e("\t.text")
    for (a, b), l in v.items():
        e(f"{l}:")
        # the masks (all-ones), and the carry-ins in CF (nv) and OF (nw)
        if a or b:
            e(f"\tvpternlogq\t$0xff, {Y(16)}, {Y(16)}, {Y(16)}")
            if b:
                e(f"\tvmovdqa64\t{Y(16)}, {Y(17)}")
        e("\txor\t%eax, %eax\t\t/* CF = OF = 0 */")
        if b:
            e("\tmov\t$0x7fffffffffffffff, %rax")
            e("\tadd\t$1, %rax\t\t/* OF = 1, CF = 0 */")
        if a:
            e("\tstc")
        body((a, b), lambda off, cin, cout, fix, a=a, b=b:
             block3(a, b, off, cin, cout, fix))
    e("\t.size\tgtc_add3v_n, .-gtc_add3v_n")


def main():
    e("/* Copyright (c) 2026 Huawei Technologies Co., Ltd.")
    e("   SPDX-License-Identifier: BSD-3-Clause-Clear  */")
    e("/* Generated by gen_avx512.py; do not edit.")
    e("")
    e("   mp_limb_t gtc_add3v_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp,")
    e("                          mp_srcptr wp, mp_size_t n, unsigned signs)")
    e("   gtc_add3_n (gtc_linear_x86.S) by AVX-512VL: {rp, n} = u + v' + w',")
    e("   v' = ~v if signs & 1, w' = ~w if signs & 2, plus their carry-ins;")
    e("   returns the carry out, 0..2 (+ the carry-ins).")
    e("")
    e("   rp may equal any source.  ymm16..ymm28 and k1, k2 only.  */")
    e("")
    e("#if defined (__x86_64__) && defined (__ELF__) "
      "&& defined (__AVX512VL__) && defined (__AVX512DQ__) "
      "&& defined (__BMI2__) && defined (__ADX__) && !defined (GTC_NO_ASM)")
    # The section aligned to 64 bytes, so that the kernels keep their place
    # in the 64-byte fetch windows whatever the size of the code before them
    # (gtc.o): with the 32 of their own .p2align, a change of gtc.c moved
    # them by 32 mod 64 at random, 0.1-0.2% of whole products.
    e("\t.text")
    e("\t.p2align\t6")
    add3v()
    e('\t.section\t.note.GNU-stack,"",@progbits')
    e("#endif")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
