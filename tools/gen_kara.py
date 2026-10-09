#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Generate gtc_kara_x86.S: one level of the Gaussian Karatsuba, one
function per size, 4..64 limbs, x86-64 with BMI2 and ADX.

  void gtc_gkaraN (mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
                   mp_srcptr br, mp_srcptr bi, const fn *tab)

computes P + iQ = (ar + i ai)(br + i bi) for N-limb components >= 0, P and Q
of 2N+1 limbs in two's complement, as gtc.c's gauss_toom22 does, where the
products of the halves (k = ceil(N/2) and s = floor(N/2) limbs) are
tab[k] and tab[s]: functions of the same form, the textbook kernels of
gtc_textbook_x86.S (which ignore tab) or these functions again:
  x = x0 + x1 X,  X = B^k,
  v0 = x0 y0,  vinf = x1 y1,  vm1 = (x0 - x1)(y0 - y1),
  x y = v0 + (v0 + vinf - vm1) X + vinf X^2.
Everything around the three kernel calls is inline and unrolled for the
size: the four differences |x0 - x1| (compared from the top, the operands
chosen by cmov, one subtraction), the rotation of the factors of vm1 into
the first quadrant, and the recomposition.  The differences read the
operands through their argument registers, not through pointers stored to
the frame and loaded back.  The recomposition takes two passes per
component: T = H0 + Li (v0 = L0 + H0 X + c0 X^2 and vinf = Li + Hi X + ...;
its first ten limbs in registers, any others over Li, stored once and
loaded twice back), then T + L0 and T + Hi along one carry
chain (CF) with -vm1 along the other (OF), where -vm1 is added or
subtracted, by the rotation, without a branch: its limbs are xored with
the sign mask in place first, by SSE, which leaves the flags alone, so
that the pass adds each as a memory operand; both factors of vm1 have
their parts swapped where that makes the subtrahend of either component
vm1's imaginary part, so one xor pass serves both.  The carries and the
top limbs go in last, as signed limbs whose propagation is a rare
branch.  Offsets past 127 bytes would take a 4-byte displacement, and
the large levels are bound by the legacy decoders' 16 bytes a cycle: the
passes move their pointers by lea (which leaves the flags alone) as they
go, so that every operand takes one byte.

Run:  ./tools/gen_kara.py > src/gtc_kara_x86.S  (or: make asm)"""
import sys

NMIN, NMAX = 4, 64


def gen(n):
    k, s = (n + 1) // 2, n // 2
    L = 2 * n + 1                      # result limbs
    name = f"gtc_gkara{n}"
    out, tail = [], []
    e = out.append
    slots, off = {}, 0

    def slot(nm, limbs=1):
        nonlocal off
        slots[nm] = off
        off += 8 * limbs

    # tab first: at 0(%rsp) it is the seventh argument of the calls.
    for nm in ("tab", "P", "Q", "ar", "ai", "br", "bi", "sxr", "sxi", "syr",
               "syi", "p0", "q0", "e", "mq", "cT"):
        slot(nm)
    for nm in ("dxr", "dxi", "dyr", "dyi"):
        slot(nm, k)
    slot("MP", 2 * k + 1)
    slot("MQ", 2 * k + 1)
    frame = off           # the calls go to our own kernels: no alignment

    def S(nm, j=0):
        return f"{slots[nm] + 8 * j}(%rsp)"

    # The ABI entry saves the callee-saved registers, passes tab in rax and
    # calls the level's own entry, name_ns, which saves none: it keeps
    # nothing in registers across its calls, and calls the _ns entries of
    # the kernels below it (tab is then the context's table of those).
    regs = ("rbx", "rbp", "r12", "r13", "r14", "r15")
    e(f"\t.globl\t{name}")
    e(f"\t.type\t{name}, @function")
    e("\t.p2align\t4")
    e(f"{name}:")
    for r in regs:
        e(f"\tpush\t%{r}")
    e("\tmov\t56(%rsp), %rax")         # tab, above the pushes
    e(f"\tcall\t{name}_ns")
    for r in reversed(regs):
        e(f"\tpop\t%{r}")
    e("\tret")
    e(f"\t.size\t{name}, .-{name}")
    e(f"\t.globl\t{name}_ns")
    e(f"\t.type\t{name}_ns, @function")
    e("\t.p2align\t5")
    e(f"{name}_ns:")
    e(f"\tsub\t${frame}, %rsp")
    for nm, r in (("P", "rdi"), ("Q", "rsi"), ("ar", "rdx"), ("ai", "rcx"),
                  ("br", "r8"), ("bi", "r9")):
        e(f"\tmov\t%{r}, {S(nm)}")
    e(f"\tmov\t%rax, {S('tab')}")

    # Evaluation at -1: |x0 - x1| and its sign, x1 of s limbs (a zero top
    # limb when s < k).
    def d8(off, base):
        return f"{off}(%{base})" if off else f"(%{base})"

    class Based:
        """A pointer register moved by lea (no flags) as a pass goes, so
        that its operands' offsets stay within a byte's displacement:
        at(offs) makes the offsets of the next instructions fit (as many
        as can from the lowest; the rest take 4 bytes), op(off) is an
        operand, reset() puts the register back."""
        def __init__(self, reg):
            self.reg, self.b = reg, 0

        def at(self, offs):
            lo, hi = min(offs), max(offs)
            if -128 <= lo - self.b and hi - self.b <= 127:
                return
            b = lo + 128
            e(f"\tlea\t{b - self.b}(%{self.reg}), %{self.reg}")
            self.b = b

        def op(self, off):
            return d8(off - self.b, self.reg)

        def reset(self):
            if self.b:
                e(f"\tlea\t{-self.b}(%{self.reg}), %{self.reg}")
                self.b = 0

    def diff(src, dst, sign):
        done = f".Lc{n}{src}"
        # The operand from its argument register, still live here: no store
        # of the pointer to the frame and reload of it before the compare.
        R = {"ar": "%rdx", "ai": "%rcx", "br": "%r8", "bi": "%r9"}[src]
        # Offsets within a byte's displacement (-128..127), not 4 bytes:
        # the compare from a base near the top limbs, the subtraction's
        # operands and the difference (in the frame, above its 128 bytes of
        # slots) from bases biased by 128.
        cb = 8 * (2 * k - 1) - 120 if 8 * (2 * k - 1) > 127 else 0
        if cb:
            e(f"\tlea\t{cb}({R}), %rsi")
        C = "%rsi" if cb else R
        top = k - 1
        if s < k:
            e(f"\tmov\t{8 * (k - 1) - cb}({C}), %rax")
            e("\ttest\t%rax, %rax")
            e(f"\tjnz\t{done}")
            top = k - 2
        for j in range(top, -1, -1):
            e(f"\tmov\t{8 * j - cb}({C}), %rax")
            e(f"\tcmp\t{8 * (k + j) - cb}({C}), %rax")
            if j > 0:
                e(f"\tjne\t{done}")
        e(f"{done}:")
        # CF: x0 < x1 at the deciding limb (0 if all are equal).
        e("\tsetb\t%al")
        sb = 128 if 8 * (k - 1) > 127 else 0
        e(f"\tlea\t{sb}({R}), %rdi")
        e("\tmov\t%rdi, %r10")
        e(f"\tlea\t{8 * k + sb}({R}), %r11")
        e("\tcmovb\t%r11, %r10")       # minuend
        e("\tcmovb\t%rdi, %r11")       # subtrahend
        e("\tmovzbl\t%al, %eax")
        e(f"\tmov\t%rax, {S(sign)}")
        e(f"\tlea\t{slots[dst] + 128}(%rsp), %rsi")
        for j in range(s):
            e(f"\tmov\t{d8(8 * j - sb, 'r10')}, %rbx")
            e(f"\t{'sub' if j == 0 else 'sbb'}\t{d8(8 * j - sb, 'r11')}, %rbx")
            e(f"\tmov\t%rbx, {d8(8 * j - 128, 'rsi')}")
        if s < k:
            # x1's missing limb is 0, and x0's top limb is 0 when x0 < x1:
            # the top limb of the difference is x0's minus the borrow.
            e(f"\tmov\t{d8(8 * (k - 1) - sb, 'rdi')}, %rbx")
            e("\tsbb\t$0, %rbx")
            e(f"\tmov\t%rbx, {d8(8 * (k - 1) - 128, 'rsi')}")

    diff("ar", "dxr", "sxr")
    diff("ai", "dxi", "sxi")
    diff("br", "dyr", "syr")
    diff("bi", "dyi", "syi")

    # v0 in place, then its top limbs saved; vinf over them.
    e(f"\tmov\t{S('P')}, %rdi")
    e(f"\tmov\t{S('Q')}, %rsi")
    e(f"\tmov\t{S('ar')}, %rdx")
    e(f"\tmov\t{S('ai')}, %rcx")
    e(f"\tmov\t{S('br')}, %r8")
    e(f"\tmov\t{S('bi')}, %r9")
    e(f"\tmov\t{S('tab')}, %rax")
    e(f"\tcall\t*{8 * k}(%rax)")
    for c, top in (("P", "p0"), ("Q", "q0")):
        e(f"\tmov\t{S(c)}, %rax")
        e(f"\tmov\t{8 * 2 * k}(%rax), %rax")
        e(f"\tmov\t%rax, {S(top)}")
    for r, nm in (("rdi", "P"), ("rsi", "Q")):
        e(f"\tmov\t{S(nm)}, %{r}")
        e(f"\tadd\t${8 * 2 * k}, %{r}")
    for r, nm in (("rdx", "ar"), ("rcx", "ai"), ("r8", "br"), ("r9", "bi")):
        e(f"\tmov\t{S(nm)}, %{r}")
        e(f"\tadd\t${8 * k}, %{r}")
    e(f"\tmov\t{S('tab')}, %rax")
    e(f"\tcall\t*{8 * s}(%rax)")

    # vm1: each factor d = (Dr + i Di) with signs is (Ar + i Ai) i^e with
    # Ar, Ai >= 0: e = (sr ^ si) + 2 si, and (Ar, Ai) = (Di, Dr) for odd e.
    # Swapping both factors' parts as well (sw = 1 when e < 2) makes the
    # kernel return (-MP, MQ), so that -vm1 never needs -MP: only MQ is
    # ever complemented (P subtracts it for e = 3, Q for e = 0).
    e(f"\tmov\t{S('sxr')}, %rax")
    e(f"\txor\t{S('sxi')}, %rax")
    e(f"\tmov\t{S('syr')}, %r11")
    e(f"\txor\t{S('syi')}, %r11")
    e(f"\tmov\t{S('sxi')}, %r10")
    e("\tlea\t(%rax,%r10,2), %r10")
    e(f"\tmov\t{S('syi')}, %rcx")
    e("\tlea\t(%r11,%rcx,2), %rcx")
    e("\tadd\t%r10, %rcx")
    e("\tand\t$3, %ecx")
    e(f"\tmov\t%rcx, {S('e')}")
    e("\tshr\t%ecx")
    e("\txor\t$1, %ecx")               # sw = (e < 2)
    e("\txor\t%rcx, %rax")
    e("\txor\t%rcx, %r11")
    e(f"\tlea\t{S('dxr')}, %rdx")
    e(f"\tlea\t{S('dxi')}, %rcx")
    e("\tmov\t%rdx, %r10")
    e("\ttest\t%rax, %rax")
    e("\tcmovnz\t%rcx, %rdx")
    e("\tcmovnz\t%r10, %rcx")
    e(f"\tlea\t{S('dyr')}, %r8")
    e(f"\tlea\t{S('dyi')}, %r9")
    e("\tmov\t%r8, %r10")
    e("\ttest\t%r11, %r11")
    e("\tcmovnz\t%r9, %r8")
    e("\tcmovnz\t%r10, %r9")
    e(f"\tlea\t{S('MP')}, %rdi")
    e(f"\tlea\t{S('MQ')}, %rsi")
    e(f"\tmov\t{S('tab')}, %rax")
    e(f"\tcall\t*{8 * k}(%rax)")
    # MQ complemented in place, once, where it is subtracted (e = 0 or 3),
    # by SSE, which leaves the flags alone; its top limb below.
    e(f"\tmov\t{S('e')}, %rax")
    e("\txor\t%r13d, %r13d")
    e("\tmov\t$-1, %rdx")
    e("\tcmp\t$0, %eax")
    e("\tcmove\t%rdx, %r13")
    e("\tcmp\t$3, %eax")
    e("\tcmove\t%rdx, %r13")
    e(f"\tmov\t%r13, {S('mq')}")
    e("\tmovq\t%r13, %xmm7")
    e("\tpunpcklqdq\t%xmm7, %xmm7")
    e(f"\tlea\t{S('MQ')}, %rsi")
    # On a Zen by scalar xor to memory instead: the SSE loads read what the
    # leaf has just stored 8 bytes at a time, which cannot be forwarded; on
    # the Intel cores the scalar form was slower.
    e("#if defined (__znver1__) || defined (__znver2__) "
      "|| defined (__znver3__) || defined (__znver4__) "
      "|| defined (__znver5__)")
    bz = Based("rsi")
    for j in range(2 * k):
        bz.at([8 * j])
        e(f"\txor\t%r13, {bz.op(8 * j)}")
    e("#else")
    bq = Based("rsi")
    for j in range(0, 2 * k, 2):
        bq.at([8 * j, 8 * j + 8])
        e(f"\tmovdqu\t{bq.op(8 * j)}, %xmm{j // 2 % 6}")
        e(f"\tpxor\t%xmm7, %xmm{j // 2 % 6}")
        e(f"\tmovdqu\t%xmm{j // 2 % 6}, {bq.op(8 * j)}")
    e("#endif")

    # With vm1 = (MP + i MQ) i^e, -vm1 adds to P: -MP, +MQ, +MP, -MQ and
    # to Q: -MQ, -MP, +MQ, +MP for e = 0, 1, 2, 3.
    labels = [0]

    def acc_signed(pos, v):
        """{rdi + pos, L - pos} += v, a signed limb in register v."""
        labels[0] += 1
        prop, back = f".Lp{n}_{labels[0]}", f".Lb{n}_{labels[0]}"
        e(f"\tmov\t{v}, %r8")
        e("\tsar\t$63, %r8")
        e(f"\tadd\t{v}, {8 * pos}(%rdi)")
        e("\tadc\t$0, %r8")             # net carry: -1, 0 or 1
        e(f"\tjnz\t{prop}")
        e(f"{back}:")
        rest = L - pos - 1
        tail.append(f"{prop}:")
        if rest == 0:
            tail.append(f"\tjmp\t{back}")
            return
        up, dn = f".Lu{n}_{labels[0]}", f".Ld{n}_{labels[0]}"
        tail.append(f"\tlea\t{8 * (pos + 1)}(%rdi), %r9")
        tail.append(f"\tmov\t${rest}, %r10")
        tail.append("\ttest\t%r8, %r8")
        tail.append(f"\tjs\t{dn}")
        tail.append(f"{up}:")
        tail.append("\taddq\t$1, (%r9)")
        tail.append(f"\tjnc\t{back}")
        tail.append("\tlea\t8(%r9), %r9")
        tail.append("\tdec\t%r10")
        tail.append(f"\tjnz\t{up}")
        tail.append(f"\tjmp\t{back}")
        tail.append(f"{dn}:")
        tail.append("\tsubq\t$1, (%r9)")
        tail.append(f"\tjnc\t{back}")
        tail.append("\tlea\t8(%r9), %r9")
        tail.append("\tdec\t%r10")
        tail.append(f"\tjnz\t{dn}")
        tail.append(f"\tjmp\t{back}")

    def component(c, top, src_even, src_odd, sub_when):
        """Recompose component c; -vm1 is (MP or MQ, as e is even or odd),
        subtracted for the e in sub_when."""
        e(f"\tmov\t{S(c)}, %rdi")
        TR = ["%rbx", "%rcx", "%rdx", "%r8", "%r9", "%r10", "%r11", "%r12",
              "%r14", "%r15"]
        # T = H0 + Li: its first limbs, up to ten, in registers through
        # both passes; the rest, if any, over Li, stored once and loaded
        # twice back.  Its carry cT to the frame.  The source of -vm1 and
        # its mask first, with r13 the mask.
        nr = min(k, len(TR))
        e(f"\tlea\t{S(src_odd)}, %rsi")
        e(f"\tlea\t{S(src_even)}, %rax")
        e(f"\ttestb\t$1, {S('e')}")
        e("\tcmovz\t%rax, %rsi")
        e("\txor\t%r13d, %r13d")
        e(f"\tcmpl\t${sub_when}, {S('e')}")
        e(f"\tcmove\t{S('mq')}, %r13")
        bd, bs = Based("rdi"), Based("rsi")
        for j in range(k):
            r = TR[j] if j < nr else "%rax"
            bd.at([8 * (k + j), 8 * (2 * k + j)])
            e(f"\tmov\t{bd.op(8 * (k + j))}, {r}")
            e(f"\t{'add' if j == 0 else 'adc'}\t{bd.op(8 * (2 * k + j))}, {r}")
            if j >= nr:
                e(f"\tmov\t%rax, {bd.op(8 * (2 * k + j))}")
        e("\tsetc\t%al")
        e("\tmovzbl\t%al, %eax")
        e(f"\tmov\t%rax, {S('cT')}")
        # CF = 0, OF = carry-in of -vm1.
        e("\txor\t%ebp, %ebp")
        e("\tmov\t%r13, %rax")
        e("\tadox\t%r13, %rax")
        # X^1: T + L0 -+ vm1_lo;  X^2: T + Hi -+ vm1_hi.
        for j in range(2 * k):
            jj = j % k
            offs = [8 * (k + j)]
            if jj >= nr:
                offs.append(8 * (2 * k + jj))
            if j < k:
                offs.append(8 * j)
            elif jj < 2 * s - k:
                offs.append(8 * (3 * k + jj))
            bd.at(offs)
            bs.at([8 * j])
            if jj < nr:
                e(f"\tmov\t{TR[jj]}, %rax")
            else:
                e(f"\tmov\t{bd.op(8 * (2 * k + jj))}, %rax")
            if j < k:
                e(f"\tadcx\t{bd.op(8 * j)}, %rax")
            elif jj < 2 * s - k:
                e(f"\tadcx\t{bd.op(8 * (3 * k + jj))}, %rax")
            else:
                e("\tadcx\t%rbp, %rax")
            e(f"\tadox\t{bs.op(8 * j)}, %rax")
            e(f"\tmov\t%rax, {bd.op(8 * (k + j))}")
        e("\tsetc\t%al")
        e("\tseto\t%cl")
        bd.reset()
        bs.reset()
        e(f"\tmov\t{S('cT')}, %rbx")
        e(f"\tmov\t{8 * (2 * k + 2 * s)}(%rdi), %r12")      # ti
        e("\tmovzbl\t%al, %eax")
        e("\tmovzbl\t%cl, %ecx")

        # At X^2: cT + c0.  At X^3: cT + the chains' carries + c0 + the top
        # limb of -+vm1 (+ ti when s = k).  At B^(k+2s): ti.
        e(f"\tmov\t{8 * 2 * k}(%rsi), %rdx")
        e("\txor\t%r13, %rdx")
        e("\tadd\t%rcx, %rax")
        e("\tadd\t%rdx, %rax")
        e(f"\tmov\t{S(top)}, %rdx")
        e("\tadd\t%rdx, %rax")
        e("\tadd\t%rbx, %rax")
        e("\tadd\t%rbx, %rdx")
        if s == k:
            e("\tadd\t%r12, %rax")
        e("\tmov\t%rax, %r15")
        acc_signed(2 * k, "%rdx")
        acc_signed(3 * k, "%r15")
        if s != k:
            acc_signed(k + 2 * s, "%r12")

    component("P", "p0", "MP", "MQ", 3)
    component("Q", "q0", "MQ", "MP", 0)

    e(f"\tadd\t${frame}, %rsp")
    e("\tret")
    out.extend(tail)
    e(f"\t.size\t{name}_ns, .-{name}_ns")
    return "\n".join(out)


def main():
    print("/* Copyright (c) 2026 Huawei Technologies Co., Ltd.")
    print("   SPDX-License-Identifier: BSD-3-Clause-Clear  */")
    print("/* Generated by gen_kara.py; do not edit.  */")
    print()
    print("#if defined (__x86_64__) && defined (__ELF__) && defined (__BMI2__) "
          "&& defined (__ADX__) && !defined (GTC_NO_ASM)")
    # The section aligned to 64 bytes, so that the kernels keep their place
    # in the 64-byte fetch windows whatever the size of the code before them
    # (gtc.o): with the 32 of their own .p2align, a change of gtc.c moved
    # them by 32 mod 64 at random, 0.1-0.2% of whole products.
    print("\t.text")
    print("\t.p2align\t6")
    for n in range(NMIN, NMAX + 1):
        print()
        print(gen(n))
    print('\t.section\t.note.GNU-stack,"",@progbits')
    print("#endif")


if __name__ == "__main__":
    sys.exit(main())
