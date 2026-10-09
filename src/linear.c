/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   Signed values and the linear passes over them.

   A value that can be negative is kept in two's complement at a fixed width
   w (in limbs) with a lazy negation flag: it is (neg ? -1 : 1) * tc(p, w).
   Multiplying by -1 or by i then never touches memory, and a signed linear
   combination costs one pass: the flags only choose between an addition
   and a subtraction.  The widths leave headroom, so the top bit of a stored
   value is its sign.  */

typedef struct
{
  mp_ptr p;
  int neg;
} sreg;

/* d = a +- b (GTC_ADD_N, GTC_SUB_N), and mpn_add, mpn_sub (gtc_add,
   gtc_sub): the kernels' own passes where there are kernels.  */
#if defined (GTC_ASM_X86) || defined (GTC_ASM_ARM)
mp_limb_t gtc_add_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t gtc_sub_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define GTC_ADD_N gtc_add_n
#define GTC_SUB_N gtc_sub_n

/* mpn_add and mpn_sub, an >= bn >= 0.  */
GTC_INLINE mp_limb_t
gtc_add (mp_ptr rp, mp_srcptr ap, mp_size_t an, mp_srcptr bp, mp_size_t bn)
{
  mp_limb_t c = gtc_add_n (rp, ap, bp, bn);
  return an > bn ? mpn_add_1 (rp + bn, ap + bn, an - bn, c) : c;
}

GTC_INLINE mp_limb_t
gtc_sub (mp_ptr rp, mp_srcptr ap, mp_size_t an, mp_srcptr bp, mp_size_t bn)
{
  mp_limb_t c = gtc_sub_n (rp, ap, bp, bn);
  return an > bn ? mpn_sub_1 (rp + bn, ap + bn, an - bn, c) : c;
}
#else
#define GTC_ADD_N mpn_add_n
#define GTC_SUB_N mpn_sub_n
#define gtc_add mpn_add
#define gtc_sub mpn_sub
#endif

/* d = x + s*y, s = +1 or -1, at width w.  dp may equal x.p or y.p.  */
GTC_INLINE sreg
reg_add (mp_ptr dp, sreg x, sreg y, int s, mp_size_t w)
{
  if ((s > 0) == (x.neg == y.neg))
    GTC_ADD_N (dp, x.p, y.p, w);
  else
    GTC_SUB_N (dp, x.p, y.p, w);
  return (sreg) { dp, x.neg };
}

/* The halvings (u +- v) / 2.  The kernels' halve a signed value: the top
   limb is shifted arithmetically (GTC_RSH1_SIGNED).  */
#if defined (GTC_ASM_X86) || defined (GTC_ASM_ARM)
mp_limb_t gtc_rsh1add_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t gtc_rsh1sub_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define GTC_RSH1ADD_N gtc_rsh1add_n
#define GTC_RSH1SUB_N gtc_rsh1sub_n
#define GTC_RSH1_SIGNED 1
#else
#define GTC_RSH1_SIGNED 0
/* GMP's, which shift the carry into the top bit.  Only GMP's assembly
   defines them, so a GMP built without assembly has neither: they are
   weak, and two passes stand in.  */
extern __typeof (mpn_rsh1add_n) mpn_rsh1add_n __attribute__ ((weak));
extern __typeof (mpn_rsh1sub_n) mpn_rsh1sub_n __attribute__ ((weak));

GTC_INLINE void
gtc_rsh1add_c (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
{
  mp_limb_t c = GTC_ADD_N (rp, up, vp, n);
  mpn_rshift (rp, rp, n, 1);
  rp[n - 1] |= c << (GMP_NUMB_BITS - 1);
}

GTC_INLINE void
gtc_rsh1sub_c (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
{
  mp_limb_t c = GTC_SUB_N (rp, up, vp, n);
  mpn_rshift (rp, rp, n, 1);
  rp[n - 1] |= c << (GMP_NUMB_BITS - 1);
}

/* -DGTC_RSH1_C: the two passes always (for the tests).  */
#ifdef GTC_RSH1_C
#define GTC_RSH1ADD_N gtc_rsh1add_c
#define GTC_RSH1SUB_N gtc_rsh1sub_c
#else
#define GTC_RSH1ADD_N(r, u, v, n)					\
  (mpn_rsh1add_n ? (void) mpn_rsh1add_n (r, u, v, n)			\
		 : gtc_rsh1add_c (r, u, v, n))
#define GTC_RSH1SUB_N(r, u, v, n)					\
  (mpn_rsh1sub_n ? (void) mpn_rsh1sub_n (r, u, v, n)			\
		 : gtc_rsh1sub_c (r, u, v, n))
#endif
#endif

/* d = (x + s*y) / 2, which must be exact, at width w.  */
GTC_INLINE sreg
reg_add_half (mp_ptr dp, sreg x, sreg y, int s, mp_size_t w)
{
  if ((s > 0) == (x.neg == y.neg))
    GTC_RSH1ADD_N (dp, x.p, y.p, w);
  else
    GTC_RSH1SUB_N (dp, x.p, y.p, w);
#if !GTC_RSH1_SIGNED
  /* the top bit must repeat the sign bit, now one below */
  dp[w - 1] = (dp[w - 1] & ~HIGHBIT) | ((dp[w - 1] << 1) & HIGHBIT);
#endif
  return (sreg) { dp, x.neg };
}

#ifdef GTC_ACC_ASM
mp_limb_t gtc_rsh1add3_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
			  mp_size_t, unsigned);

/* d = (x + sy*y + sz*z) / 2 in one pass.  The kernel adds three operands,
   one of them complemented; where y and z are both subtracted it makes
   y + z - x and the flag takes the sign.  The operands are picked by index,
   not by a branch on the signs.  */
GTC_INLINE sreg
reg_add3_half (mp_ptr dp, sreg x, sreg y, int sy, sreg z, int sz,
	       mp_size_t w)
{
  static const unsigned char perm[4][3] = {
    { 0, 1, 2 }, { 1, 0, 2 }, { 2, 0, 1 }, { 0, 1, 2 } };
  int my = (sy > 0) != (x.neg == y.neg), mz = (sz > 0) != (x.neg == z.neg);
  int v = my + 2 * mz;
  mp_srcptr src[3] = { x.p, y.p, z.p };
  gtc_rsh1add3_n (dp, src[perm[v][0]], src[perm[v][1]], src[perm[v][2]], w,
		  my | mz);
  return (sreg) { dp, x.neg ^ (my & mz) };
}
#endif

/* GTC_ADDEXT_BF: reg_add_ext's high limbs by a branch-free loop where
   they are few; else by a branch on the carry against the extension.
   Per microarchitecture.  */
#if defined (__x86_64__) && !defined (GTC_ZEN)
#define GTC_ADDEXT_BF 1
#else
#define GTC_ADDEXT_BF 0
#endif

/* d = x + s*y at width w, where y has width yw <= w (sign-extended).  */
GTC_OUTLINE sreg
reg_add_ext (mp_ptr dp, sreg x, sreg y, mp_size_t yw, int s, mp_size_t w)
{
  mp_limb_t ext, c;

  if (yw == w)
    return reg_add (dp, x, y, s, w);
  ext = (y.p[yw - 1] & HIGHBIT) != 0;
#if GTC_ADDEXT_BF
  if (w - yw <= 8)
    {
      mp_limb_signed_t d;
      mp_size_t j;
      if ((s > 0) == (x.neg == y.neg))
	d = (mp_limb_signed_t) (GTC_ADD_N (dp, x.p, y.p, yw) - ext);
      else
	d = (mp_limb_signed_t) (ext - GTC_SUB_N (dp, x.p, y.p, yw));
      for (j = yw; j < w; j++)
	{
	  __int128 t = (__int128) d + x.p[j];
	  dp[j] = (mp_limb_t) t;
	  d = (mp_limb_signed_t) (t >> GMP_NUMB_BITS);
	}
      return (sreg) { dp, x.neg };
    }
#endif
  if ((s > 0) == (x.neg == y.neg))
    {
      /* the high part changes by c - ext */
      c = GTC_ADD_N (dp, x.p, y.p, yw);
      if (c > ext)
	mpn_add_1 (dp + yw, x.p + yw, w - yw, 1);
      else if (c < ext)
	mpn_sub_1 (dp + yw, x.p + yw, w - yw, 1);
      else if (dp != x.p)
	mpn_copyi (dp + yw, x.p + yw, w - yw);
    }
  else
    {
      /* the high part changes by ext - c */
      c = GTC_SUB_N (dp, x.p, y.p, yw);
      if (ext > c)
	mpn_add_1 (dp + yw, x.p + yw, w - yw, 1);
      else if (ext < c)
	mpn_sub_1 (dp + yw, x.p + yw, w - yw, 1);
      else if (dp != x.p)
	mpn_copyi (dp + yw, x.p + yw, w - yw);
    }
  return (sreg) { dp, x.neg };
}

/* q2 = R2 - R4 at width w into the gap {C + 2k, 2k+1} of the result, where
   R4 sits at C + 4k: the low 2k limbs are disjoint from it, and the top
   limb is done after both its operands have been read.  */
GTC_INLINE sreg
reg_sub_ext_gap (mp_ptr dp, sreg x, sreg y, mp_size_t yw, mp_size_t w,
		 mp_size_t k)
{
  mp_limb_t c;

  if (yw < w)				/* dp + yw <= y.p: no overlap */
    return reg_add_ext (dp, x, y, yw, -1, w);
  if (x.neg == y.neg)
    {
      c = GTC_SUB_N (dp, x.p, y.p, 2 * k);
      dp[2 * k] = x.p[2 * k] - y.p[2 * k] - c;
    }
  else
    {
      c = GTC_ADD_N (dp, x.p, y.p, 2 * k);
      dp[2 * k] = x.p[2 * k] + y.p[2 * k] + c;
    }
  return (sreg) { dp, x.neg };
}

/* {p, len} += q, where q has width qw and is sign-extended (mod B^len).  */
GTC_OUTLINE void
acc (mp_ptr p, mp_size_t len, sreg q, mp_size_t qw)
{
  mp_limb_t ext, c;

  if (qw >= len)
    {
      if (q.neg)
	GTC_SUB_N (p, p, q.p, len);
      else
	GTC_ADD_N (p, p, q.p, len);
      return;
    }
  ext = (q.p[qw - 1] & HIGHBIT) != 0;
  c = q.neg ? GTC_SUB_N (p, p, q.p, qw) : GTC_ADD_N (p, p, q.p, qw);
  /* the rest changes by c - ext when adding, by ext - c when subtracting */
  if (c != ext)
    {
      if ((c > ext) != (q.neg != 0))
	mpn_add_1 (p + qw, p + qw, len - qw, 1);
      else
	mpn_sub_1 (p + qw, p + qw, len - qw, 1);
    }
}

/* acc's limbs above qw (qw < len), after its qw limbs were added (or
   subtracted) elsewhere with carry (or borrow) c.  */
GTC_INLINE void
acc_tail (mp_ptr p, mp_size_t len, sreg q, mp_size_t qw, mp_limb_t c)
{
  mp_limb_t ext = (q.p[qw - 1] & HIGHBIT) != 0;
  if (c != ext)
    {
      if ((c > ext) != (q.neg != 0))
	mpn_add_1 (p + qw, p + qw, len - qw, 1);
      else
	mpn_sub_1 (p + qw, p + qw, len - qw, 1);
    }
}

/* {p, len} += v, v a signed limb (mod B^len).  */
GTC_INLINE void
acc_limb (mp_ptr p, mp_size_t len, mp_limb_t v)
{
  if ((mp_limb_signed_t) v >= 0)
    mpn_add_1 (p, p, len, v);
  else
    mpn_sub_1 (p, p, len, -v);
}

/* The same without a branch on the sign of v: after the first limb the net
   carry is almost always 0, so the rare propagation is well predicted.  */
GTC_INLINE void
acc_limb_bf (mp_ptr p, mp_size_t len, mp_limb_t v)
{
  mp_limb_t s = p[0] + v;
  mp_limb_signed_t d = (mp_limb_signed_t) (s < v)
    - (mp_limb_signed_t) (v >> (GMP_NUMB_BITS - 1));
  p[0] = s;
  if (__builtin_expect (d != 0, 0) && len > 1)
    {
      if (d > 0)
	mpn_add_1 (p + 1, p + 1, len - 1, 1);
      else
	mpn_sub_1 (p + 1, p + 1, len - 1, 1);
    }
}

#ifdef GTC_ACC_ASM
/* The accumulation kernels: gtc_aors2_n adds a signed value into each of
   two destinations in one pass, gtc_add3_n adds two into one (the
   contracts are in the generators).  With AVX-512VL and DQ, gtc_add3_n is
   gtc_avx512_x86.S's.  */
#if defined (GTC_ASM_X86) && defined (__AVX512VL__) && defined (__AVX512DQ__)
#define GTC_X86_AVX512 1
mp_limb_t gtc_add3v_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t,
		       unsigned);
#define gtc_add3_n gtc_add3v_n
#endif
mp_limb_t gtc_aors2_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t,
		       unsigned);
mp_limb_t gtc_add3_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t,
		      unsigned);

/* The sign of an operand as the kernels add it: the top bit of its top
   limb, complemented when it is subtracted.  */
#define ACC_SIGN(q, w) (((q).p[(w) - 1] >> (GMP_NUMB_BITS - 1)) ^ (q).neg)

/* GTC_ACC2_SPLIT: acc2 by two passes of GTC_ADD_N or GTC_SUB_N rather
   than by gtc_aors2_n's two carry chains; per microarchitecture.  */
#if defined (GTC_ASM_X86) && !defined (GTC_ZEN)
#define GTC_ACC2_SPLIT 1
#else
#define GTC_ACC2_SPLIT 0
#endif

/* {P, len} += qp and {Q, len} += qq, both of width qw, sign-extended (mod
   B^len).  A subtracted operand is added as its complement plus 1, so the
   carry out counts B^m once too often; what is left for the limbs above is
   the carry less the sign bit of the operand as added, -1, 0 or 1.  */
GTC_INLINE void
acc2 (mp_ptr P, mp_ptr Q, mp_size_t len, sreg qp, sreg qq, mp_size_t qw)
{
  mp_size_t m = qw < len ? qw : len;
#if GTC_ACC2_SPLIT
  mp_limb_t cp = qp.neg ? 1 - GTC_SUB_N (P, P, qp.p, m)
			: GTC_ADD_N (P, P, qp.p, m);
  mp_limb_t cq = qq.neg ? 1 - GTC_SUB_N (Q, Q, qq.p, m)
			: GTC_ADD_N (Q, Q, qq.p, m);
  mp_limb_t c = cp | cq << 1;
#else
  mp_limb_t c = gtc_aors2_n (P, Q, qp.p, qq.p, m, qp.neg | qq.neg << 1);
#endif

  if (m < len)
    {
      acc_limb_bf (P + m, len - m, (c & 1) - ACC_SIGN (qp, qw));
      acc_limb_bf (Q + m, len - m, (c >> 1) - ACC_SIGN (qq, qw));
    }
}

/* {C, len} += x + y, both of width qw, sign-extended (mod B^len), in one
   pass.  */
GTC_INLINE void
acc3 (mp_ptr C, mp_size_t len, sreg x, sreg y, mp_size_t qw)
{
  mp_size_t m = qw < len ? qw : len;
  mp_limb_t c = gtc_add3_n (C, C, x.p, y.p, m, x.neg | y.neg << 1);

  if (m < len)
    acc_limb_bf (C + m, len - m, c - ACC_SIGN (x, qw) - ACC_SIGN (y, qw));
}
#endif

#ifdef GTC_PAIR_ASM
mp_limb_t gtc_aors2x_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
			mp_srcptr, mp_size_t, unsigned);
#endif

/* Both components of a Gaussian addition: *dre = xre + sre*yre into dre_p
   and *dim = xim + sim*yim into dim_p, at width w.  */
GTC_INLINE void
reg_add2 (sreg *dre, sreg *dim, mp_ptr dre_p, mp_ptr dim_p, sreg xre,
	  sreg yre, int sre, sreg xim, sreg yim, int sim, mp_size_t w)
{
#ifdef GTC_PAIR_ASM
  unsigned sg = ((sre > 0) != (xre.neg == yre.neg))
    | ((sim > 0) != (xim.neg == yim.neg)) << 1;
  gtc_aors2x_n (dre_p, dim_p, xre.p, xim.p, yre.p, yim.p, w, sg);
  *dre = (sreg) { dre_p, xre.neg };
  *dim = (sreg) { dim_p, xim.neg };
#else
  *dre = reg_add (dre_p, xre, yre, sre, w);
  *dim = reg_add (dim_p, xim, yim, sim, w);
#endif
}

/* c ? a : b by masks: where the condition is a sign of the data, a branch
   would be mispredicted half the time.  */
GTC_INLINE mp_srcptr
gtc_sel (int c, mp_srcptr a, mp_srcptr b)
{
  uintptr_t m = -(uintptr_t) c;
  return (mp_srcptr) (((uintptr_t) a & m) | ((uintptr_t) b & ~m));
}

/* {rp, an} = |{ap, an} - {bp, bn}|, an >= bn >= 1.  Returns 1 if a < b.
   rp may equal ap.  */
GTC_INLINE int
abs_sub (mp_ptr rp, mp_srcptr ap, mp_size_t an, mp_srcptr bp, mp_size_t bn)
{
  int neg;

  if (an > bn && !mpn_zero_p (ap + bn, an - bn))
    {
      gtc_sub (rp, ap, an, bp, bn);
      return 0;
    }
  /* the operand order by a flag and masks, not by a branch */
  {
    mp_size_t i = bn - 1;
    while (i > 0 && ap[i] == bp[i])
      i--;
    neg = ap[i] < bp[i];
  }
  {
    mp_srcptr u = gtc_sel (neg, bp, ap), v = gtc_sel (neg, ap, bp);
    GTC_SUB_N (rp, u, v, bn);
  }
  if (an > bn && rp != ap)
    mpn_zero (rp + bn, an - bn);
  return neg;
}

#ifdef GTC_ACC_ASM
/* The evaluations of a Toom-3 level in one pass each: s0 x0 + s1 x1 + s2
   x2, x0 and x1 of k limbs, x2 of h limbs (k-2 <= h <= k), signs +1 or -1,
   by gtc_add3_n, in sign-magnitude form.  The sign comes first, from the
   top limbs: with V = T B^(k-1) + L, T the signed sum of the parts' limbs
   at k-1, L lies strictly between -nneg B^(k-1) and npos B^(k-1).  Only
   where T falls between (about never) is the value negated after.  */

/* {d, k} = e0 x0 + e1 x1 + e2 x2 with one ej = +1 at least; returns the
   signed limb at k.  */
GTC_INLINE mp_limb_t
eval3_signed (mp_ptr d, mp_srcptr x0, int e0, mp_srcptr x1, int e1,
	      mp_srcptr x2, int e2, mp_size_t k, mp_size_t h)
{
  mp_srcptr u, v, w;
  int nv, nw;
  mp_limb_signed_t cy;
  mp_size_t i;

  /* u is a part that is added; x2, of h limbs, is v or w when it can be */
  if (e0 > 0)
    { u = x0; v = x1; nv = e1 < 0; w = x2; nw = e2 < 0; }
  else if (e1 > 0)
    { u = x1; v = x0; nv = 1; w = x2; nw = e2 < 0; }
  else
    { u = x2; v = x0; nv = 1; w = x1; nw = 1; }
  cy = (mp_limb_signed_t) gtc_add3_n (d, u, v, w, h, nv | nw << 1) - nv - nw;
  for (i = h; i < k; i++)
    {
      __int128 t = (__int128) cy;
      t += e0 > 0 ? (__int128) x0[i] : -(__int128) x0[i];
      t += e1 > 0 ? (__int128) x1[i] : -(__int128) x1[i];
      d[i] = (mp_limb_t) t;
      cy = (mp_limb_signed_t) (t >> GMP_NUMB_BITS);
    }
  return (mp_limb_t) cy;
}

/* {d, k+1} = |s0 x0 + s1 x1 + s2 x2|; returns 1 if the value is negative.
   One of the signs is +1.  */
GTC_INLINE int
eval3 (mp_ptr d, mp_srcptr x0, int s0, mp_srcptr x1, int s1, mp_srcptr x2,
       int s2, mp_size_t k, mp_size_t h)
{
  int npos = (s0 > 0) + (s1 > 0) + (s2 > 0), neg;
  __int128 t = (s0 > 0 ? (__int128) x0[k - 1] : -(__int128) x0[k - 1])
    + (s1 > 0 ? (__int128) x1[k - 1] : -(__int128) x1[k - 1]);

  if (h == k)
    t += s2 > 0 ? (__int128) x2[k - 1] : -(__int128) x2[k - 1];
  if (t >= 3 - npos)
    neg = 0;
  else if (t <= -npos)
    neg = 1;
  else
    {
      d[k] = eval3_signed (d, x0, s0, x1, s1, x2, s2, k, h);
      if ((mp_limb_signed_t) d[k] >= 0)
	return 0;
      mpn_neg (d, d, k + 1);
      return 1;
    }
  if (neg)
    d[k] = eval3_signed (d, x0, -s0, x1, -s1, x2, -s2, k, h);
  else
    d[k] = eval3_signed (d, x0, s0, x1, s1, x2, s2, k, h);
  return neg;
}
#endif
