/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   The Gaussian Toom-Cook-5 (gtc_55): points 0, 1, -1, i, -i, 1+i, 1-i,
   -1+i and infinity, the evaluation of paper II, Section 5.3, and its
   interpolation, Section 5.2.  Nine Gaussian products of n/5 limbs.  Then
   gtc_q1, the dispatch of the Gaussian products.  */

/* One operand's evaluations, k+1 limbs per component in sign-magnitude
   form: at 1 in the first quadrant (no signs), at -1, i, -i, 1+i, 1-i
   and -1+i with a sign per component.  */
typedef struct
{
  mp_ptr p1r, p1i, m1r, m1i, pir, pii, mir, mii;
  mp_ptr ppr, ppi, pmr, pmi, mpr, mpi;	/* at 1+i, 1-i, -1+i */
  int m1r_neg, m1i_neg, pir_neg, pii_neg, mir_neg, mii_neg;
  int ppr_neg, ppi_neg, pmr_neg, pmi_neg, mpr_neg, mpi_neg;
} geval5;

/* GTC_NEG_VEC: tc_abs's negation by vector intrinsics (AVX2, NEON) after
   the low zero limbs; else GMP's mpn_neg.  Per microarchitecture.  */
#if defined (__AVX2__) && !defined (GTC_ZEN)
#define GTC_NEG_VEC 1
#else
#define GTC_NEG_VEC 0
#endif
#if GTC_NEG_VEC && defined (__AVX2__)
#include <immintrin.h>
#elif GTC_NEG_VEC && defined (__aarch64__)
#include <arm_neon.h>
#endif

/* {d, n} = -{d, n} in place, n >= 1, {d, n} nonzero.  */
GTC_INLINE void
gtc_neg_inplace (mp_ptr d, mp_size_t n)
{
#if GTC_NEG_VEC
  mp_size_t i = 0;
  while (d[i] == 0)
    i++;
  d[i] = -d[i];
  i++;
#if defined (__AVX2__)
  {
    __m256i ones = _mm256_set1_epi64x (-1);
    for (; i + 4 <= n; i += 4)
      {
	__m256i v = _mm256_loadu_si256 ((const __m256i *) (d + i));
	_mm256_storeu_si256 ((__m256i *) (d + i), _mm256_xor_si256 (v, ones));
      }
  }
#else
  for (; i + 2 <= n; i += 2)
    vst1q_u64 ((uint64_t *) (d + i), vreinterpretq_u64_u32
	       (vmvnq_u32 (vreinterpretq_u32_u64
			   (vld1q_u64 ((const uint64_t *) (d + i))))));
#endif
  for (; i < n; i++)
    d[i] = ~d[i];
#else
  mpn_neg (d, d, n);
#endif
}

/* {d, n} = |{d, n}|, a two's complement value; returns 1 if it was
   negative.  */
GTC_INLINE int
tc_abs (mp_ptr d, mp_size_t n)
{
  int neg = (mp_limb_signed_t) d[n - 1] < 0;
  if (neg)
    gtc_neg_inplace (d, n);
  return neg;
}

/* One limb of eval5_t5r: d = u - v - b and dm = fy - d - c, b and c the
   chains' borrows, 0 or 1.  On x86-64 the intrinsics store straight into
   d and dm, the second chain reading d back.  */
#ifdef __x86_64__
typedef unsigned long long gtc_ull_alias __attribute__ ((may_alias));
#define EVAL5_T5R_STEP(d, dm, u, v, fy, b, c)				\
  do {									\
    (b) = _subborrow_u64 ((b), (u), (v), (gtc_ull_alias *) &(d));	\
    (c) = _subborrow_u64 ((c), (fy), (d), (gtc_ull_alias *) &(dm));	\
  } while (0)
#else
#define EVAL5_T5R_STEP(d, dm, u, v, fy, b, c)				\
  do {									\
    mp_limb_t d0_, d1_;							\
    unsigned o1_, o2_;							\
    o1_ = __builtin_sub_overflow ((u), (v), &d0_);			\
    o2_ = __builtin_sub_overflow (d0_, (mp_limb_t) (b), &d0_);		\
    (b) = o1_ | o2_;							\
    o1_ = __builtin_sub_overflow ((fy), d0_, &d1_);			\
    o2_ = __builtin_sub_overflow (d1_, (mp_limb_t) (c), &d1_);		\
    (c) = o1_ | o2_;							\
    (d) = d0_;								\
    (dm) = d1_;								\
  } while (0)
#endif

/* The loops of eval5_t5r, d = 4 x4 - x0 (neg) or x0 - 4 x4, from limb
   i: to h with x4, then without (fz the last bits shifted out of x4).  */
#define EVAL5_T5R_LOOPS(neg)						\
  do {									\
    for (; i < h; i++)							\
      {									\
	mp_limb_t z = x4[i], yy = y[i], fz = z << 2 | zp >> SH;		\
	EVAL5_T5R_STEP (d[i], dm[i], (neg) ? fz : x0[i],		\
			(neg) ? x0[i] : fz, yy << 2 | yp >> SH, b, c);	\
	zp = z;								\
	yp = yy;							\
      }									\
    for (; i < k; i++)							\
      {									\
	mp_limb_t yy = y[i], fz = zp >> SH;				\
	EVAL5_T5R_STEP (d[i], dm[i], (neg) ? fz : x0[i],		\
			(neg) ? x0[i] : fz, yy << 2 | yp >> SH, b, c);	\
	zp = 0;								\
	yp = yy;							\
      }									\
  } while (0)

#ifdef GTC_ACC_ASM
/* eval5_t5r's limbs from the second on in blocks of four by gtc_t5r_n,
   from h >= GTC_TC5_T5ASM_MIN.  */
#define GTC_TC5_T5ASM_MIN 40
mp_limb_t gtc_t5r_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
		     mp_limb_t);
#endif

/* {d, k+1} = -t5 = 4 x4 - x0 (neg) or t5 = x0 - 4 x4, and {dm, k+1} =
   4 y - d, in two's complement, in one pass along two borrow chains; x0,
   y of k limbs, x4 of h.  */
GTC_INLINE void
eval5_t5r (mp_ptr d, mp_ptr dm, mp_srcptr x0, mp_srcptr x4, mp_srcptr y,
	   mp_size_t k, mp_size_t h, int neg)
{
#ifdef __x86_64__
  unsigned char b = 0, c = 0;
#else
  unsigned b = 0, c = 0;
#endif
  mp_limb_t zp = 0, yp = 0, top;
  mp_size_t i = 0;
  enum { SH = GMP_NUMB_BITS - 2 };

#ifdef GTC_ACC_ASM
  if (h >= GTC_TC5_T5ASM_MIN)
    {
      /* the first limb here (the kernel reads the limb before), then
	 blocks of four of those up to h by the kernel */
      mp_size_t m = (h - 1) & -(mp_size_t) 4;
      mp_limb_t bc, z = x4[0], yy = y[0], fz = z << 2;
      EVAL5_T5R_STEP (d[0], dm[0], neg ? fz : x0[0], neg ? x0[0] : fz,
		      yy << 2, b, c);
      bc = gtc_t5r_n (d + 1, dm + 1, x0 + 1, x4 + 1, y + 1,
		      (mp_limb_t) m << 3 | (mp_limb_t) neg << 2
		      | (mp_limb_t) c << 1 | b);
      b = bc & 1;
      c = bc >> 1;
      i = m + 1;
      zp = x4[i - 1];
      yp = y[i - 1];
    }
#endif
  if (neg)
    EVAL5_T5R_LOOPS (1);
  else
    EVAL5_T5R_LOOPS (0);
  /* the borrow and the bits shifted out of 4 x4, then of 4 y */
  top = neg ? (zp >> SH) - b : -(mp_limb_t) b - (zp >> SH);
  d[k] = top;
  dm[k] = (yp >> SH) - top - c;
}

/* {d, w} = |X - Y|, X and Y two's complement numbers of w >= 3 limbs;
   returns 1 if X < Y.  The comparison takes the top two limbs at once (a
   signed top limb) and loops only where both pairs are equal; the
   operands are chosen by masks.  */
GTC_INLINE int
tc_abs_sub (mp_ptr d, mp_srcptr x, mp_srcptr y, mp_size_t w)
{
  mp_limb_signed_t x1 = (mp_limb_signed_t) x[w - 1];
  mp_limb_signed_t y1 = (mp_limb_signed_t) y[w - 1];
  mp_limb_t x0 = x[w - 2], y0 = y[w - 2];
  int lt = (x1 < y1) | ((x1 == y1) & (x0 < y0));

  if (((mp_limb_t) (x1 ^ y1) | (x0 ^ y0)) == 0)
    {
      mp_size_t i = w - 3;
      while (i > 0 && x[i] == y[i])
	i--;
      lt = x[i] < y[i];
    }
  GTC_SUB_N (d, gtc_sel (lt, y, x), gtc_sel (lt, x, y), w);
  return lt;
}

/* GTC_TC5_E2: E = a0 + a2 + a4 and |D0| = |a0 - a2 + a4| in one
   gtc_add3_n pass each (eval3); else D = a0 + a4, E = D + a2 and |D -
   a2|, three passes.  GTC_TC5_PPIORI: Im P(1+i)'s sum oriented by its
   sign from the top limbs; else always led by t5i, then tc_abs.  Per
   microarchitecture.  */
#if defined (GTC_ACC_ASM) && !defined (__aarch64__)
#define GTC_TC5_E2 1
#else
#define GTC_TC5_E2 0
#endif
#if defined (__aarch64__) || (defined (__x86_64__) && !defined (__AVX512F__) \
			       && !defined (GTC_ZEN))
#define GTC_TC5_PPIORI 1
#else
#define GTC_TC5_PPIORI 0
#endif

/* Evaluation of a Gaussian polynomial a0 + a1 x + ... + a4 x^4 (all parts
   in the first quadrant; a0 to a3 of k limbs, a4 of h limbs) at 1, -1, i,
   -i, 1+i, 1-i and -1+i.  With E = a0 + a2 + a4, O = a1 + a3, D0 = a0 - a2
   + a4 and D1 = a1 - a3, P(+-1) = E +- O and P(+-i) = D0 +- i D1 are
   butterflies per component, as in gauss_eval4; the other three points
   from those values by paper II's (5.15), made sign-magnitude after.
   T is scratch of 7k + 7 limbs, T2 of 2k + 2.  */
GTC_OUTLINE void
gauss_eval5 (geval5 *e, mp_srcptr ar, mp_srcptr ai, mp_size_t k,
	     mp_size_t h, mp_ptr T, mp_ptr T2)
{
  mp_size_t k1 = k + 1;
  mp_srcptr a0r = ar, a1r = ar + k, a2r = ar + 2 * k, a3r = ar + 3 * k;
  mp_srcptr a4r = ar + 4 * k;
  mp_srcptr a0i = ai, a1i = ai + k, a2i = ai + 2 * k, a3i = ai + 3 * k;
  mp_srcptr a4i = ai + 4 * k;
  mp_ptr E = T, O = T + k1, D0r = T + 2 * k1, D0i = D0r + k1;
  mp_ptr D1r = D0i + k1, D1i = D1r + k1;
  int s0r, s0i, s1r, s1i, eq;
  mp_limb_t f;

  /* at 1 and -1, per component; D0 = |a0 + a4 - a2| on the way */
#if GTC_TC5_E2
  E[k] = eval3_signed (E, a0r, 1, a2r, 1, a4r, 1, k, h);
  s0r = eval3 (D0r, a0r, 1, a2r, -1, a4r, 1, k, h);
#else
  D0r[k] = gtc_add (D0r, a0r, k, a4r, h);
  gtc_add (E, D0r, k1, a2r, k);
  s0r = abs_sub (D0r, D0r, k1, a2r, k);
#endif
  O[k] = GTC_ADD_N (O, a1r, a3r, k);
  e->m1r_neg = bfly (e->p1r, e->m1r, E, O, k1) >> 1;
#if GTC_TC5_E2
  E[k] = eval3_signed (E, a0i, 1, a2i, 1, a4i, 1, k, h);
  s0i = eval3 (D0i, a0i, 1, a2i, -1, a4i, 1, k, h);
#else
  D0i[k] = gtc_add (D0i, a0i, k, a4i, h);
  gtc_add (E, D0i, k1, a2i, k);
  s0i = abs_sub (D0i, D0i, k1, a2i, k);
#endif
  O[k] = GTC_ADD_N (O, a1i, a3i, k);
  e->m1i_neg = bfly (e->p1i, e->m1i, E, O, k1) >> 1;

  /* at i and -i */
  s1r = abs_sub (D1r, a1r, k, a3r, k);
  s1i = abs_sub (D1i, a1i, k, a3i, k);
  D1r[k] = D1i[k] = 0;
  /* Re P(i) = D0r - D1i, Re P(-i) = D0r + D1i: with equal signs the
     magnitudes' sum goes to -i, else to i */
  {
    eq = s0r == s1i;
    mp_ptr dS = (mp_ptr) gtc_sel (eq, e->mir, e->pir);
    mp_ptr dD = (mp_ptr) gtc_sel (eq, e->pir, e->mir);
    f = bfly (dS, dD, D0r, D1i, k1);
    e->pir_neg = s0r ^ (eq & (int) (f >> 1));
    e->mir_neg = s0r ^ ((!eq) & (int) (f >> 1));
  }
  /* Im P(i) = D0i + D1r, Im P(-i) = D0i - D1r */
  {
    eq = s0i == s1r;
    mp_ptr dS = (mp_ptr) gtc_sel (eq, e->pii, e->mii);
    mp_ptr dD = (mp_ptr) gtc_sel (eq, e->mii, e->pii);
    f = bfly (dS, dD, D0i, D1r, k1);
    e->pii_neg = s0i ^ ((!eq) & (int) (f >> 1));
    e->mii_neg = s0i ^ (eq & (int) (f >> 1));
  }

  /* at 1+i, 1-i and -1+i by paper II, (5.15), from P(1), P(i) and P(-i): with t5 = a0 - 4 a4,
     t8 = (-1+i) a3 = -alpha + i beta (alpha = r3 + s3, beta = r3 - s3),
     t9 = P(1) - P(-i) - t8 and t7 = i (P(1) - P(i)),
       P(1+i) = t7 + t8 + t5,  P(1-i) = t5 - i t9,
       P(-1+i) = i (t9 - 4 a2) + t5 = (t5 - 4 i a2) + i t9,
     at k+1 limbs in two's complement, made sign-magnitude after */
  {
    mp_ptr T5r = T, T5i = T5r + k1, T5mr = T5i + k1, T5mi = T5mr + k1;
    mp_ptr T9r = T5mi + k1, T9i = T9r + k1, Al = T2, Be = T2 + k1;
    sreg p1r = { e->p1r, 0 }, p1i = { e->p1i, 0 };
    sreg pir = { e->pir, e->pir_neg }, pii = { e->pii, e->pii_neg };
    sreg mir = { e->mir, e->mir_neg }, mii = { e->mii, e->mii_neg };
    sreg t5r = { T5r, 0 }, t5i = { T5i, 0 }, t5mr = { T5mr, 0 };
    sreg t5mi = { T5mi, 0 }, al = { Al, 0 }, be, t9r, t9i, x;

    f = bfly (Al, Be, a3r, a3i, k);
    Al[k] = f & 1;
    Be[k] = 0;
    be = (sreg) { Be, (int) (f >> 1) };
    /* T5r = -t5r, T5mr = t5r + 4 a2i; T5i = t5i, T5mi = 4 a2r - t5i */
    eval5_t5r (T5r, T5mr, a0r, a4r, a2i, k, h, 1);
    eval5_t5r (T5i, T5mi, a0i, a4i, a2r, k, h, 0);
    t5r.neg = 1;
    t5mi.neg = 1;
    t9r = reg_add3 (T9r, p1r, mir, -1, al, 1, k1);
    t9i = reg_add3 (T9i, p1i, mii, -1, be, -1, k1);
    /* P(1+i): Re t5r - alpha - (Re P(1) - Re P(i)) by the components of
       t7, Im t5i + beta + (Re P(1) - Re P(i)) */
    x = reg_add3 (e->ppr, t5r, p1i, -1, pii, 1, k1);
    x = reg_add (e->ppr, x, al, -1, k1);
    e->ppr_neg = x.neg ^ tc_abs (e->ppr, k1);
#if GTC_TC5_PPIORI
    /* Im P(1+i) = t5i + p1r - pir + beta, often negative where h = k:
       its sign from the top two limbs of each (the rest adds
       less than 3 B^(k-1)), and where negative the sum led by a term of
       negative value, pir or beta, so that the pass stores its
       magnitude */
    {
#define TOP2(x) ((((__int128) (mp_limb_signed_t) (x).p[k]		\
		   * ((__int128) 1 << GMP_NUMB_BITS)) + (x).p[k - 1])	\
		 * ((x).neg ? -1 : 1))
      if (h == k && (!pir.neg || be.neg)
	  && TOP2 (t5i) + TOP2 (p1r) - TOP2 (pir) + TOP2 (be) <= -3)
	{
	  sreg lead = pir.neg ? be : (sreg) { pir.p, 1 };
	  sreg last = pir.neg ? pir : be;
	  x = reg_add3 (e->ppi, lead, t5i, 1, p1r, 1, k1);
	  x = reg_add (e->ppi, x, last, pir.neg ? -1 : 1, k1);
	}
      else
	{
	  x = reg_add3 (e->ppi, t5i, p1r, 1, pir, -1, k1);
	  x = reg_add (e->ppi, x, be, 1, k1);
	}
#undef TOP2
    }
#else
    x = reg_add3 (e->ppi, t5i, p1r, 1, pir, -1, k1);
    x = reg_add (e->ppi, x, be, 1, k1);
#endif
    e->ppi_neg = x.neg ^ tc_abs (e->ppi, k1);
    /* t9's flags are clear (P(1)'s): P(1-i) = -(T5r - Im t9) + i (T5i -
       Re t9), P(-1+i) = (T5mr - Im t9) + i (Re t9 - T5mi) */
    (void) t5mr;
    (void) t5mi;
    (void) t9r;
    (void) t9i;
    e->pmr_neg = 1 ^ tc_abs_sub (e->pmr, T5r, T9i, k1);
    e->pmi_neg = tc_abs_sub (e->pmi, T5i, T9r, k1);
    e->mpr_neg = tc_abs_sub (e->mpr, T5mr, T9i, k1);
    e->mpi_neg = tc_abs_sub (e->mpi, T9r, T5mi, k1);
  }
}

/* Where one level keeps its evaluations and products: eight regions of
   two product buffers each (2k+3 limbs a buffer); the first seven hold
   the evaluations (a and b, both components, k+1 limbs each) at 1, -1,
   i, -i, 1+i, 1-i and -1+i, and the eighth takes the product at 1.  Each
   later product goes over the evaluations of the one before it, which are
   consumed, so the seventh region is free at the end: the interpolation's
   spare buffers.  The result area holds the evaluation's scratch and,
   last, the products at 0 and infinity.  */
/* GTC_TC5_ALIGN: buffers of 2k + 4 limbs and the evaluations at even
   offsets in them, so that every one starts on 16 bytes; else 2k + 3 and
   k + 1.  Per microarchitecture.  */
#if defined (__x86_64__) && !defined (__AVX512F__) && !defined (GTC_ZEN)
#define GTC_TC5_ALIGN 1
#else
#define GTC_TC5_ALIGN 0
#endif
#define GTC_55_PW0(k) (2 * (k) + 3 + GTC_TC5_ALIGN)
#define GTC_55_ITCH(k) (16 * (GTC_55_PW0 (k) + 31 * GTC_TC4_PAD))

GTC_INLINE void
gtc_55_layout (mp_ptr ws, mp_size_t k, mp_size_t pw, geval5 *ea,
		  geval5 *eb)
{
  mp_size_t k1 = k + 1 + (GTC_TC5_ALIGN & (k + 1)), rw = 2 * pw;
  mp_ptr G = ws;

#define GTC_TC5_REGION(x, y)						\
  ea->x = G; ea->y = G + k1; eb->x = G + 2 * k1; eb->y = G + 3 * k1;	\
  G += rw
  GTC_TC5_REGION (p1r, p1i);
  GTC_TC5_REGION (m1r, m1i);
  GTC_TC5_REGION (pir, pii);
  GTC_TC5_REGION (mir, mii);
  GTC_TC5_REGION (ppr, ppi);
  GTC_TC5_REGION (pmr, pmi);
  GTC_TC5_REGION (mpr, mpi);
#undef GTC_TC5_REGION
}

/* The rows of the interpolation, Q at 1, -1, i, -i, 1+i, 1-i and -1+i
   (paper II's R1 to R7), real and imaginary parts, and two free
   buffers.  */
typedef struct
{
  sreg re[8], im[8];
  mp_ptr spare[2];
  /* per component: the top limbs of q0, q2, q4 and
     q6 where they are in place (over the next one's low limb), and
     whether q6 is */
  mp_limb_t top[2][4];
  int q6in[2];
} gauss_rows5;

/* Pointwise products: each into the region of the point before it, the
   one at 1 into the eighth; then q0 and q8 into the result.  */
GTC_INLINE void
gtc_55_products (gtc_ctx *c, mp_ptr P, mp_ptr Q, mp_srcptr ar,
		    mp_srcptr ai, mp_srcptr br, mp_srcptr bi, mp_size_t k,
		    mp_size_t h, mp_size_t pw, const geval5 *ea,
		    const geval5 *eb, mp_ptr ws, gauss_rows5 *r)
{
  mp_size_t rw = 2 * pw, k1 = k + 1;
  mp_ptr G[8];

  for (int j = 0; j < 8; j++)
    G[j] = ws + j * rw;
  gtc_q1 (c, 5, G[7], G[7] + pw, ea->p1r, ea->p1i, eb->p1r, eb->p1i, k1);
  r->re[1] = (sreg) { G[7], 0 };
  r->im[1] = (sreg) { G[7] + pw, 0 };
  gtc_signed (c, 5, &r->re[2], &r->im[2], G[0], G[0] + pw,
	ea->m1r, ea->m1r_neg, ea->m1i, ea->m1i_neg,
	eb->m1r, eb->m1r_neg, eb->m1i, eb->m1i_neg, k1, 1);
  gtc_signed (c, 5, &r->re[3], &r->im[3], G[1], G[1] + pw,
	ea->pir, ea->pir_neg, ea->pii, ea->pii_neg,
	eb->pir, eb->pir_neg, eb->pii, eb->pii_neg, k1, 1);
  gtc_signed (c, 5, &r->re[4], &r->im[4], G[2], G[2] + pw,
	ea->mir, ea->mir_neg, ea->mii, ea->mii_neg,
	eb->mir, eb->mir_neg, eb->mii, eb->mii_neg, k1, 1);
  gtc_signed (c, 5, &r->re[5], &r->im[5], G[3], G[3] + pw,
	ea->ppr, ea->ppr_neg, ea->ppi, ea->ppi_neg,
	eb->ppr, eb->ppr_neg, eb->ppi, eb->ppi_neg, k1, 0);
  gtc_signed (c, 5, &r->re[6], &r->im[6], G[4], G[4] + pw,
	ea->pmr, ea->pmr_neg, ea->pmi, ea->pmi_neg,
	eb->pmr, eb->pmr_neg, eb->pmi, eb->pmi_neg, k1, 0);
  gtc_signed (c, 5, &r->re[7], &r->im[7], G[5], G[5] + pw,
	ea->mpr, ea->mpr_neg, ea->mpi, ea->mpi_neg,
	eb->mpr, eb->mpr_neg, eb->mpi, eb->mpi_neg, k1, 0);
  r->spare[0] = G[6];		/* the evaluations at -1+i, consumed */
  r->spare[1] = G[6] + pw;
  gtc_q1 (c, 5, P, Q, ar, ai, br, bi, k);
  gtc_q1 (c, 5, P + 8 * k, Q + 8 * k, ar + 4 * k, ai + 4 * k, br + 4 * k,
	   bi + 4 * k, h);
}

/* x / 2^s at width w in place, exact: an arithmetic shift.  */
GTC_INLINE sreg
reg_rsh (sreg x, unsigned s, mp_size_t w)
{
  mp_limb_t top = x.p[w - 1];
  mpn_rshift (x.p, x.p, w, s);
  x.p[w - 1] |= -(top >> (GMP_NUMB_BITS - 1)) << (GMP_NUMB_BITS - s);
  return x;
}

/* d = (x + sy y + sz z)/2, exact, at width w: one pass with the kernels,
   two without.  */
GTC_INLINE sreg
reg_add3_half_any (mp_ptr dp, sreg x, sreg y, int sy, sreg z, int sz,
		   mp_size_t w)
{
#ifdef GTC_ACC_ASM
  return reg_add3_half (dp, x, y, sy, z, sz, w);
#else
  x = reg_add (dp, x, y, sy, w);
  return reg_add_half (dp, x, z, sz, w);
#endif
}

#ifdef GTC_ACC_ASM
/* (x +- y)/4 in one pass.  */
mp_limb_t gtc_rsh2add_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t gtc_rsh2sub_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

/* d = (x + s*y) / 4, which must be exact, at width w.  */
GTC_INLINE sreg
reg_add_quarter (mp_ptr dp, sreg x, sreg y, int s, mp_size_t w)
{
  if ((s > 0) == (x.neg == y.neg))
    gtc_rsh2add_n (dp, x.p, y.p, w);
  else
    gtc_rsh2sub_n (dp, x.p, y.p, w);
  return (sreg) { dp, x.neg };
}

/* The Toom-5 interpolation's kernels.  GTC_TC5Q_N, the q pass:
   gtc_tc5q_n on aarch64 and the Zen, gtc_tc5q8_n on the other x86-64
   cores.  Per microarchitecture.  */
void gtc_tc5quad_n (mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, unsigned);
mp_limb_t gtc_tc5s_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
		      mp_size_t, unsigned);
#if defined (__aarch64__) || defined (GTC_ZEN)
mp_limb_t gtc_tc5q_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
		      mp_size_t, unsigned);
#define GTC_TC5Q_N gtc_tc5q_n
#else
mp_limb_t gtc_tc5q8_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
		       mp_size_t, unsigned);
#define GTC_TC5Q_N gtc_tc5q8_n
#endif
#if defined (__aarch64__)
void gtc_tc5q4c2_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
		    mp_srcptr, mp_size_t, mp_size_t);
#else
void gtc_tc5q4_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_srcptr,
		  mp_size_t, mp_size_t);
#endif
#if defined (GTC_ZEN)
void gtc_tc5a_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t,
		 mp_size_t);
mp_limb_t gtc_aors2c_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t,
			unsigned);
#else
void gtc_tc5uys_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
		   mp_size_t, mp_size_t, mp_ptr, mp_srcptr);
#if !defined (__aarch64__)
void gtc_tc5uy_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_srcptr,
		  mp_size_t, mp_size_t, mp_ptr);
#endif
#endif

/* The interpolation's choices, per microarchitecture:
   GTC_TC5_QUAD: from k = GTC_TC5_QUAD on (1: always), steps 1 to 3 by
   g_quad and tc5_q4f; below, by g_bfly1, g_bfly_quarter and tc5_q4.
   GTC_TC5_BFLYQ: from k = GTC_TC5_BFLYQ on, step 3 by g_bflyq1 and
   tc5_q4; below, by tc5_q4f.
   GTC_TC5_UY: from k = GTC_TC5_UY on, -U and Y by gtc_tc5uy_n and S by a
   pass of its own; below (0: always), all three by gtc_tc5uys_n.
   GTC_TC5_TEST: small thresholds, so that the tests take every path.  */
#if defined (GTC_TC5_TEST) && defined (GTC_ASM_X86)
#define GTC_TC5_QUAD 24
#define GTC_TC5_BFLYQ 40
#define GTC_TC5_UY 32
#elif defined (__aarch64__)
#define GTC_TC5_QUAD 1
#define GTC_TC5_BFLYQ 0
#define GTC_TC5_UY 0
#elif defined (GTC_ZEN)
#define GTC_TC5_QUAD 100
#define GTC_TC5_BFLYQ 0
#define GTC_TC5_UY 0
#elif defined (__AVX512F__)
#define GTC_TC5_QUAD 1
#define GTC_TC5_BFLYQ 900
#define GTC_TC5_UY 40
#else
#define GTC_TC5_QUAD 1
#define GTC_TC5_BFLYQ 0
#define GTC_TC5_UY 0
#endif

/* Rows a, b, c, d (Q(1), Q(-1) of one component, Q(i), Q(-i) of either):
   a <- a + b, c <- c + d, b <- (G + s D)/4 and d <- (G - s D)/4 with G =
   a - b and D = c - d, s = +-1, in place; b's and d's flags a's.  */
GTC_INLINE void
g_quad (sreg *a, sreg *b, sreg *c, sreg *d, int s, mp_size_t w)
{
  /* where a pair's flags differ, its stored sum is the difference; D
     subtracted as s and the flags of a and c say */
  unsigned m0 = a->neg != b->neg, m1 = c->neg != d->neg;
  unsigned m2 = (s < 0) != (a->neg != c->neg);
  gtc_tc5quad_n (a->p, b->p, c->p, d->p, w, m0 | m1 << 1 | m2 << 2);
  b->neg = a->neg;
  d->neg = a->neg;
}

/* q4 = (x + y)/4 - q0 - q8 of component C into C + 4k (x's flag clear,
   q8 of w8 limbs sign-extended) and y <- (x - y)/4 in place: two passes
   on x86-64, one on aarch64.  */
GTC_INLINE void
tc5_q4f (mp_ptr C, mp_size_t k, sreg x, sreg *y, mp_size_t w8)
{
  mp_size_t w = 2 * k + 1;
#if defined (__aarch64__)
  gtc_tc5q4c2_n (C + 4 * k, y->p, x.p, y->p, C, C + 8 * k, w,
		 w8 << 1 | (x.neg != y->neg));
  y->neg = x.neg;
#else
  gtc_tc5q4_n (C + 4 * k, x.p, y->p, C, C + 8 * k, w,
	       w8 << 1 | (x.neg != y->neg));
  *y = reg_add_quarter (y->p, x, *y, -1, w);
#endif
}

#if GTC_TC5_QUAD > 1 || GTC_TC5_BFLYQ
/* q4 = c0 - q0 - q8 of component C into C + 4k: c0 = {c0, w} with its
   flag clear, q0 = {C, w}, q8 = {C + 8k, w8} sign-extended, w = 2k + 1.
   One pass over q8's limbs, a loop over the few above.  */
GTC_INLINE void
tc5_q4 (mp_ptr C, mp_size_t k, mp_srcptr c0, mp_size_t w8)
{
  mp_size_t w = 2 * k + 1;
  /* the true carry of the two subtractions is the kernel's count - 2 */
  mp_limb_t cy = gtc_add3_n (C + 4 * k, c0, C, C + 8 * k, w8, 3) - 2;
  if (w8 < w)
    {
      /* - q8's sign extension adds 1 at w8 where q8 < 0; above q8 (at
	 most 8 limbs) c0 - q0 with the carry signed, without branches */
      cy += C[8 * k + w8 - 1] >> (GMP_NUMB_BITS - 1);
      for (mp_size_t i = w8; i < w; i++)
	{
	  __int128 d = (__int128) c0[i] - C[i] + (mp_limb_signed_t) cy;
	  C[4 * k + i] = (mp_limb_t) d;
	  cy = (mp_limb_t) (d >> GMP_NUMB_BITS);
	}
    }
}
#endif

#if GTC_TC5_QUAD > 1
mp_limb_t gtc_bfly_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

/* Rows x and y, both components: (x, y) <- (x + y, x - y) in place, y's
   flags x's, one pass per component.  Where the flags differ the stored
   sum is x - y, so the destinations swap (without a branch).  */
GTC_INLINE void
g_bfly1 (sreg *xr, sreg *xi, sreg *yr, sreg *yi, mp_size_t w)
{
  sreg *xs[2] = { xr, xi }, *ys[2] = { yr, yi };
  for (int c = 0; c < 2; c++)
    {
      sreg x = *xs[c], y = *ys[c];
      int d = x.neg != y.neg;
      gtc_bfly_n ((mp_ptr) gtc_sel (d, y.p, x.p),
		  (mp_ptr) gtc_sel (d, x.p, y.p), x.p, y.p, w);
      ys[c]->neg = x.neg;
    }
}

/* Rows x and y, both components: (x, y) <- ((x + y)/4, (x - y)/4), the
   stored sum into the spare buffers, which then take x's.  Where the
   flags differ the stored sum is x - y, so the pointers swap (without a
   branch).  */
GTC_INLINE void
g_bfly_quarter (sreg *xr, sreg *xi, sreg *yr, sreg *yi, mp_ptr *sp,
		mp_size_t w)
{
  int dr = xr->neg != yr->neg, di = xi->neg != yi->neg;
  mp_ptr ar = sp[0], ai = sp[1];
  gtc_rsh2add_n (ar, xr->p, yr->p, w);
  gtc_rsh2sub_n (yr->p, xr->p, yr->p, w);
  gtc_rsh2add_n (ai, xi->p, yi->p, w);
  gtc_rsh2sub_n (yi->p, xi->p, yi->p, w);
  sp[0] = xr->p;
  sp[1] = xi->p;
  xr->p = (mp_ptr) gtc_sel (dr, yr->p, ar);
  yr->p = (mp_ptr) gtc_sel (dr, ar, yr->p);
  xi->p = (mp_ptr) gtc_sel (di, yi->p, ai);
  yi->p = (mp_ptr) gtc_sel (di, ai, yi->p);
  yr->neg = xr->neg;
  yi->neg = xi->neg;
}
#endif

#if GTC_TC5_BFLYQ
mp_limb_t gtc_bflyq_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

/* Rows x and y, both components: (x, y) <- ((x + y)/4, (x - y)/4) in
   place, y's flags x's, one pass per component; where the flags differ
   the destinations swap.  */
GTC_INLINE void
g_bflyq1 (sreg *xr, sreg *xi, sreg *yr, sreg *yi, mp_size_t w)
{
  sreg *xs[2] = { xr, xi }, *ys[2] = { yr, yi };
  for (int c = 0; c < 2; c++)
    {
      sreg x = *xs[c], y = *ys[c];
      int d = x.neg != y.neg;
      gtc_bflyq_n ((mp_ptr) gtc_sel (d, y.p, x.p),
		   (mp_ptr) gtc_sel (d, x.p, y.p), x.p, y.p, w);
      ys[c]->neg = x.neg;
    }
}
#endif

/* q6 = (x + c2)/5 and q2 = c2 - q6 of component c in one pass, x = (v +
   U)/4, negated with neg, never stored.  q6 into C + 6k and q2 into C +
   2k; their top limbs and q0's into r->top, the low limbs of q4 and q8
   put back.  */
GTC_INLINE sreg
tc5_q6q2w (mp_ptr C, mp_size_t k, sreg v, sreg U, int neg, sreg c2,
	   gauss_rows5 *r, int c)
{
  mp_size_t w = 2 * k + 1;
  mp_limb_t l4 = C[4 * k], t;
  r->q6in[c] = 1;
  t = C[8 * k];			/* q8's low limb */
  r->top[c][0] = C[2 * k];
  gtc_tc5s_n (C + 6 * k, C + 2 * k, v.p, U.p, c2.p, w,
	      (v.neg != U.neg) | (v.neg != neg ? 4 : 0));
  r->top[c][1] = C[4 * k];
  C[4 * k] = l4;
  r->top[c][3] = C[8 * k];	/* q6's top limb, q8's low one back */
  C[8 * k] = t;
  return (sreg) { C + 6 * k, 0 };
}

/* With x = S + sy Y: q7 = (x/8 + c3)/5 (j = 7) or q5 = -(x/4 - c1)/5 (j
   = 5) of component C (len limbs), q added at C + jk and c - q at C + (j
   - 4)k, both sign-extended, in one pass, q never stored.  q and c - q
   are components of the product's coefficients, below 10 B^(w-1) in
   magnitude: q is never -B^w/2, which the q pass excludes.  tmp: w
   limbs, used where q's region runs past C's end.  */
GTC_INLINE void
tc5_qacc (mp_ptr C, mp_size_t len, mp_size_t k, int j, sreg S, sreg Y,
	  int sy, sreg c, mp_ptr tmp)
{
  mp_size_t w = 2 * k + 1, o1 = j * k, o2 = (j - 4) * k, m1 = len - o1;
  unsigned ysub = (sy > 0) != (S.neg == Y.neg);
  unsigned csub = j == 7 ? S.neg != c.neg : S.neg == c.neg;
  unsigned asub = j == 7 ? S.neg : !S.neg;
  mp_ptr a1 = C + o1;
  mp_limb_t nc;

  if (m1 < w)
    a1 = memcpy (tmp, a1, m1 * sizeof (mp_limb_t));
  nc = GTC_TC5Q_N (a1, C + o2, S.p, Y.p, c.p, w,
		   ysub | csub << 1 | asub << 2 | (j == 7) << 3);
  if (m1 < w)
    memcpy (C + o1, tmp, m1 * sizeof (mp_limb_t));
  else if (m1 > w)
    acc_limb_bf (C + o1 + w, m1 - w,
		 (mp_limb_t) (mp_limb_signed_t) (int32_t) nc);
  acc_limb_bf (C + o2 + w, len - o2 - w,
	       (mp_limb_t) (mp_limb_signed_t) (int32_t) (nc >> 32));
}

#else /* !GTC_ACC_ASM */

/* (x/2^t + s y)/5 at width w into dp (x's buffer or one that overlaps
   neither), exact; x's buffer is changed.  */
GTC_INLINE sreg
reg_rsh_div5 (mp_ptr dp, sreg x, sreg y, int s, unsigned t, mp_size_t w)
{
  x = reg_add (dp, reg_rsh (x, t, w), y, s, w);
  mpn_bdiv_dbm1c (dp, dp, w, GMP_NUMB_MAX / 5, 0);
  return x;
}

/* Rows x and y, both components: (x, y) <- ((x + y)/2, (x - y)/2); the
   first goes into the spare buffers, which then take x's.  One
   component's two passes after the other's, its rows still in L1.  */
GTC_INLINE void
g_bfly_half (sreg *xr, sreg *xi, sreg *yr, sreg *yi, mp_ptr *sp,
	     mp_size_t w)
{
  sreg sr = reg_add_half (sp[0], *xr, *yr, 1, w);
  *yr = reg_add_half (yr->p, *xr, *yr, -1, w);
  sreg si = reg_add_half (sp[1], *xi, *yi, 1, w);
  *yi = reg_add_half (yi->p, *xi, *yi, -1, w);
  sp[0] = xr->p;
  sp[1] = xi->p;
  *xr = sr;
  *xi = si;
}

/* q6 = (x/4 + y)/5 of component c into C + 6k where x's flag is clear
   (q8's low limb saved in r->top), else into x's buffer.  */
GTC_INLINE sreg
tc5_q6 (mp_ptr C, mp_size_t k, sreg x, sreg y, gauss_rows5 *r, int c)
{
  mp_size_t w = 2 * k + 1;
  r->q6in[c] = !x.neg;
  if (x.neg)
    return reg_rsh_div5 (x.p, x, y, 1, 2, w);
  r->top[c][3] = C[8 * k];	/* q8's low limb, for now */
  return reg_rsh_div5 (C + 6 * k, x, y, 1, 2, w);
}

/* q2 = c2 - q6 of component c into the result at 2k: q0's top limb and
   q4's low one, which it covers, saved and put back, and q6's top limb
   (where q6 is in place) traded back for q8's low one.  */
GTC_INLINE void
tc5_q2 (mp_ptr C, mp_size_t k, sreg c2, sreg q6, gauss_rows5 *r, int c)
{
  mp_size_t w = 2 * k + 1;
  mp_limb_t l4 = C[4 * k], t;
  r->top[c][0] = C[2 * k];
  reg_add (C + 2 * k, c2, q6, -1, w);
  r->top[c][1] = C[4 * k];
  C[4 * k] = l4;
  if (r->q6in[c])
    {
      t = C[8 * k];
      C[8 * k] = r->top[c][3];
      r->top[c][3] = t;
    }
}
#endif

/* Interpolation of one product's rows by paper II's Section 5.2, q0 in
   {C, 2k+1} and q8 in {C + 8k, 2h+1} of both components C = P, Q.
   Steps 1 to 3, the inverse transform at the fourth roots of unity (each
   level halving without the kernels):
     H1 = Q(1) + Q(-1),  G1 = Q(1) - Q(-1),
     H2 = Q(i) + Q(-i),  G2 = (Q(i) - Q(-i))/i,
     c0 = (H1 + H2)/4 = q0 + q4 + q8,  c2 = (H1 - H2)/4 = q2 + q6,
     c1 = (G1 + G2)/4 = q1 + q5,  c3 = (G1 - G2)/4 = q3 + q7,
     q4 = c0 - q0 - q8.
   Steps 4 to 8, with u, v, z = Q(1+i), Q(1-i), Q(-1+i), A = 2 (q0 - 4
   q4 + 16 q8), U = z - A, W = v + U, Y = u + U and S = Y + v - z:
     q6 = (W/(4i) + c2)/5,  q2 = c2 - q6,
     q7 = ((S + i Y)/8 + c3)/5,  q5 = -((S - i Y)/4 - c1)/5,
     q1 = c1 - q5,  q3 = c3 - q7.
   q2, q4 and q6 go into the result (q6 without the kernels only where
   its flag allows), their top limbs into r->top.  With the kernels q1,
   q3, q5 and q7 are added into the result too; without, rows 1 to 7 end
   holding c1, c2, c3, q4, q5, q6, q7.  */
GTC_INLINE void
gtc_55_interp3 (mp_ptr P, mp_ptr Q, mp_size_t k, mp_size_t h,
		gauss_rows5 *r)
{
  mp_size_t w = 2 * k + 1, w8 = 2 * h + 1;
  sreg *re = r->re, *im = r->im, t, q6r, q6i;
  mp_ptr *sp = r->spare;
  /* the rows: 1 Q(1) .. 7 Q(-1+i) */
  sreg h1r = re[1], h1i = im[1], g1r = re[2], g1i = im[2];
  sreg h2r = re[3], h2i = im[3], g2r = re[4], g2i = im[4];

  /* steps 1 to 3: H1, G1, H2, G2, then c0, c1, c2, c3, then q4 = c0 -
     q0 - q8 into the result (flags clear: c0's is the product at 1's,
     and each butterfly's sum keeps x's) */
#ifdef GTC_ACC_ASM
#if GTC_TC5_QUAD > 1
  if (k < GTC_TC5_QUAD)
    {
      /* the first level one pass per butterfly, G2 = D2/i: Re G2 = Im
	 D2, Im G2 = -Re D2; the second by the quarter butterflies into
	 the spare buffers; q4 by one pass */
      g_bfly1 (&h1r, &h1i, &g1r, &g1i, w);
      g_bfly1 (&h2r, &h2i, &g2r, &g2i, w);
      t = g2r;
      g2r = g2i;
      g2i = (sreg) { t.p, !t.neg };
      g_bfly_quarter (&h1r, &h1i, &h2r, &h2i, sp, w);
      g_bfly_quarter (&g1r, &g1i, &g2r, &g2i, sp, w);
      tc5_q4 (P, k, h1r.p, w8);
      tc5_q4 (Q, k, h1i.p, w8);
    }
  else
#endif
    {
      /* H1 and H2, and c1 = (G1 + G2)/4, c3 = (G1 - G2)/4 with G2 =
	 D2/i: Re G2 = Im D2, Im G2 = -Re D2; then q4 = (H1 + H2)/4 - q0
	 - q8, c0 never stored, and c2 = (H1 - H2)/4 into H2's place, per
	 component */
      g_quad (&h1r, &g1r, &h2i, &g2i, 1, w);
      g_quad (&h1i, &g1i, &h2r, &g2r, -1, w);
      t = g2r;
      g2r = g2i;		/* c3's real part */
      g2i = t;
#if GTC_TC5_BFLYQ
      if (k >= GTC_TC5_BFLYQ)
	{
	  /* c0 and c2 in place, then q4 by one pass */
	  g_bflyq1 (&h1r, &h1i, &h2r, &h2i, w);
	  tc5_q4 (P, k, h1r.p, w8);
	  tc5_q4 (Q, k, h1i.p, w8);
	}
      else
#endif
	{
	  tc5_q4f (P, k, h1r, &h2r, w8);
	  tc5_q4f (Q, k, h1i, &h2i, w8);
	}
    }
#else
  g_bfly_half (&h1r, &h1i, &g1r, &g1i, sp, w);
  g_bfly_half (&h2r, &h2i, &g2r, &g2i, sp, w);
  t = g2r;
  g2r = g2i;
  g2i = (sreg) { t.p, !t.neg };
  g_bfly_half (&h1r, &h1i, &h2r, &h2i, sp, w);
  g_bfly_half (&g1r, &g1i, &g2r, &g2i, sp, w);
  {
    sreg r0re = { P, 0 }, r0im = { Q, 0 };
    sreg r8re = { P + 8 * k, 0 }, r8im = { Q + 8 * k, 0 };
    sreg xr = reg_add (P + 4 * k, h1r, r0re, -1, w);
    sreg xi = reg_add (Q + 4 * k, h1i, r0im, -1, w);
    reg_add_ext (xr.p, xr, r8re, w8, -1, w);
    reg_add_ext (xi.p, xi, r8im, w8, -1, w);
  }
#endif
  /* H1's buffers (c0's where it was stored) are free now, the spare
     ones fr */
  mp_ptr fr[2] = { sp[0], sp[1] };
  sp[0] = h1r.p;
  sp[1] = h1i.p;
  re[1] = g1r; im[1] = g1i;	/* c1 */
  re[2] = h2r; im[2] = h2i;	/* c2 */
  re[3] = g2r; im[3] = g2i;	/* c3 */
  re[4] = (sreg) { P + 4 * k, 0 };	/* q4 */
  im[4] = (sreg) { Q + 4 * k, 0 };

  /* steps 4 to 6, per component: U = z - A into sp, Y = u + U over u, S
     = Y + v - z; with the kernels W = v + U is left to q6's pass */
  sreg wv[2], yv[2], sv[2];
  {
    sreg *q4c[2] = { &re[4], &im[4] }, *uc[2] = { &re[5], &im[5] };
    sreg *vc[2] = { &re[6], &im[6] }, *zc[2] = { &re[7], &im[7] };
    mp_ptr q0p[2] = { P, Q }, q8p[2] = { P + 8 * k, Q + 8 * k };
    for (int c = 0; c < 2; c++)
      {
	mp_ptr a = fr[c], b = sp[c];
	sreg q4 = *q4c[c], u = *uc[c], v = *vc[c], z = *zc[c], U;
#ifdef GTC_ACC_ASM
#if defined (GTC_ZEN)
	/* A = 2 A0 into a; U = z - A into b and Y = u + U over u in one
	   pass, both with z's flag */
	gtc_tc5a_n (a, q0p[c], q4.p, q8p[c], w, w8);
	gtc_aors2c_n (b, u.p, z.p, a, w, (z.neg ^ 1) | (u.neg != z.neg) << 1);
	U = (sreg) { b, z.neg };
	yv[c] = (sreg) { u.p, z.neg };
#else
#if GTC_TC5_UY
	if (k < GTC_TC5_UY)
#endif
	  {
	    /* -U into b, Y over u, S over z, Y's and S's flags u's */
	    gtc_tc5uys_n (b, z.p, q0p[c], q4.p, q8p[c], w,
			  w8 << 3 | z.neg | u.neg << 1
			  | (v.neg != u.neg) << 2, u.p, v.p);
	    U = (sreg) { b, 1 };
	    yv[c] = (sreg) { u.p, u.neg };
	    sv[c] = (sreg) { z.p, u.neg };
	    wv[c] = U;
	    continue;
	  }
#if GTC_TC5_UY
	/* -U = A - z, z's flag folded in, into b, and Y = u + U over u */
	gtc_tc5uy_n (b, z.p, q0p[c], q4.p, q8p[c], w,
		     w8 << 2 | u.neg << 1 | z.neg, u.p);
	U = (sreg) { b, 1 };
	yv[c] = (sreg) { u.p, u.neg };
#endif
#endif
	/* W left to q6's pass; S into a */
	wv[c] = U;
	sv[c] = reg_add3 (a, v, yv[c], 1, z, -1, w);
#else
	/* A0 = q0 + 4 (4 q8 - q4) (q8 of w8 limbs sign-extended) */
	mp_limb_t rr = (mp_limb_t) GTC_RSBLSH2_N (a, q4.p, q8p[c], w8)
	  - 4 * (q8p[c][w8 - 1] >> (GMP_NUMB_BITS - 1));
	/* above q8 (2 (k - h) <= 8 limbs): a = rr - q4, the carry
	   signed */
	for (mp_size_t i = w8; i < w; i++)
	  {
	    __int128 d = (__int128) (mp_limb_signed_t) rr - q4.p[i];
	    a[i] = (mp_limb_t) d;
	    rr = (mp_limb_t) (d >> GMP_NUMB_BITS);
	  }
	GTC_ADDLSH2_N (a, q0p[c], a, w);
	/* U = z - 2 A0, into b */
	if (z.neg)
	  GTC_ADDLSH1_N (b, z.p, a, w);
	else
	  GTC_SUBLSH1_N (b, z.p, a, w);
	U = (sreg) { b, z.neg };
	yv[c] = reg_add (u.p, u, U, 1, w);
	wv[c] = reg_add (a, v, U, 1, w);
	sv[c] = reg_add3 (v.p, v, yv[c], 1, z, -1, w);
#endif
      }
  }
  /* free: U's buffers (after q6) and z's (f[2], f[3]: without the
     kernels only, where S is never over z) */
  mp_ptr f[4] = { sp[0], sp[1], re[7].p, im[7].p };
  r->top[0][2] = P[6 * k];	/* q4's top limbs */
  r->top[1][2] = Q[6 * k];
  /* q6 = (W/i + 4 c2)/20 (W/i = Im W - i Re W) and q2 = c2 - q6 */
#ifdef GTC_ACC_ASM
  /* W = v + U (U in wv) */
  q6r = tc5_q6q2w (P, k, im[6], wv[1], 0, re[2], r, 0);
  q6i = tc5_q6q2w (Q, k, re[6], wv[0], 1, im[2], r, 1);
  /* q7 = ((S + i Y)/8 + c3)/5 and q5 = -((S - i Y)/4 - c1)/5, each added
     with c3 - q7 or c1 - q5; scratch f[0], free after q6 */
  {
    mp_size_t len = 8 * k + w8;
    tc5_qacc (P, len, k, 7, sv[0], yv[1], -1, re[3], f[0]);
    tc5_qacc (P, len, k, 5, sv[0], yv[1], 1, re[1], f[0]);
    tc5_qacc (Q, len, k, 7, sv[1], yv[0], 1, im[3], f[0]);
    tc5_qacc (Q, len, k, 5, sv[1], yv[0], -1, im[1], f[0]);
  }
#else
  q6r = tc5_q6 (P, k, wv[1], re[2], r, 0);
  q6i = tc5_q6 (Q, k, (sreg) { wv[0].p, !wv[0].neg }, im[2], r, 1);
  tc5_q2 (P, k, re[2], q6r, r, 0);
  tc5_q2 (Q, k, im[2], q6i, r, 1);
  {
    /* as ((S + i Y)/8 + c3)/5 and -((S - i Y)/4 - c1)/5 */
    sreg x7r = reg_add (f[0], sv[0], yv[1], -1, w);
    sreg x7i = reg_add (f[1], sv[1], yv[0], 1, w);
    sreg x5r = reg_add (f[2], sv[0], yv[1], 1, w);
    sreg x5i = reg_add (f[3], sv[1], yv[0], -1, w);
    re[7] = reg_rsh_div5 (x7r.p, x7r, re[3], 1, 3, w);
    im[7] = reg_rsh_div5 (x7i.p, x7i, im[3], 1, 3, w);
    x5r = reg_rsh_div5 (x5r.p, x5r, re[1], -1, 2, w);
    x5i = reg_rsh_div5 (x5i.p, x5i, im[1], -1, 2, w);
    re[5] = (sreg) { x5r.p, !x5r.neg };
    im[5] = (sreg) { x5i.p, !x5i.neg };
  }
#endif
  re[6] = q6r;
  im[6] = q6i;
  sp[0] = sv[0].p;
  sp[1] = sv[1].p;
}

#ifdef GTC_ACC_ASM
/* Recomposition with the kernels: q0 to q8 are in place but for the top
   limbs in r->top, which are added back.  */
GTC_INLINE void
gtc_55_recompose3 (mp_ptr P, mp_ptr Q, mp_size_t n, mp_size_t k,
		   const gauss_rows5 *r)
{
  mp_size_t len = 2 * n + 1;
  mp_ptr Cs[2] = { P, Q };

  for (int c = 0; c < 2; c++)
    for (int j = 0; j < 4; j++)
      acc_limb_bf (Cs[c] + (2 * j + 2) * k, len - (2 * j + 2) * k,
		   r->top[c][j]);
}
#else
/* {C, len} += x + y, both of width w, sign-extended (mod B^len).  */
GTC_INLINE void
acc_pair (mp_ptr C, mp_size_t len, sreg x, sreg y, mp_size_t w)
{
  acc (C, len, x, w);
  acc (C, len, y, w);
}

/* Recomposition without the kernels: q0, q2, q4, q8 in place, and q6
   where it is; the limbs from 6k to 8k cleared where q6 is not; the top
   limbs in r->top added back; then q1 = c1 - q5 and q3 = c3 - q7 as
   their two terms, q5, q7, and q6 where it is not in place.  */
GTC_INLINE void
gtc_55_recompose2 (mp_ptr P, mp_ptr Q, mp_size_t n, mp_size_t k,
		   const gauss_rows5 *r)
{
  mp_size_t len = 2 * n + 1, w = 2 * k + 1;
  mp_ptr Cs[2] = { P, Q };

  for (int c = 0; c < 2; c++)
    {
      mp_ptr C = Cs[c];
      const sreg *q = c ? r->im : r->re;
      if (!r->q6in[c])
	mpn_zero (C + 6 * k, 2 * k);
      for (int j = 0; j < 3 + r->q6in[c]; j++)
	acc_limb (C + (2 * j + 2) * k, len - (2 * j + 2) * k, r->top[c][j]);
      acc_pair (C + k, len - k, q[1], (sreg) { q[5].p, !q[5].neg }, w);
      acc_pair (C + 3 * k, len - 3 * k, q[3], (sreg) { q[7].p, !q[7].neg },
		w);
      acc (C + 5 * k, len - 5 * k, q[5], w);
      acc (C + 7 * k, len - 7 * k, q[7], w);
      if (!r->q6in[c])
	acc (C + 6 * k, len - 6 * k, q[6], w);
    }
}
#endif

/* One level of the Gaussian Toom-Cook-5 for gtc_q1 (n >= 17).  */
GTC_OUTLINE void
gtc_55 (gtc_ctx *c, mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
	   mp_srcptr br, mp_srcptr bi, mp_size_t n)
{
  mp_size_t k = (n + 4) / 5, h = n - 4 * k;	/* parts of k (4) and h */
  mp_size_t pw = gtc_44_pw (c, k) + GTC_TC5_ALIGN;
  tmp_mark mk;
  geval5 ea, eb;
  gauss_rows5 r;
  mp_ptr ws = tmp_alloc (c, &mk, GTC_55_ITCH (k));

  gtc_55_layout (ws, k, pw, &ea, &eb);
  gauss_eval5 (&ea, ar, ai, k, h, P, Q);
  gauss_eval5 (&eb, br, bi, k, h, P, Q);
  gtc_55_products (c, P, Q, ar, ai, br, bi, k, h, pw, &ea, &eb, ws, &r);
  gtc_55_interp3 (P, Q, k, h, &r);
#ifdef GTC_ACC_ASM
  gtc_55_recompose3 (P, Q, n, k, &r);
#else
  gtc_55_recompose2 (P, Q, n, k, &r);
#endif
  tmp_free (c, &mk);
}

/* Gaussian product P + iQ = (ar + i ai)(br + i bi), ar, ai, br, bi >= 0 of
   n limbs each (the closed first quadrant), P and Q of 2n+1 limbs in two's
   complement (P may be negative, Q >= 0), as a product of gtc_ii (the
   cap I = i; gtc: I = 5): one level of gtc_jj, j <= I, or a base case.
   Inline: see r2gtc.  */
GTC_INLINE void
gtc_q1 (gtc_ctx *c, int I, mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
	mp_srcptr br, mp_srcptr bi, mp_size_t n)
{
  if (n == 1)			/* four hardware multiplications */
    {
      gtc_1x1 (P, Q, ar[0], ai[0], br[0], bi[0]);
      return;
    }
  /* The algorithm: the context's table for the cap (gtc_table and the
     thresholds) below GTC_TABLE_N limbs, the thresholds above.  */
  int a = n < GTC_TABLE_N ? c->alg[GTC_CAP (I)][n]
    : gtc_alg_thresh (&c->cfg, I, n);
  if (a == GTC_ALG_33)
    gtc_33 (c, P, Q, ar, ai, br, bi, n);
  else if (a == GTC_ALG_44)
    gtc_44 (c, P, Q, ar, ai, br, bi, n);
  else if (a == GTC_ALG_55)
    gtc_55 (c, P, Q, ar, ai, br, bi, n);
  else if (a == GTC_ALG_22)
    gtc_22 (c, P, Q, ar, ai, br, bi, n);
  else if (a == GTC_ALG_LEAF)
    c->leaf[GTC_CAP (I)][n] (P, Q, ar, ai, br, bi, c->leaf_ns);
  else				/* the base case */
    gtc_textbook (P, Q, ar, ai, br, bi, n);
}
