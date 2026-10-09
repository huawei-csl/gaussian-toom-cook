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

/* GTC_GBFLY_COMP: the Gaussian butterflies (g_bfly, g_bfly_half,
   g_bfly_quarter) by component, sum and difference of the real parts,
   then of the imaginary ones; else both sums, then both differences.
   Per microarchitecture.  */
#if defined (__aarch64__)
#define GTC_GBFLY_COMP 0
#else
#define GTC_GBFLY_COMP 1
#endif

/* Rows x and y, both components: (x, y) <- (x + y, x - y); the sum goes
   into the spare buffers, which then take x's.  */
GTC_INLINE void
g_bfly (sreg *xr, sreg *xi, sreg *yr, sreg *yi, mp_ptr *sp, mp_size_t w)
{
#if GTC_GBFLY_COMP
  /* one component's two passes after the other's, its rows still in L1 */
  sreg sr = reg_add (sp[0], *xr, *yr, 1, w);
  *yr = reg_add (yr->p, *xr, *yr, -1, w);
  sreg si = reg_add (sp[1], *xi, *yi, 1, w);
  *yi = reg_add (yi->p, *xi, *yi, -1, w);
#else
  sreg sr = reg_add (sp[0], *xr, *yr, 1, w);
  sreg si = reg_add (sp[1], *xi, *yi, 1, w);
  *yr = reg_add (yr->p, *xr, *yr, -1, w);
  *yi = reg_add (yi->p, *xi, *yi, -1, w);
#endif
  sp[0] = xr->p;
  sp[1] = xi->p;
  *xr = sr;
  *xi = si;
}

#ifndef GTC_ACC_ASM
/* (x + s 4 y) / (5 2^t) at width w into dp (x's buffer or one that
   overlaps neither), exact.  x's buffer may be changed.  */
GTC_INLINE sreg
reg_div5_lsh (mp_ptr dp, sreg x, sreg y, int s, unsigned t, mp_size_t w)
{
  x = reg_aorsmul (x, y, 4, s, w, w);
  mpn_bdiv_dbm1c (dp, x.p, w, GMP_NUMB_MAX / 5, 0);
  x.p = dp;
  return t ? reg_rsh (x, t, w) : x;
}
#endif

/* reg_aorsmul by GTC_ADDMUL_1 and GTC_SUBMUL_1; on x86-64, c a power of
   2, by gtc_aorslsh_n.  */
GTC_INLINE sreg
reg_aorsmul5 (sreg x, sreg y, mp_limb_t c, int s, mp_size_t yw,
	      mp_size_t w)
{
#ifdef GTC_ASM_X86
  {
    mp_limb_t e = c & -(y.p[yw - 1] >> (GMP_NUMB_BITS - 1)), hi;
    int sub = (s > 0) != (x.neg == y.neg);
    hi = gtc_aorslsh_n (x.p, x.p, y.p, yw, __builtin_ctzl (c), sub);
    if (yw < w)
      acc_limb (x.p + yw, w - yw, sub ? e - hi : hi - e);
    return x;
  }
#else
  mp_limb_t e = c & -(y.p[yw - 1] >> (GMP_NUMB_BITS - 1)), hi;

  if ((s > 0) == (x.neg == y.neg))
    {
      hi = GTC_ADDMUL_1 (x.p, y.p, yw, c);
      if (yw < w)
	acc_limb (x.p + yw, w - yw, hi - e);
    }
  else
    {
      hi = GTC_SUBMUL_1 (x.p, y.p, yw, c);
      if (yw < w)
	acc_limb (x.p + yw, w - yw, e - hi);
    }
  return x;
#endif
}

/* Rows x and y, both components: (x, y) <- ((x + y)/2, (x - y)/2); the
   first goes into the spare buffers, which then take x's.  */
GTC_INLINE void
g_bfly_half (sreg *xr, sreg *xi, sreg *yr, sreg *yi, mp_ptr *sp,
	     mp_size_t w)
{
#if GTC_GBFLY_COMP
  /* one component's two passes after the other's, its rows still in L1 */
  sreg sr = reg_add_half (sp[0], *xr, *yr, 1, w);
  *yr = reg_add_half (yr->p, *xr, *yr, -1, w);
  sreg si = reg_add_half (sp[1], *xi, *yi, 1, w);
  *yi = reg_add_half (yi->p, *xi, *yi, -1, w);
#else
  sreg sr = reg_add_half (sp[0], *xr, *yr, 1, w);
  sreg si = reg_add_half (sp[1], *xi, *yi, 1, w);
  *yr = reg_add_half (yr->p, *xr, *yr, -1, w);
  *yi = reg_add_half (yi->p, *xi, *yi, -1, w);
#endif
  sp[0] = xr->p;
  sp[1] = xi->p;
  *xr = sr;
  *xi = si;
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
/* q5, q6 and q7 as (x/4 + y)/5 (x + 4y = 0 mod 20 makes x = 0 mod 4), x
   divided by 4 or 8 in the pass that makes it, the division by 5 with no
   shift.  */
mp_limb_t gtc_rsh2add_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t gtc_rsh2sub_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t gtc_rsh2add3_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
			  mp_size_t, unsigned);
mp_limb_t gtc_rsh3add3_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
			  mp_size_t, unsigned);
mp_limb_t gtc_div5_add_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t,
			  unsigned);

/* GTC_TC5_AB: below k = GTC_TC5_AB, q7's and q5's shifted sums from two
   shared sums, A = R6r + Vr and B = R6i + Vi, by four 2-input shifted
   passes; else by four 3-input ones.  Per microarchitecture.  */
#if defined (__aarch64__)
#define GTC_TC5_AB 1000
#elif defined (GTC_ZEN)
#define GTC_TC5_AB 300
#elif defined (__AVX512F__)
#define GTC_TC5_AB 100
#else
#define GTC_TC5_AB 250
#endif
mp_limb_t gtc_rsh3add_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t gtc_rsh3sub_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

/* d = (x + s*y) / 8, which must be exact, at width w.  */
GTC_INLINE sreg
reg_add_eighth (mp_ptr dp, sreg x, sreg y, int s, mp_size_t w)
{
  if ((s > 0) == (x.neg == y.neg))
    gtc_rsh3add_n (dp, x.p, y.p, w);
  else
    gtc_rsh3sub_n (dp, x.p, y.p, w);
  return (sreg) { dp, x.neg };
}

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

/* Rows x and y, both components: (x, y) <- ((x + y)/4, (x - y)/4); the
   first goes into the spare buffers, which then take x's.  */
GTC_INLINE void
g_bfly_quarter (sreg *xr, sreg *xi, sreg *yr, sreg *yi, mp_ptr *sp,
		mp_size_t w)
{
#if GTC_GBFLY_COMP
  /* one component's two passes after the other's, its rows still in L1 */
  sreg sr = reg_add_quarter (sp[0], *xr, *yr, 1, w);
  *yr = reg_add_quarter (yr->p, *xr, *yr, -1, w);
  sreg si = reg_add_quarter (sp[1], *xi, *yi, 1, w);
  *yi = reg_add_quarter (yi->p, *xi, *yi, -1, w);
#else
  sreg sr = reg_add_quarter (sp[0], *xr, *yr, 1, w);
  sreg si = reg_add_quarter (sp[1], *xi, *yi, 1, w);
  *yr = reg_add_quarter (yr->p, *xr, *yr, -1, w);
  *yi = reg_add_quarter (yi->p, *xi, *yi, -1, w);
#endif
  sp[0] = xr->p;
  sp[1] = xi->p;
  *xr = sr;
  *xi = si;
}

/* d = (x + sy y + sz z) / 2^sh, sh = 2 or 3, exact, at width w: one pass,
   as reg_add3_half.  */
GTC_INLINE sreg
reg_add3_shr (mp_ptr dp, sreg x, sreg y, int sy, sreg z, int sz,
	      unsigned sh, mp_size_t w)
{
  static const unsigned char perm[4][3] = {
    { 0, 1, 2 }, { 1, 0, 2 }, { 2, 0, 1 }, { 0, 1, 2 } };
  int my = (sy > 0) != (x.neg == y.neg), mz = (sz > 0) != (x.neg == z.neg);
  int v = my + 2 * mz;
  mp_srcptr src[3] = { x.p, y.p, z.p };
  (sh == 2 ? gtc_rsh2add3_n : gtc_rsh3add3_n)
    (dp, src[perm[v][0]], src[perm[v][1]], src[perm[v][2]], w, my | mz);
  return (sreg) { dp, x.neg ^ (my & mz) };
}

/* (x + s y) / 5 at width w into dp (x's buffer, y's or one that
   overlaps neither), exact.  */
GTC_INLINE sreg
reg_div5_add (mp_ptr dp, sreg x, sreg y, int s, mp_size_t w)
{
  gtc_div5_add_n (dp, x.p, y.p, w, (s > 0) != (x.neg == y.neg));
  return (sreg) { dp, x.neg };
}
#endif

/* GTC_TC5_Q4ADD3: q4 = c0 - q0 - q8 in one pass over q8's 2h + 1 limbs
   (gtc_add3_n), from k = GTC_TC5_Q4ADD3 on where h < k; else two passes.
   Per microarchitecture.  */
#if defined (GTC_ACC_ASM) && !defined (__aarch64__)
#define GTC_TC5_Q4ADD3 40
#else
#define GTC_TC5_Q4ADD3 0
#endif
#if GTC_TC5_Q4ADD3
/* q4 = c0 - q0 - q8 of one component C (P or Q) into C + 4k: c0 in
   {c0, w} with its flag clear, q0 = {C, w}, q8 = {C + 8k, w8} sign
   extended, w = 2k + 1.  */
GTC_INLINE void
tc5_q4 (mp_ptr C, mp_size_t k, mp_srcptr c0, mp_size_t w8)
{
  mp_size_t w = 2 * k + 1;
  /* the true carry of the two subtractions is the kernel's count - 2 */
  mp_limb_t cy = gtc_add3_n (C + 4 * k, c0, C, C + 8 * k, w8, 3) - 2;
  if (w8 < w)
    {
      /* - q8's sign extension adds 1 at w8 where q8 < 0 */
      cy += C[8 * k + w8 - 1] >> (GMP_NUMB_BITS - 1);
      GTC_SUB_N (C + 4 * k + w8, c0 + w8, C + w8, w - w8);
      acc_limb (C + 4 * k + w8, w - w8, cy);
    }
}
#endif

/* (x + s 4 y)/20, where x is the sum as the interpolation leaves it:
   divided by 4 already with the kernels.  */
#ifdef GTC_ACC_ASM
#define TC5_DIV20(dp, x, y, s, w) reg_div5_add (dp, x, y, s, w)
#else
#define TC5_DIV20(dp, x, y, s, w) reg_div5_lsh (dp, x, y, s, 2, w)
#endif

/* GTC_TC5_UNHALVED (with the kernels): the inverse transform's first
   level by plain sums and differences, the second by quarterings; else
   both levels halve.  Per microarchitecture.  */
#if defined (GTC_ACC_ASM) && !defined (__aarch64__)
#define GTC_TC5_UNHALVED 1
#else
#define GTC_TC5_UNHALVED 0
#endif

/* q6 = (x + 4 y)/20 of component c (C = P or Q) into the result at 6k,
   its top limb over q8's low one (put back by tc5_q2); without the
   kernels only where x's flag is clear, so that the limbs are q6's own,
   else into x's buffer.  */
GTC_INLINE sreg
tc5_q6 (mp_ptr C, mp_size_t k, sreg x, sreg y, gauss_rows5 *r, int c)
{
  mp_size_t w = 2 * k + 1;
#ifdef GTC_ACC_ASM
  /* y's (c2's) flag is clear: with x's set, q6 = (y - x)/5 of the
     buffers, flag clear too */
  r->q6in[c] = 1;
  r->top[c][3] = C[8 * k];	/* q8's low limb, for now */
  gtc_div5_add_n (C + 6 * k, x.p, y.p, w, x.neg ? 2 : 0);
  return (sreg) { C + 6 * k, 0 };
#else
  r->q6in[c] = !x.neg;
  if (x.neg)
    return TC5_DIV20 (x.p, x, y, 1, w);
  r->top[c][3] = C[8 * k];	/* q8's low limb, for now */
  return TC5_DIV20 (C + 6 * k, x, y, 1, w);
#endif
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

#ifdef GTC_ACC_ASM
/* gtc_div5_add_n with a third chain in its pass: z - q into a second
   destination, or q added into one.  */
mp_limb_t gtc_div5_add_sub_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t,
			      unsigned, mp_ptr);
mp_limb_t gtc_div5_add_acc_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t,
			      unsigned, mp_ptr);
#endif
/* GTC_TC5_Q6Q2: q6 and q2 = c2 - q6 of one component in one pass
   (gtc_div5_add_sub_n); else tc5_q6, then tc5_q2.  Per
   microarchitecture.  */
#if defined (GTC_ACC_ASM) && !defined (GTC_ZEN)
#define GTC_TC5_Q6Q2 1

/* tc5_q6, then tc5_q2, of component c in one pass.  */
GTC_INLINE sreg
tc5_q6q2 (mp_ptr C, mp_size_t k, sreg x, sreg c2, gauss_rows5 *r, int c)
{
  mp_size_t w = 2 * k + 1;
  mp_limb_t l4 = C[4 * k], t;
  r->q6in[c] = 1;
  t = C[8 * k];			/* q8's low limb */
  r->top[c][0] = C[2 * k];
  gtc_div5_add_sub_n (C + 6 * k, x.p, c2.p, w, (x.neg ? 2 : 0) | 4,
		      C + 2 * k);
  r->top[c][1] = C[4 * k];
  C[4 * k] = l4;
  r->top[c][3] = C[8 * k];	/* q6's top limb, q8's low one back */
  C[8 * k] = t;
  return (sreg) { C + 6 * k, 0 };
}
#endif

/* GTC_TC5_A0LSH: U = Q(-1+i) - 2 A0 with A0 = q0 + 4 (4 q8 - q4) by
   GTC_RSBLSH2_N and GTC_ADDLSH2_N in a free buffer, one pass over R7;
   else 8 q4, 2 q0 and 32 q8 by three passes over R7.  Per
   microarchitecture.  */
#if defined (__aarch64__) || defined (__AVX512F__)
#define GTC_TC5_A0LSH 1
#else
#define GTC_TC5_A0LSH 0
#endif

/* Interpolation of one product's rows, paper II's (5.7) to (5.14) in
   fewer passes, with q0 in {C, 2k+1} and q8 in {C + 8k, 2h+1} for both
   components C = P, Q.  First the inverse transform at the fourth roots
   of unity, by halvings:
     H1 = (Q(1) + Q(-1))/2,  G1 = (Q(1) - Q(-1))/2,
     H2 = (Q(i) + Q(-i))/2,  G2 = (Q(i) - Q(-i))/(2i),
     c0 = (H1 + H2)/2 = q0 + q4 + q8,  c2 = (H1 - H2)/2 = q2 + q6,
     c1 = (G1 + G2)/2 = q1 + q5,  c3 = (G1 - G2)/2 = q3 + q7,
   so q4 = c0 - q0 - q8 ((5.7) to (5.9) for R1).  Then the three points z
   with z^4 = -4: with U = Q(-1+i) + 8 q4 - 2 q0 - 32 q8 (R7 of (5.8) to
   (5.10) without its term in q2 + q6, which (5.11) takes out of R5
   again), V = Q(1+i) + U, W = Q(1-i) + U and R6 = Q(1-i) - Q(-1+i),
     q6 = (W/i + 4 c2)/20,
     q7 = (R6 + (1+i) V + 8 c3)/40,  q5 = (4 c1 - R6 - (1-i) V)/20,
   whose components are a sum of three.  With the kernels each is (x/4 +
   y)/5: W/4, (R6 + (1+i) V)/8 and (R6 + (1-i) V)/4 come out of the
   passes that make them, and the division by 5 is one pass with no
   shift (gtc_div5_add_n); without, (x +- 4 y)/20 by reg_div5_lsh.  With
   GTC_TC5_UNHALVED the first level of the transform does not halve and
   the second quarters instead.  q2 = c2 - q6.  q4 and q2 go into the
   result, and q6 (always with the kernels, else where its flag allows;
   tc5_q6), their top limbs aside; q1 = c1 - q5 and q3 = c3 - q7 are left to the recomposition.
   At the end the rows 1 to 7 hold c1, c2, c3, q4, q5, q6, q7.  */
GTC_INLINE void
gtc_55_interp2 (mp_ptr P, mp_ptr Q, mp_size_t k, mp_size_t h,
		   gauss_rows5 *r)
{
  mp_size_t w = 2 * k + 1, w8 = 2 * h + 1;
  sreg r0re = { P, 0 }, r0im = { Q, 0 };
  sreg r8re = { P + 8 * k, 0 }, r8im = { Q + 8 * k, 0 };
  sreg *re = r->re, *im = r->im, t, q6r, q6i, x, y;
  mp_ptr *sp = r->spare;
  /* the rows: 1 Q(1) .. 7 Q(-1+i) */
  sreg h1r = re[1], h1i = im[1], g1r = re[2], g1i = im[2];
  sreg h2r = re[3], h2i = im[3], g2r = re[4], g2i = im[4];

#if GTC_TC5_UNHALVED
  /* 2 H1, 2 G1; 2 H2, 2 G2 */
  g_bfly (&h1r, &h1i, &g1r, &g1i, sp, w);
  g_bfly (&h2r, &h2i, &g2r, &g2i, sp, w);
#else
  /* H1, G1; H2, G2 */
  g_bfly_half (&h1r, &h1i, &g1r, &g1i, sp, w);
  g_bfly_half (&h2r, &h2i, &g2r, &g2i, sp, w);
#endif
  t = g2r;
  g2r = g2i;
  g2i = (sreg) { t.p, !t.neg };
  /* c0 into H1's place and c2 into H2's; c1 into G1's and c3 into G2's */
#if GTC_TC5_UNHALVED
  g_bfly_quarter (&h1r, &h1i, &h2r, &h2i, sp, w);
  g_bfly_quarter (&g1r, &g1i, &g2r, &g2i, sp, w);
#else
  g_bfly_half (&h1r, &h1i, &h2r, &h2i, sp, w);
  g_bfly_half (&g1r, &g1i, &g2r, &g2i, sp, w);
#endif
  /* q4 = c0 - q0 - q8 into the result, where it stays (c0's flag is
     clear, as c2's below) */
#if GTC_TC5_A0LSH
  mp_ptr fr[2] = { sp[0], sp[1] };	/* free from here on */
#endif
  sp[0] = h1r.p;
  sp[1] = h1i.p;
#if GTC_TC5_Q4ADD3
  if (h == k || k >= GTC_TC5_Q4ADD3)
    {
      tc5_q4 (P, k, h1r.p, w8);
      tc5_q4 (Q, k, h1i.p, w8);
      h1r = (sreg) { P + 4 * k, 0 };
      h1i = (sreg) { Q + 4 * k, 0 };
    }
  else
#endif
    {
      h1r = reg_add (P + 4 * k, h1r, r0re, -1, w);
      h1i = reg_add (Q + 4 * k, h1i, r0im, -1, w);
      h1r = reg_add_ext (h1r.p, h1r, r8re, w8, -1, w);
      h1i = reg_add_ext (h1i.p, h1i, r8im, w8, -1, w);
    }
  re[1] = g1r; im[1] = g1i;	/* c1 */
  re[2] = h2r; im[2] = h2i;	/* c2 */
  re[3] = g2r; im[3] = g2i;	/* c3 */
  re[4] = h1r; im[4] = h1i;	/* q4 */

  /* R6 = Q(1-i) - Q(-1+i), into the spare buffers */
  {
    sreg r6r = reg_add (sp[0], re[6], re[7], -1, w);
    sreg r6i = reg_add (sp[1], im[6], im[7], -1, w);
    /* U = Q(-1+i) + 8 q4 - 2 q0 - 32 q8 */
    sreg ur = re[7], ui = im[7];
#if GTC_TC5_A0LSH
    if ((re[4].neg | im[4].neg) == 0)
      {
	/* U = Q(-1+i) - 2 A0, A0 = q0 + 4 (4 q8 - q4), in the free buffers
	   (q0, q4, q8 with clear flags; q8 of w8 limbs sign-extended) */
	sreg *u_[2] = { &ur, &ui };
	mp_srcptr q4p[2] = { re[4].p, im[4].p }, q0p[2] = { P, Q };
	mp_srcptr q8p[2] = { P + 8 * k, Q + 8 * k };
	for (int c = 0; c < 2; c++)
	  {
	    mp_ptr t = fr[c];
	    mp_limb_t rr = (mp_limb_t) GTC_RSBLSH2_N (t, q4p[c], q8p[c], w8)
	      - 4 * (q8p[c][w8 - 1] >> (GMP_NUMB_BITS - 1));
	    if (w8 < w)
	      {
		mpn_neg (t + w8, q4p[c] + w8, w - w8);
		acc_limb (t + w8, w - w8, rr);
	      }
	    GTC_ADDLSH2_N (t, q0p[c], t, w);
	    *u_[c] = reg_aorslsh1 (*u_[c], (sreg) { t, 0 }, -1, w, w);
	  }
      }
    else
#endif
      {
	ur = reg_aorsmul5 (ur, re[4], 8, 1, w, w);
	ui = reg_aorsmul5 (ui, im[4], 8, 1, w, w);
	ur = reg_aorslsh1 (ur, r0re, -1, w, w);
	ui = reg_aorslsh1 (ui, r0im, -1, w, w);
	ur = reg_aorsmul5 (ur, r8re, 32, -1, w8, w);
	ui = reg_aorsmul5 (ui, r8im, 32, -1, w8, w);
      }
    /* V = Q(1+i) + U, W = Q(1-i) + U; U's buffers are free then */
    sreg vr = reg_add (re[5].p, re[5], ur, 1, w);
    sreg vi = reg_add (im[5].p, im[5], ui, 1, w);
#ifdef GTC_ACC_ASM
    sreg wr = reg_add_quarter (re[6].p, re[6], ur, 1, w);	/* W/4 */
    sreg wi = reg_add_quarter (im[6].p, im[6], ui, 1, w);
#else
    sreg wr = reg_add (re[6].p, re[6], ur, 1, w);
    sreg wi = reg_add (im[6].p, im[6], ui, 1, w);
#endif
    sp[0] = ur.p;
    sp[1] = ui.p;
    r->top[0][2] = P[6 * k];	/* q4's top limbs */
    r->top[1][2] = Q[6 * k];
    /* q6 = (W/i + 4 c2)/20: W/i = Im W - i Re W */
#ifdef GTC_TC5_Q6Q2
    /* q6 and q2 = c2 - q6 in one pass per component */
    q6r = tc5_q6q2 (P, k, wi, re[2], r, 0);
    q6i = tc5_q6q2 (Q, k, (sreg) { wr.p, !wr.neg }, im[2], r, 1);
#else
    q6r = tc5_q6 (P, k, wi, re[2], r, 0);
    q6i = tc5_q6 (Q, k, (sreg) { wr.p, !wr.neg }, im[2], r, 1);
    /* q2 = c2 - q6 into the result, over q0's top limb and q4's low one;
       c2's flag is clear, so the limbs are q2's own */
    tc5_q2 (P, k, re[2], q6r, r, 0);
    tc5_q2 (Q, k, im[2], q6i, r, 1);
#endif
    /* q7 = ((R6 + (1+i) V)/2 + 4 c3)/20:
       Re (1+i) V = Vr - Vi, Im (1+i) V = Vr + Vi */
#ifdef GTC_ACC_ASM
    int ab = k < GTC_TC5_AB;
    if (ab)
      {
	/* A = R6r + Vr, B = R6i + Vi in R6's buffers; (R6 + (1+i) V)/8 =
	   (A - Vi)/8 + i (B + Vr)/8, then q7 = (that + c3)/5 */
	r6r = reg_add (r6r.p, r6r, vr, 1, w);
	r6i = reg_add (r6i.p, r6i, vi, 1, w);
	x = reg_add_eighth (sp[0], r6r, vi, -1, w);
	y = reg_add_eighth (sp[1], r6i, vr, 1, w);
      }
    else
      {
	/* (R6 + (1+i) V)/8, then q7 = (that + c3)/5 */
	x = reg_add3_shr (sp[0], r6r, vr, 1, vi, -1, 3, w);
	y = reg_add3_shr (sp[1], r6i, vr, 1, vi, 1, 3, w);
      }
#else
    x = reg_add3_half_any (sp[0], r6r, vr, 1, vi, -1, w);
    y = reg_add3_half_any (sp[1], r6i, vr, 1, vi, 1, w);
#endif
    re[7] = TC5_DIV20 (x.p, x, re[3], 1, w);
    im[7] = TC5_DIV20 (y.p, y, im[3], 1, w);
    /* q5 = -((R6 + (1-i) V) - 4 c1)/20:
       Re (1-i) V = Vr + Vi, Im (1-i) V = Vi - Vr */
#ifdef GTC_ACC_ASM
    if (ab)
      {
	/* (R6 + (1-i) V)/4 = (A + Vi)/4 + i (B - Vr)/4 */
	x = reg_add_quarter (r6r.p, r6r, vi, 1, w);
	y = reg_add_quarter (r6i.p, r6i, vr, -1, w);
      }
    else
      {
	/* (R6 + (1-i) V)/4, then q5 = -(that - c1)/5 */
	x = reg_add3_shr (r6r.p, r6r, vr, 1, vi, 1, 2, w);
	y = reg_add3_shr (r6i.p, r6i, vi, 1, vr, -1, 2, w);
      }
#else
    x = reg_add3 (r6r.p, r6r, vr, 1, vi, 1, w);
    y = reg_add3 (r6i.p, r6i, vi, 1, vr, -1, w);
#endif
    x = TC5_DIV20 (x.p, x, re[1], -1, w);
    y = TC5_DIV20 (y.p, y, im[1], -1, w);
    re[5] = (sreg) { x.p, !x.neg };
    im[5] = (sreg) { y.p, !y.neg };
    re[6] = q6r;
    im[6] = q6i;
    sp[0] = vr.p;
    sp[1] = vi.p;
  }
}

/* {C, len} += x + y, both of width w, sign-extended (mod B^len): one pass
   with the kernels, two without.  */
GTC_INLINE void
acc_pair (mp_ptr C, mp_size_t len, sreg x, sreg y, mp_size_t w)
{
#ifdef GTC_ACC_ASM
  acc3 (C, len, x, y, w);
#else
  acc (C, len, x, w);
  acc (C, len, y, w);
#endif
}

/* Recomposition after gtc_55_interp2: q0, q2, q4, q8 in place, and
   q6 where it is; the limbs from 6k to 8k cleared where q6 is not; the
   top limbs the next one's low limbs took added back; then q1 = c1 - q5
   and q3 = c3 - q7 as their two terms, q5, q7, and q6 where it is not in
   place.  */
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
  gtc_55_interp2 (P, Q, k, h, &r);
  gtc_55_recompose2 (P, Q, n, k, &r);
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
