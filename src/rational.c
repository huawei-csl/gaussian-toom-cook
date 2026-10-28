/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   The reductions of rational multiplication to Gaussian multiplication,
   r2gtc_22 to r2gtc_55 (paper I, Algorithm 1.1, and paper II, Sections
   4.4 to 4.6), their dispatch r2gtc, and mul_any for unbalanced
   operands.  */

GTC_OUTLINE void r2gtc_55 (gtc_ctx *, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
GTC_OUTLINE void r2gtc_44 (gtc_ctx *, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
GTC_OUTLINE void r2gtc_33 (gtc_ctx *, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
GTC_OUTLINE void r2gtc_22 (gtc_ctx *, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

/* GMP's rational Karatsuba, mpn_toom22_mul, which recurses by GMP's own
   thresholds and never comes back to gtc.  n >= GTC_GMP22_MIN.  */
GTC_OUTLINE void
gmp_22 (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
	       mp_size_t n)
{
  tmp_mark mk;
  mp_ptr ws = tmp_alloc (c, &mk, mpn_toom22_mul_itch (n, n));
  mpn_toom22_mul (rp, ap, n, bp, n, ws);
  tmp_free (c, &mk);
}

/* Rational n x n product {rp, 2n} = {ap, n} * {bp, n} as a product of
   r2gtc_ii (the cap I = i; r2gtc: I = 5): r2gtc_55 from r2gtc55_thresh
   on, r2gtc_44 from r2gtc44_thresh on, r2gtc_33 from r2gtc33_thresh on,
   r2gtc_22 from r2gtc22_thresh on, skipping those above the cap, GMP's
   Karatsuba from gmp22_thresh on, the textbook product below; never
   GMP's Toom.  r2gtc_table[GTC_CAP (I)] overrides them by size.  Inline,
   like gtc_q1: the reduction's nodes call each other directly.  */
GTC_INLINE void
r2gtc (gtc_ctx *c, int I, mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  int a = r2gtc_alg (c, I, n);

  if (a == R2GTC_ALG_55)
    r2gtc_55 (c, rp, ap, bp, n);
  else if (a == R2GTC_ALG_44)
    r2gtc_44 (c, rp, ap, bp, n);
  else if (a == R2GTC_ALG_33)
    r2gtc_33 (c, rp, ap, bp, n);
  else if (a == R2GTC_ALG_22)
    r2gtc_22 (c, rp, ap, bp, n);
  else if (a == R2GTC_ALG_GMP22)
    gmp_22 (c, rp, ap, bp, n);
  else
    mpn_mul_basecase (rp, ap, n, bp, n);
}

/* Whether a rational product of k + 1 limbs whose top limbs x, y (0 to
   2) are not both 0 is taken as the k-limb product and the corrections
   (a pass of mpn_add_n or mpn_addlsh1_n for each nonzero one): where k +
   1 begins a step of the product's cost.  */
GTC_INLINE int
r2gtc_top_rule (const gtc_ctx *c, int I, mp_size_t k, mp_limb_t x,
		mp_limb_t y)
{
  int a = r2gtc_alg (c, I, k);
  return a < R2GTC_ALG_33 ? (k & 1) == 0
    : a == R2GTC_ALG_33 ? k % 6 == 0
    : a == R2GTC_ALG_44 ? (k & 3) == 0
    : (x == 0 || y == 0) && k % 5 == 0;
}

/* The product of k + 1 limbs {v, 2k+1} = {a, k+1} {b, k+1} whose top
   limbs are 0 to 2, by r2gtc_top_rule.  */
GTC_INLINE void
r2gtc_top (gtc_ctx *c, int I, mp_ptr v, mp_srcptr a, mp_srcptr b,
	   mp_size_t k, int always)
{
  mp_limb_t x = a[k], y = b[k];
  if ((x | y) == 0 || always || r2gtc_top_rule (c, I, k, x, y))
    {
      /* (A + x X)(C + y X) = A C + (y A + x C) X + x y X^2 */
      mp_limb_t t = x * y;
      r2gtc (c, I, v, a, b, k);
      if (y)
	t += y == 1 ? GTC_ADD_N (v + k, v + k, a, k)
	  : GTC_ADDLSH1_N (v + k, v + k, a, k);
      if (x)
	t += x == 1 ? GTC_ADD_N (v + k, v + k, b, k)
	  : GTC_ADDLSH1_N (v + k, v + k, b, k);
      v[2 * k] = t;
    }
  else
    r2gtc (c, I, v, a, b, k + 1);
}

#ifdef GTC_ASM_X86
/* r2gtc_22's recomposition in one pass (gtc_linear_x86.S).  */
void gtc_k2_recomp (mp_ptr, mp_srcptr, mp_size_t, mp_size_t);
#endif

/* One level of the Karatsuba-style reduction to Gaussian multiplication of
   paper I, Algorithm 1.1 (MultZ).  With a = a0 + a1 X and b = b0 + b1 X,
   X = B^k, k = ceil(n/2), the high halves a1, b1 of s = floor(n/2) limbs:
     c0 + i c1 = (a0 + i a1)(b0 + i b1),   c2 = a1 b1 (recursively),
     a b = (c0 + c2) + c1 X + c2 X^2,
   as c0 = a0 b0 - a1 b1 and c1 = a0 b1 + a1 b0.  The Gaussian factors are
   the halves themselves and lie in the closed first quadrant, so there is
   no evaluation and no rotation.  Besides the two products, the paper's
   three additions: c0 + c2 (2k limbs, its carry and c0's sign vanish
   modulo B^(2k), as c0 + c2 = a0 b0 < B^(2k)), and c1 at X with its carry.
   n >= 2.  */
/* GTC_K2_VCOPY: r2gtc_22's copies of the high halves for odd n (s <= 32
   limbs) by 16-byte moves, the last one overlapping; else a scalar loop.
   Per microarchitecture.  */
#if defined (__aarch64__)
#define GTC_K2_VCOPY 0
#else
#define GTC_K2_VCOPY 1
#endif

GTC_OUTLINE void
r2gtc_22 (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  mp_size_t s = n >> 1, k = n - s, w = 2 * k + 1;
  tmp_mark mk;
  mp_ptr Q = tmp_alloc (c, &mk, w + (s < k ? 2 * k : 0));
  mp_srcptr ai = ap + k, bi = bp + k;

  if (s < k)
    {
      /* The Gaussian product takes components of equal length: the high
	 halves get a zero top limb.  */
      mp_ptr t = Q + w;
#if GTC_K2_VCOPY
      if (s <= 32 && s >= 2)
	{
	  /* 16 bytes at a time, the last chunk overlapping */
	  mp_size_t i;
	  for (i = 0; i + 2 < s; i += 2)
	    {
	      __builtin_memcpy (t + i, ap + k + i, 16);
	      __builtin_memcpy (t + k + i, bp + k + i, 16);
	    }
	  __builtin_memcpy (t + s - 2, ap + k + s - 2, 16);
	  __builtin_memcpy (t + k + s - 2, bp + k + s - 2, 16);
	}
      else
#endif
      if (s <= 32)		/* short: inline rather than by two calls */
	{
	  mp_size_t i;
	  for (i = 0; i < s; i++)
	    {
	      t[i] = ap[k + i];
	      t[k + i] = bp[k + i];
	    }
	}
      else
	{
	  mpn_copyi (t, ap + k, s);
	  mpn_copyi (t + k, bp + k, s);
	}
      t[s] = 0;
      t[k + s] = 0;
      ai = t;
      bi = t + k;
    }
  /* c0 in the result area, c1 in Q; then c2 in its place at X^2, over
     c0's top limb, which c0 + c2 modulo B^(2k) does not read.  */
  gtc_q1 (c, 2, rp, Q, ap, ai, bp, bi, k);
  r2gtc (c, 2, rp + 2 * k, ap + k, bp + k, s);
#ifdef GTC_ASM_X86
  /* both additions below in one pass of two carry chains */
  if (n >= 64)
    gtc_k2_recomp (rp, Q, k, s);
  else
#endif
    {
      /* c0 + c2 = a0 b0 modulo B^(2k).  The carry out of the s limbs of
	 c2 is [c0 < 0], as often 1 as 0: it goes into the other limbs, at
	 most two, without a branch on it, and out of them it vanishes.  */
      mp_limb_t cy = GTC_ADD_N (rp, rp, rp + 2 * k, 2 * s);
      if (s < k)
	{
	  unsigned __int128 t = ((unsigned __int128) rp[2 * k - 1] << 64
				 | rp[2 * k - 2]) + cy;
	  rp[2 * k - 2] = (mp_limb_t) t;
	  rp[2 * k - 1] = (mp_limb_t) (t >> 64);
	}
      /* + c1 X; c1 < 2 B^n fits in n + 1 limbs.  */
      gtc_add (rp + k, rp + k, k + 2 * s, Q, n + 1);
    }
  tmp_free (c, &mk);
}

/* With the kernels (GTC_ACC_ASM), r2gtc_33 makes q2 and adds q1 into the
   product in one pass each, by gtc_add3_n, and the real part of the
   product at (i,1) goes into the gap between q0 and q4 in the result,
   where q2 is made in place over it; the scratch then holds the product
   at (-1,1), the imaginary part and one limb: 4k+4 limbs instead of
   6k+4.  gtc_signed picks the buffer of the real part by the rotation,
   which the signs of the evaluations decide, so the buffers are passed in
   the order that puts it in the gap.  The evaluations at -1 lie in the
   gap until their product, those at i below it; q0 goes last into its
   place, q4 over the real part's top limb, which trades places with q4's
   low limb through the one limb.  */

/* One level of the Toom-Cook-3 style reduction to Gaussian multiplication
   of paper II, Section 4.4, with the point (1,1) replaced by (-1,1): points
   (0,1), (-1,1), (1,0) and (i,1), the latter as one Gaussian product whose
   real and imaginary parts give two rational equations.  With X = B^k, the
   value at (-1,1), a0 - a1 + a2, lies in (-X, 2X): its top limb is 0 or 1,
   and 1 for only about 1/6 of random inputs, where a0 + a1 + a2 at (1,1)
   has a top limb of 0, 1 or 2, nonzero for about 5/6.  The price is a sign,
   which the interpolation absorbs by choosing between additions and
   subtractions.  n = 3 or n >= 5.  */
GTC_OUTLINE void
r2gtc_33 (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  mp_size_t k, h, w;
  tmp_mark mk;
  mp_ptr ws, asm1, bsm1, ar, br, v1, R, I;
  mp_srcptr a0, a1, a2, b0, b1, b2;
  mp_limb_t cy, q2t;
  int am1_neg, bm1_neg, ar_neg, br_neg;
  sreg vm1, re, im;
#ifdef GTC_ACC_ASM
  mp_ptr sl;
#endif

  k = (n + 2) / 3;		/* parts of k, k and h limbs */
  h = n - 2 * k;
  w = 2 * k + 1;		/* every coefficient q_j fits in w limbs */
  a0 = ap; a1 = ap + k; a2 = ap + 2 * k;
  b0 = bp; b1 = bp + k; b2 = bp + 2 * k;

  /* The evaluations in the result area, as GMP's Toom-3 keeps them,
     which the products at 0 and infinity, taken last, overwrite: 4k+2 <=
     2n limbs.  */
#ifdef GTC_ACC_ASM
  ws = tmp_alloc (c, &mk, 4 * k + 4);
  ar = rp;
  br = ar + k;
  asm1 = rp + 2 * k;		/* up to 4k+2 <= 2n */
  bsm1 = asm1 + (k + 1);
  v1 = ws;			/* 2k+2 limbs */
  R = rp + 2 * k;		/* w limbs, the real part */
  I = v1 + (2 * k + 2);		/* w limbs */
  sl = I + w;			/* 1 limb */
#else
  ws = tmp_alloc (c, &mk, 6 * k + 4);
  asm1 = rp;
  bsm1 = asm1 + (k + 1);
  ar = bsm1 + (k + 1);
  br = ar + k;
  v1 = ws;			/* 2k+2 limbs */
  R = v1 + (2 * k + 2);		/* w limbs */
  I = R + w;			/* w limbs */
#endif

  /* Evaluation, paper II (4.13) with (-1,1) in place of (1,1):
     P(-1,1) = (a0 + a2) - a1 in sign-magnitude form, as GMP's Toom-3
     evaluates at -1, Re P(i,1) = a0 - a2, Im P(i,1) = a1.  */
#ifdef GTC_ACC_ASM
  am1_neg = eval3 (asm1, a0, 1, a1, -1, a2, 1, k, h);
  bm1_neg = eval3 (bsm1, b0, 1, b1, -1, b2, 1, k, h);
#else
  asm1[k] = gtc_add (asm1, a0, k, a2, h);
  am1_neg = abs_sub (asm1, asm1, k + 1, a1, k);
  bsm1[k] = gtc_add (bsm1, b0, k, b2, h);
  bm1_neg = abs_sub (bsm1, bsm1, k + 1, b1, k);
#endif
  ar_neg = abs_sub (ar, a0, k, a2, h);
  br_neg = abs_sub (br, b0, k, b2, h);

  /* Products: three rational, one Gaussian.  q0 = Q(0,1) and q4 = Q(1,0)
     are written to their final places, last, over the evaluations.
     asm1 and bsm1 are below 2 B^k, so their product is below 4 B^2k and
     fits in the w = 2k+1 limbs the interpolation reads, whatever the top
     limbs are.  So take the k-limb product of the low parts always, and
     add the tails the top limbs stand for, as GMP's SMALLER_RECURSION
     does for the same point:
       (A + a B^k)(C + c B^k) = A C + (a C + c A) B^k + a c B^2k,
     a and c in {0,1}.  */
    {
      mp_limb_t t = 0;
      r2gtc (c, 3, v1, asm1, bsm1, k);
      if (asm1[k])
	t = bsm1[k] + GTC_ADD_N (v1 + k, v1 + k, bsm1, k);
      if (bsm1[k])
	t += GTC_ADD_N (v1 + k, v1 + k, asm1, k);
      v1[2 * k] = t;
    }
  vm1 = (sreg) { v1, am1_neg ^ bm1_neg };
#ifdef GTC_ACC_ASM
  /* the rotation is odd, and gtc_signed's second buffer the real part, when
     exactly one of ar and br is negative */
  {
    /* the buffers picked by index, not by a branch: the signs are random */
    mp_ptr rb[2] = { R, I };
    int sw = ar_neg ^ br_neg;
    gtc_signed (c, 3, &re, &im, rb[sw], rb[sw ^ 1], ar, ar_neg, a1, 0, br, br_neg,
	  b1, 0, k, 0);
  }
  r2gtc (c, 3, rp, a0, b0, k);
  sl[0] = rp[4 * k];			/* the real part's top limb */
  r2gtc (c, 3, rp + 4 * k, a2, b2, h);
  {
    mp_limb_t t = rp[4 * k];		/* q4's low limb */
    rp[4 * k] = sl[0];
    sl[0] = t;
  }
#else
  gtc_signed (c, 3, &re, &im, R, I, ar, ar_neg, a1, 0, br, br_neg, b1, 0, k, 0);
  r2gtc (c, 3, rp, a0, b0, k);
  r2gtc (c, 3, rp + 4 * k, a2, b2, h);
#endif

  /* Interpolation, paper II (4.11) with the row of (-1,1), at width w.
     With D = Q(-1,1) = q0 - q1 + q2 - q3 + q4 and Re, Im the parts of
     Q(i,1), D - Re + Im = 2 (q2 - q3), so
       q2 = q0 + q4 - Re
       q3 = q2 - (D - Re + Im) / 2
       q1 = Im + q3,
     6 additions and one halving, as (4.11).  D, Re and Im carry lazy
     signs, which only choose between additions and subtractions.  */

  /* v1 <- (D - Re + Im) / 2 = q2 - q3, which lies in (-2X^2, 3X^2); the
     value is (vm1.neg ? -1 : 1) {v1, w}.  */
#ifdef GTC_ACC_ASM
  vm1 = reg_add3_half (v1, vm1, re, -1, im, +1, w);
#else
  vm1 = reg_add (v1, vm1, re, -1, w);
  vm1 = reg_add_half (v1, vm1, im, +1, w);
#endif

  /* q2: the low 2k limbs go directly to {rp + 2k, 2k}, between q0 and q4;
     the top limb (0, 1 or 2) is kept in q2t.  */
#ifdef GTC_ACC_ASM
  {
    /* q0 + q4 - Re in one pass over q4's 2h limbs, then q0 - Re over the
       rest, h < k: at most four limbs.  sr: Re's limbs are subtracted.  */
    unsigned sr = !re.neg;
    mp_size_t t = 2 * k - 2 * h;
#ifdef GTC_ACC_ASM
    /* In place over the real part, whose top limb goes apart and gives
       its place back to q4's low limb.  */
    mp_limb_t retop = rp[4 * k];
    rp[4 * k] = sl[0];
#else
    mp_limb_t retop = re.p[2 * k];
#endif
    q2t = gtc_add3_n (rp + 2 * k, rp, rp + 4 * k, re.p, 2 * h, sr << 1) - sr;
    if (t != 0)
      {
	/* Two or four limbs, inline and without a branch on the signs:
	   -Re as ~Re + 1 - B^t.  */
	mp_ptr d = rp + 2 * k + 2 * h;
	mp_srcptr u = rp + 2 * h, v = re.p + 2 * h;
	mp_limb_t m = -(mp_limb_t) sr;
	__int128 acc = (__int128) (mp_limb_signed_t) q2t + sr;
	mp_size_t i;
	for (i = 0; i < t; i++)
	  {
	    acc += (unsigned __int128) u[i] + (v[i] ^ m);
	    d[i] = (mp_limb_t) acc;
	    acc >>= 64;
	  }
	q2t = (mp_limb_t) acc - sr;
      }
    q2t = sr ? q2t - retop : q2t + retop;
  }
#else
  q2t = gtc_add (rp + 2 * k, rp, 2 * k, rp + 4 * k, 2 * h);
  if (re.neg)
    q2t += re.p[2 * k] + GTC_ADD_N (rp + 2 * k, rp + 2 * k, re.p, 2 * k);
  else
    q2t -= re.p[2 * k] + GTC_SUB_N (rp + 2 * k, rp + 2 * k, re.p, 2 * k);
#endif

  /* v1 <- q2 - (q2 - q3) = q3; with the flag set, v1 holds q3 - q2.  */
  if (vm1.neg)
    v1[2 * k] += q2t + GTC_ADD_N (v1, rp + 2 * k, v1, 2 * k);
  else
    v1[2 * k] = q2t - v1[2 * k] - GTC_SUB_N (v1, rp + 2 * k, v1, 2 * k);

  /* Recomposition: rp = q0 + q1 X + q2 X^2 + q3 X^3 + q4 X^4, where q0,
     the low part of q2 and q4 are already in place.  q3 < 2 B^(k+h).  */
  mpn_add_1 (rp + 4 * k, rp + 4 * k, 2 * h, q2t);
#ifdef GTC_ACC_ASM
  /* q1 = q3 + Im straight into the product, in one pass.  */
  acc3 (rp + k, 2 * n - k, (sreg) { v1, 0 }, im, w);
  (void) cy;
#else
  /* I <- Im + q3 = q1 */
  reg_add (I, (sreg) { v1, 0 }, im, +1, w);
  cy = GTC_ADD_N (rp + k, rp + k, I, w);
  mpn_add_1 (rp + 3 * k + 1, rp + 3 * k + 1, 2 * n - 3 * k - 1, cy);
#endif
  gtc_add (rp + 3 * k, rp + 3 * k, 2 * n - 3 * k, v1, k + h + 1);

  tmp_free (c, &mk);
}

/* The evaluations of one factor of r2gtc_44 at its four points: P(-1) in
   s1, P(i) = xr + i xi, P(-1-i) = yr + i yi, in sign-magnitude form.  */
typedef struct
{
  mp_ptr s1, xr, xi, yr, yi;
  int xr_neg, xi_neg, yr_neg, s1_neg, yi_neg;
} reval4;

/* r2gtc_44's points are (0,1), (-1,1), (i,1), (-1-i,1), (1,0): paper II's
   (Section 4.5) mirrored by x -> -x.  Q(-x) has the coefficients (-1)^j
   q_j and Q(-i) is Q(i)'s conjugate, so the interpolation of paper II's
   points runs on (Q(-1,1), Re Q(i,1), -Im Q(i,1), Q(-1-i,1)), the odd
   coefficients negated at the end: sign flags only.  P(-1) is in (-2X,
   2X), its top limb nonzero 1 time in 12, so the product there is mostly
   the k-limb one, where at (1,1) it is the k+1 limb one almost always.  */

/* The sign of 2 {t, k} - {a, k}: 1 if positive, 0 if not; from the top,
   as abs_sub's comparison.  */
GTC_INLINE int
cmp_lsh1 (mp_srcptr t, mp_srcptr a, mp_size_t k)
{
  mp_size_t i = k - 1;
  mp_limb_t x;

  if (t[i] >> (GMP_NUMB_BITS - 1))
    return 1;
  for (;; i--)
    {
      x = (t[i] << 1) | (i > 0 ? t[i - 1] >> (GMP_NUMB_BITS - 1) : 0);
      if (x != a[i] || i == 0)
	return x > a[i];
    }
}

/* With a = a0 + a1 X + a2 X^2 + a3 X^3, X = B^k, a3 of h limbs:
     P(-1) = (a0 - a1) + (a2 - a3)              in (-2X, 2X)
     P(i) = (a0 - a2) + i (a1 - a3)             each in (-X, X)
     P(-1-i) = (a0 - a1 + 2 a3) + i (2 (a2 - a3) - a1), each in (-3X, 3X)
   s1, yr, yi of k+1 limbs, xr, xi of k; t of k+1 limbs.  */
GTC_INLINE void
rat_eval4 (reval4 *e, mp_srcptr a, mp_size_t k, mp_size_t h, mp_ptr t)
{
  mp_srcptr a0 = a, a1 = a + k, a2 = a + 2 * k, a3 = a + 3 * k;
  /* |v| = |a0 - a1| into yr and |t0| = |a2 - a3| into t with their signs
     vn, tn, each operand order chosen by a comparison first (abs_sub):
       P(-1) = v + t0, P(-1-i) = (v + 2 a3) + i (2 t0 - a1).  */
  int vn, tn;

  vn = abs_sub (e->yr, a0, k, a1, k);
  tn = abs_sub (t, a2, k, a3, h);
  if (vn == tn)
    {
      e->s1[k] = GTC_ADD_N (e->s1, e->yr, t, k);
      e->s1_neg = vn;
    }
  else
    {
      e->s1[k] = 0;
      e->s1_neg = vn ^ abs_sub (e->s1, e->yr, k, t, k);
    }
  /* 2 t0 - a1 */
  if (tn)
    {
      e->yi[k] = GTC_ADDLSH1_N (e->yi, a1, t, k);
      e->yi_neg = 1;
    }
  else if (cmp_lsh1 (t, a1, k))
    {
      e->yi[k] = GTC_RSBLSH1_N (e->yi, a1, t, k);
      e->yi_neg = 0;
    }
  else
    {
      GTC_SUBLSH1_N (e->yi, a1, t, k);
      e->yi[k] = 0;
      e->yi_neg = 1;
    }
  /* v + 2 a3 */
  if (!vn)
    {
      e->yr[k] = GTC_ADDLSH1_N (e->yr, e->yr, a3, h);
      if (h < k)
	e->yr[k] = mpn_add_1 (e->yr + h, e->yr + h, k - h, e->yr[k]);
      e->yr_neg = 0;
    }
  else
    {
      /* 2 a3 - |v|: the sign by a comparison from the top, then one shifted
	 difference, |v| - 2 a3 or 2 a3 - |v| */
      mp_ptr y = e->yr;
      int ge;			/* |v| >= 2 a3 */
      mp_size_t i;
      y[k] = 0;
      if (h + 1 < k && !mpn_zero_p (y + h + 1, k - h - 1))
	ge = 1;
      else
	{
	  /* 2 a3 has h + 1 limbs, its top one a3[h-1] >> 63; y[h] is 0 or a
	     limb of |v| (y[k] = 0 when h = k) */
	  mp_limb_t t2 = a3[h - 1] >> (GMP_NUMB_BITS - 1);
	  if (y[h] != t2)
	    ge = y[h] > t2;
	  else
	    {
	      for (i = h - 1;; i--)
		{
		  mp_limb_t x = (a3[i] << 1)
		    | (i > 0 ? a3[i - 1] >> (GMP_NUMB_BITS - 1) : 0);
		  if (x != y[i] || i == 0)
		    {
		      ge = y[i] >= x;
		      break;
		    }
		}
	    }
	}
      if (ge)
	{
	  /* |v| - 2 a3 >= 0: the value 2 a3 - |v| <= 0 */
	  mp_limb_t bw = GTC_SUBLSH1_N (y, y, a3, h);
	  if (h < k)
	    mpn_sub_1 (y + h, y + h, k - h, bw);
	  e->yr_neg = 1;
	}
      else
	{
	  /* 2 a3 - |v| > 0, below 2 B^h: |v|'s limbs above h are 0 */
	  mp_limb_t top = GTC_RSBLSH1_N (y, y, a3, h);
	  y[h] = top - y[h];
	  e->yr_neg = 0;
	}
    }
  e->xr_neg = abs_sub (e->xr, a0, k, a2, k);
  e->xi_neg = abs_sub (e->xi, a1, k, a3, h);
}

/* x += s c u at width w in place, u >= 0 of un < w limbs, c a small limb
   (1: a plain addition).  */
GTC_INLINE sreg
reg_aorsmul_u (sreg x, mp_srcptr u, mp_size_t un, mp_limb_t c, int s,
	       mp_size_t w)
{
  mp_limb_t hi;

  if ((s > 0) == !x.neg)
    {
      hi = c == 1 ? GTC_ADD_N (x.p, x.p, u, un) : GTC_ADDMUL_1 (x.p, u, un, c);
      mpn_add_1 (x.p + un, x.p + un, w - un, hi);
    }
  else
    {
      hi = c == 1 ? GTC_SUB_N (x.p, x.p, u, un) : GTC_SUBMUL_1 (x.p, u, un, c);
      mpn_sub_1 (x.p + un, x.p + un, w - un, hi);
    }
  return x;
}

/* GTC_RTC4_TOP: r2gtc_44's product at (-1,1), whose factors have top
   limbs 0 or 1, as the k-limb product and the corrections of the top
   limbs where r2gtc_top_rule says (r2gtc_top); else the k+1 limb product
   whenever a top limb is nonzero.  GTC_RTC4_PTOP: the product at
   (-1-i,1), whose factors have top limbs up to 2, as gtc_q1_top's; else
   the k+1 limb product always.  Per microarchitecture.  */
#if defined (__aarch64__)
#define GTC_RTC4_TOP 0
#define GTC_RTC4_PTOP 0
#else
#define GTC_RTC4_TOP 1
#define GTC_RTC4_PTOP 1
#endif

/* The rows of r2gtc_44's interpolation (paper II's R1 to R5).  */
typedef struct
{
  sreg r1, r2, r3, r4, r5;
} rrows4;

/* The evaluations at (-1-i,1), 4k+4 limbs, go into the result area,
   which the products at 0 and infinity overwrite last: 17k+13 limbs of
   scratch, not 21k+17.  */
#define R2GTC_44_ITCH(k) (17 * (k) + 13)

/* Places the evaluations in r2gtc_44's scratch ws of R2GTC_44_ITCH (k)
   limbs, behind the product buffers, and those at (-1-i,1) in the result
   area rp.  */
GTC_INLINE void
r2gtc_44_layout (mp_ptr ws, mp_ptr rp, mp_size_t k, reval4 *ea, reval4 *eb)
{
  mp_size_t k1 = k + 1;
  mp_ptr t = ws + 10 * k + 10;		/* the products' 10k+10 limbs */

  ea->s1 = t + k1; ea->xr = ea->s1 + k1; ea->xi = ea->xr + k;
  eb->s1 = ea->xi + k; eb->xr = eb->s1 + k1; eb->xi = eb->xr + k;
  ea->yr = rp; ea->yi = rp + k1;	/* 4k+4 <= 2n limbs */
  eb->yr = rp + 2 * k1; eb->yi = rp + 3 * k1;
}

/* The evaluations of both factors; t of k+1 limbs.  */
GTC_INLINE void
r2gtc_44_eval (mp_srcptr ap, mp_srcptr bp, mp_size_t k, mp_size_t h,
	   reval4 *ea, reval4 *eb, mp_ptr ws)
{
  mp_ptr t = ws + 10 * k + 10;
  rat_eval4 (ea, ap, k, h, t);
  rat_eval4 (eb, bp, k, h, t);
}

/* Products: R1 = Q(-1,1), R2 + i R3 = Q(i,1), R4 + i R5 = Q(-1-i,1); q0
   = Q(0,1) and q6 = Q(1,0) into their places in the result.  Every value
   fits in w = 2k+1 limbs: |Q(-1-i,1)| < 29 X^2.  */
GTC_INLINE void
r2gtc_44_products (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
	       mp_size_t k, mp_size_t h, const reval4 *ea, const reval4 *eb,
	       mp_ptr ws, rrows4 *r)
{
  mp_size_t w = 2 * k + 1, k1 = k + 1;
  mp_ptr v1 = ws;			/* 2k+2 limbs */
  mp_ptr gi0 = v1 + 2 * k + 2;		/* 2k+1 */
  mp_ptr gi1 = gi0 + w;			/* 2k+1 */
  mp_ptr gp0 = gi1 + w;			/* 2k+3 */
  mp_ptr gp1 = gp0 + 2 * k + 3;		/* 2k+3 */

#if GTC_RTC4_TOP
  r2gtc_top (c, 4, v1, ea->s1, eb->s1, k, 0);
#else
  if ((ea->s1[k] | eb->s1[k]) == 0)
    {
      r2gtc (c, 4, v1, ea->s1, eb->s1, k);
      v1[2 * k] = 0;
    }
  else
    r2gtc (c, 4, v1, ea->s1, eb->s1, k1);
#endif
  r->r1 = (sreg) { v1, ea->s1_neg ^ eb->s1_neg };
  gtc_signed (c, 4, &r->r2, &r->r3, gi0, gi1, ea->xr, ea->xr_neg, ea->xi, ea->xi_neg,
	eb->xr, eb->xr_neg, eb->xi, eb->xi_neg, k, 0);
  gtc_signed (c, 4, &r->r4, &r->r5, gp0, gp1, ea->yr, ea->yr_neg, ea->yi, ea->yi_neg,
	eb->yr, eb->yr_neg, eb->yi, eb->yi_neg, k1, GTC_RTC4_PTOP);
  r2gtc (c, 4, rp, ap, bp, k);
  r2gtc (c, 4, rp + 6 * k, ap + 3 * k, bp + 3 * k, h);
}

#ifdef GTC_ACC_ASM
/* (x +- 4 y)/5 in one pass.  */
mp_limb_t gtc_div5_lsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t,
			  unsigned);
#endif

/* GTC_RTC4_SHLD: r2gtc_44's division by 5 by gtc_div5n_rlsh_n, which
   makes the limbs of 4 q4 by shld and the quotient's negative by two
   carry chains, the flag flipped after; else gtc_div5_lsh_n.  Per
   microarchitecture.  */
#if defined (GTC_ASM_X86) && !defined (GTC_ZEN)
#define GTC_RTC4_SHLD 1
mp_limb_t gtc_div5n_rlsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t,
			    unsigned);
#else
#define GTC_RTC4_SHLD 0
#endif

/* x with its flag cleared, the value negated in place if it was set.  */
GTC_INLINE sreg
reg_true (sreg x, mp_size_t w)
{
  if (x.neg)
    {
      mpn_neg (x.p, x.p, w);
      x.neg = 0;
    }
  return x;
}

/* Interpolation at width w, R0 = q0 and R6 = q6 read in the result; then
   the recomposition.  On the values of paper II's points (S = Q(1,1),
   Re_i + i Im_i = Q(i,1), Re_p + i Im_p = Q(1+i,1)) of the mirrored
   product, Q(-1,1), Re Q(i,1) - i Im Q(i,1) and Q(-1-i,1): with A0 = S -
   Im_i - q0 = q2 + 2 q3 + q4 + q6 and B = Im_p - (Re_p - q0) + 10 q6 =
   2 q2 + 4 q3 + 4 q4 + 2 q6,
     q4 = (B - 2 A0)/2,  q2' = q0 + q4 - Re_i = q2 + q6,
     q3 = (A0 - q2' - q4)/2,
     5 q5 = Im_i - q3 - (Re_p - q0) - 4 q4,  q1 = Im_i + q3 - q5,
   and q2 = q2' - q6 last: q6 enters in one pass of 2h limbs, not three.
   The sums of three are one pass each; q4 goes straight into its place in
   the result (its top limb over q6's low one, saved meanwhile), q2 into
   its place as q2' - q6, so neither is copied; q1 is added in one pass
   with q5.  With the kernels, A0's buffer holds -A0 = q0 + Im_i - S, so
   that q4 and q2' come out with a clear flag; without them (the
   two-pass sums, which must not write their third operand), they are
   negated where the flag is set.  */
GTC_INLINE void
r2gtc_44_interp (mp_ptr rp, mp_size_t n, mp_size_t k, mp_size_t h, rrows4 *r)
{
  mp_size_t w = 2 * k + 1;
  mp_srcptr q6 = rp + 6 * k;
  /* S, Re_i, Im_i, Re_p, Im_p; q0 at width w over a cleared limb of the
     gap */
  sreg s = r->r1, ri = r->r2, ii = r->r3, rq = r->r4, iq = r->r5;
  sreg q0w = { rp, 0 }, a, b, x, q4;
  mp_limb_t t2, t4, s6, cy;

  rp[2 * k] = 0;
  ii.neg ^= 1;				/* Im Q(-i) */
  /* A0 = S - Im_i - q0 */
#ifdef GTC_ACC_ASM
  a = reg_add3 (s.p, q0w, ii, 1, s, -1, w);
  a.neg = 1;				/* the buffer holds -A0 */
#else
  a = reg_add (s.p, s, ii, -1, w);
  a = reg_add (s.p, a, q0w, -1, w);
#endif
  /* Re_p - q0;  B = Im_p - (Re_p - q0) + 10 q6 */
  rq = reg_add (rq.p, rq, q0w, -1, w);
  b = reg_add (iq.p, iq, rq, -1, w);
  b = reg_aorsmul_u (b, q6, 2 * h, 10, 1, w);
  /* q4 = (B - 2 A0)/2 into its place, its top limb over q6's low one */
  s6 = rp[6 * k];
  q4 = reg_true (reg_add3_half_any (rp + 4 * k, b, a, -1, a, -1, w), w);
  /* q2' = q0 + q4 - Re_i = q2 + q6, into Re_i */
#ifdef GTC_ACC_ASM
  ri = reg_add3 (ri.p, q0w, q4, 1, ri, -1, w);
#else
  ri = reg_add (ri.p, ri, q4, -1, w);
  ri = reg_add (ri.p, ri, q0w, -1, w);
  ri.neg ^= 1;
#endif
  ri = reg_true (ri, w);
  /* q3 = (A0 - q2' - q4)/2, into A0 */
  a = reg_add3_half_any (a.p, a, ri, -1, q4, -1, w);
  /* 5 q5 = Im_i - q3 - (Re_p - q0) - 4 q4, into Re_p - q0, as -((Re_p -
     q0) - Im_i + q3): the destination is the first operand, which the
     two passes without gtc_add3_n need */
  x = reg_add3 (rq.p, rq, ii, -1, a, 1, w);
  x.neg ^= 1;
#if GTC_RTC4_SHLD
  {
    /* (x - 4 q4)/5 of the stored limbs where the flags agree, -(x + 4
       q4)/5 where they differ, whose flag flips */
    unsigned sub = x.neg == q4.neg;
    gtc_div5n_rlsh_n (x.p, x.p, q4.p, w, sub);
    x.neg ^= sub ^ 1;
  }
#elif defined (GTC_ACC_ASM)
  /* the kernel subtracts 4 q4 where the flags agree */
  gtc_div5_lsh_n (x.p, x.p, q4.p, w, x.neg == q4.neg);
#else
  x = reg_aorsmul (x, q4, 4, -1, w, w);
  mpn_bdiv_dbm1c (x.p, x.p, w, GMP_NUMB_MAX / 5, 0);
#endif
  /* q1 + q5 = Im_i + q3, into Im_i */
  ii = reg_add (ii.p, ii, a, 1, w);
  /* the odd coefficients of Q(-x) negated */
  a.neg ^= 1;
  x.neg ^= 1;
  ii.neg ^= 1;

  /* Recomposition: q4's top limb out and q6's low limb back; q2 = q2' -
     q6 into its place; the top limbs carried on; then q1 = (q1 + q5) -
     q5, q3 and q5 added.  */
  t4 = rp[6 * k];
  rp[6 * k] = s6;
  cy = GTC_SUB_N (rp + 2 * k, ri.p, q6, 2 * h);
  if (h < k)
    cy = mpn_sub_1 (rp + 2 * k + 2 * h, ri.p + 2 * h, 2 * k - 2 * h, cy);
  t2 = ri.p[2 * k] - cy;
  acc_limb (rp + 6 * k, 2 * n - 6 * k, t4);
  acc_limb (rp + 4 * k, 2 * n - 4 * k, t2);
#ifdef GTC_ACC_ASM
  acc3 (rp + k, 2 * n - k, ii, (sreg) { x.p, !x.neg }, w);
#else
  acc (rp + k, 2 * n - k, ii, w);
  acc (rp + k, 2 * n - k, (sreg) { x.p, !x.neg }, w);
#endif
  acc (rp + 3 * k, 2 * n - 3 * k, a, w);
  acc (rp + 5 * k, 2 * n - 5 * k, x, w);
}

/* One level of the Toom-Cook-4 style reduction to Gaussian multiplication
   of paper II, Section 4.5, with its points mirrored: (0,1), (-1,1), (1,0)
   (rational) and (i,1), (-1-i,1) (Gaussian), the latter as two Gaussian
   products whose real and imaginary parts give four rational equations
   (see reval4).  Three rational products and two Gaussian ones of k = ceil(n/4) limbs.  n >= 13.  */
GTC_OUTLINE void
r2gtc_44 (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  mp_size_t k = (n + 3) / 4, h = n - 3 * k;	/* parts of k, k, k and h */
  tmp_mark mk;
  mp_ptr ws = tmp_alloc (c, &mk, R2GTC_44_ITCH (k));
  reval4 ea, eb;
  rrows4 r;

  r2gtc_44_layout (ws, rp, k, &ea, &eb);
  r2gtc_44_eval (ap, bp, k, h, &ea, &eb, ws);
  r2gtc_44_products (c, rp, ap, bp, k, h, &ea, &eb, ws, &r);
  r2gtc_44_interp (rp, n, k, h, &r);
  tmp_free (c, &mk);
}

/* rat_eval5's choices, per microarchitecture.
   GTC_RTC5_S1SIGN: where h < k and k >= GTC_RTC5_S1SIGN_K, P(-1) by the
   sign its top limbs give, no negation; else in two's complement, then
   tc_abs.
   GTC_RTC5_XRABS: Re P(i) = |D - a2| by a branch-free comparison of the
   top two limbs; else abs_sub.
   GTC_RTC5_MISIGN: Im P(-1+i) = v - 2 a2 by its sign from the top limbs,
   2 a2 - v by GTC_RSBLSH1_N where negative; else v - 2 a2, then tc_abs.
   And r2gtc_55_interp's:
   GTC_RTC5_Q8LSH2: 4 q8 by GTC_ADDLSH2_N; else GTC_ADDMUL_1.  */
#if defined (GTC_ZEN)
#define GTC_RTC5_S1SIGN 0
#define GTC_RTC5_XRABS 0
#else
#define GTC_RTC5_S1SIGN 1
#define GTC_RTC5_XRABS 1
#endif
#if defined (__AVX512F__)
#define GTC_RTC5_S1SIGN_K 600
#else
#define GTC_RTC5_S1SIGN_K 60
#endif
#if defined (__aarch64__)
#define GTC_RTC5_MISIGN 1
#else
#define GTC_RTC5_MISIGN 0
#endif
#if defined (__aarch64__) || defined (__AVX512F__) || defined (GTC_ASM_X86)
#define GTC_RTC5_Q8LSH2 1
#else
#define GTC_RTC5_Q8LSH2 0
#endif

/* The evaluations of one factor of r2gtc_55 at its four points: P(-1) in
   s1, P(i) = xr + i xi, P(1+i) = pr + i pi, P(-1+i) = mr + i mi, k+1
   limbs each, in sign-magnitude form.  */
typedef struct
{
  mp_ptr s1, xr, xi, pr, pi, mr, mi;
  int s1_neg, xr_neg, xi_neg, pr_neg, pi_neg, mr_neg, mi_neg;
} reval5;

/* With a = a0 + a1 X + ... + a4 X^4, X = B^k, a4 of h limbs, E = a0 + a2
   + a4, O = a1 + a3, t4 = a0 - 4 a4, u = a1 - 2 a3 and v = a1 + 2 a3
   (paper II, (4.30)):
     P(-1) = E - O                              in (-2X, 3X)
     P(i) = (a0 - a2 + a4) + i (a1 - a3)        (-X, 2X), (-X, X)
     P(1+i) = (t4 + u) + i (v + 2 a2)           (-6X, 2X), [0, 5X)
     P(-1+i) = (t4 - u) + i (v - 2 a2)          (-5X, 3X), (-2X, 3X)
   t of 3k+3 limbs.  */
GTC_INLINE void
rat_eval5 (reval5 *e, mp_srcptr a, mp_size_t k, mp_size_t h, mp_ptr t)
{
  mp_size_t k1 = k + 1;
  mp_srcptr a0 = a, a1 = a + k, a2 = a + 2 * k, a3 = a + 3 * k;
  mp_srcptr a4 = a + 4 * k;
  mp_ptr u = t, t4 = t + k1;
  mp_limb_t b;

  /* D = a0 + a4 into xr; P(-1) = D + a2 - a1 - a3 in two's complement */
  e->xr[k] = gtc_add (e->xr, a0, k, a4, h);
#ifdef GTC_ACC_ASM
#if GTC_RTC5_S1SIGN
  /* The sign from the top limbs: with T = (D[k] B + D[k-1]) + a2[k-1] -
     a1[k-1] - a3[k-1], P(-1) = T B^(k-1) + L, |L| < 2 B^(k-1); T >= 2 is
     positive, T <= -2 negative, and either is |P(-1)| = x + y - z - w in
     two passes, (D, a2, a1, a3) or (a1, a3, D, a2) picked by masks.
     Otherwise (|T| <= 1, about never) the difference, then tc_abs.  */
  __int128 tt;
  if (h < k && k >= GTC_RTC5_S1SIGN_K
      && (tt = ((__int128) e->xr[k] << GMP_NUMB_BITS) + e->xr[k - 1]
	  + a2[k - 1] - a1[k - 1] - a3[k - 1],
	  __builtin_expect (tt >= 2 || tt <= -2, 1)))
    {
      int n1 = tt < 0;
      mp_limb_t m = -(mp_limb_t) n1;
      b = gtc_add3_n (e->s1, gtc_sel (n1, a1, e->xr), gtc_sel (n1, a3, a2),
		      gtc_sel (n1, e->xr, a1), k, 2);
      b -= GTC_SUB_N (e->s1, e->s1, gtc_sel (n1, a2, a3), k);
      e->s1[k] = (e->xr[k] ^ m) + n1 + b - 1;
      e->s1_neg = n1;
    }
  else
#endif
    {
      b = gtc_add3_n (e->s1, e->xr, a2, a1, k, 2);
      b -= GTC_SUB_N (e->s1, e->s1, a3, k);
      e->s1[k] = e->xr[k] + b - 1;
      e->s1_neg = tc_abs (e->s1, k1);
    }
#else
  e->s1[k] = e->xr[k] + GTC_ADD_N (e->s1, e->xr, a2, k);
  t[k] = GTC_ADD_N (t, a1, a3, k);
  e->s1_neg = abs_sub (e->s1, e->s1, k1, t, k1);
#endif
#if GTC_RTC5_XRABS
  {
    /* |D - a2|, D of k+1 limbs whose top is 0 or 1: the comparison of
       the top two limbs at once, a loop only where they tie, the
       operands by masks (abs_sub branches on D's top limb, nonzero up to
       half the time) */
    mp_ptr d = e->xr;
    mp_limb_t d1 = d[k], d0 = d[k - 1], cy;
    int lt = (d1 == 0) & (d0 < a2[k - 1]);
    if (__builtin_expect ((d1 | (d0 ^ a2[k - 1])) == 0, 0))
      {
	mp_size_t i = k - 2;
	while (i > 0 && d[i] == a2[i])
	  i--;
	lt = d[i] < a2[i];
      }
    cy = GTC_SUB_N (d, gtc_sel (lt, a2, d), gtc_sel (lt, d, a2), k);
    d[k] = (d1 - cy) & ~-(mp_limb_t) lt;
    e->xr_neg = lt;
  }
#else
  e->xr_neg = abs_sub (e->xr, e->xr, k1, a2, k);
#endif
  e->xi_neg = abs_sub (e->xi, a1, k, a3, k);
  e->xi[k] = 0;
  /* u and t4 in two's complement, v into pi */
  b = GTC_SUBLSH1_N (u, a1, a3, k);
  u[k] = -b;
  e->pi[k] = GTC_ADDLSH1_N (e->pi, a1, a3, k);
  {
    /* nt4 = 4 a4 - a0 = -t4: Re P(1+i) = u - nt4 by a comparison, Re
       P(-1+i) = -(nt4 + u) with the negation, as t4 + u is mostly
       negative (h >= k - 4, so a4 is about as long as a0) */
    mp_limb_signed_t r = GTC_RSBLSH2_N (t4, a0, a4, h);
    if (h < k)
      {
	/* r, sign-extended, minus a0's high limbs (k - h <= 4) */
	mp_limb_t c = (mp_limb_t) r, sx = -(mp_limb_t) (r < 0), bw = 0;
	for (mp_size_t i = h; i < k; i++)
	  {
	    mp_limb_t d, o1 = __builtin_sub_overflow (c, a0[i], &d);
	    o1 |= __builtin_sub_overflow (d, bw, &d);
	    t4[i] = d;
	    bw = o1;
	    c = sx;
	  }
	t4[k] = sx - bw;
      }
    else
      t4[k] = (mp_limb_t) r;
    e->pr_neg = tc_abs_sub (e->pr, u, t4, k1);
    GTC_ADD_N (e->mr, t4, u, k1);
    e->mr_neg = 1 ^ tc_abs (e->mr, k1);
  }
  /* v - 2 a2, then v + 2 a2 in place */
#if GTC_RTC5_MISIGN
  /* where the top limbs say it is negative (the limbs below add between
     -2 and 1 units of B^(k-1)), 2 a2 - v */
  if ((((__int128) e->pi[k] << GMP_NUMB_BITS) + e->pi[k - 1]
       - ((__int128) a2[k - 1] << 1)) <= -2)
    {
      e->mi[k] = (mp_limb_t) GTC_RSBLSH1_N (e->mi, e->pi, a2, k) - e->pi[k];
      e->mi_neg = 1;
    }
  else
#endif
    {
      b = GTC_SUBLSH1_N (e->mi, e->pi, a2, k);
      e->mi[k] = e->pi[k] - b;
      e->mi_neg = tc_abs (e->mi, k1);
    }
  e->pi[k] += GTC_ADDLSH1_N (e->pi, e->pi, a2, k);
  e->pi_neg = 0;
}

/* The rows of r2gtc_55's interpolation: Q(-1,1), Q(i,1), Q(1+i,1),
   Q(-1+i,1).  */
typedef struct
{
  sreg m, ri, ii, rp, ip, rm, im;
} rrows5;

/* GTC_RTC5_PTOP: the products at (1+i,1) and (-1+i,1), whose factors
   have top limbs up to 5, as gtc_q1_top's; else the k+1 limb product
   always.  Per microarchitecture.  */
#if defined (__aarch64__)
#define GTC_RTC5_PTOP 0
#else
#define GTC_RTC5_PTOP 1
#endif

/* With gap, the first factor's evaluations, 7k+7 limbs, go into the
   result area where they fit (7k + 7 <= 2n), which the products at 0 and
   infinity overwrite last; the evaluations' temporaries in the products'
   buffers, the interpolation's spare one over the second factor's
   evaluations.  */
#define R2GTC_55_ITCH(k, gap) ((gap) ? 21 * (k) + 27 : 28 * (k) + 34)

/* d = (x + s y) / 4, exact, at width w.  */
GTC_INLINE sreg
r2gtc_55_quarter (mp_ptr dp, sreg x, sreg y, int s, mp_size_t w)
{
#ifdef GTC_ACC_ASM
  return reg_add_quarter (dp, x, y, s, w);
#else
  return reg_rsh (reg_add (dp, x, y, s, w), 2, w);
#endif
}

#ifdef GTC_ACC_ASM
/* q = (c - A)/5 with A = (x + s y)/2^sh (q5: sh = 2, q7: 3): q added at
   rp + o1 and c - q at rp + o2, both sign-extended over the len limbs,
   in one pass.  q and c - q are coefficients of the product, below 5
   B^(w-1): q is never -B^w/2, which the q pass excludes.  tmp: w limbs,
   used where q's region runs past rp's end.  */
GTC_INLINE void
rtc5_qacc (mp_ptr rp, mp_size_t len, mp_size_t o1, mp_size_t o2,
	   mp_size_t w, sreg x, sreg y, int s, unsigned sh, sreg c,
	   mp_ptr tmp)
{
  /* t = A x's sign as stored, q' = (t -+ c)/5, q = -x's sign q' */
  unsigned ysub = (s > 0) != (x.neg == y.neg);
  unsigned csub = x.neg == c.neg, asub = !x.neg;
  mp_size_t m1 = len - o1;
  mp_ptr a1 = rp + o1;
  mp_limb_t nc;

  if (m1 < w)
    a1 = memcpy (tmp, a1, m1 * sizeof (mp_limb_t));
  nc = GTC_TC5Q_N (a1, rp + o2, x.p, y.p, c.p, w,
		   ysub | csub << 1 | asub << 2 | (sh == 3) << 3);
  if (m1 < w)
    memcpy (rp + o1, tmp, m1 * sizeof (mp_limb_t));
  else if (m1 > w)
    acc_limb_bf (rp + o1 + w, m1 - w,
		 (mp_limb_t) (mp_limb_signed_t) (int32_t) nc);
  acc_limb_bf (rp + o2 + w, len - o2 - w,
	       (mp_limb_t) (mp_limb_signed_t) (int32_t) (nc >> 32));
}
#else
/* d = (x + sy y + sz z) / 2^sh, exact, at width w; dp may be x's
   buffer.  */
GTC_INLINE sreg
r2gtc_55_add3_shr (mp_ptr dp, sreg x, sreg y, int sy, sreg z, int sz,
	       unsigned sh, mp_size_t w)
{
  x = reg_add (dp, x, y, sy, w);
  return reg_rsh (reg_add (dp, x, z, sz, w), sh, w);
}

/* d = (x + s y) / 5, exact, at width w; dp x's buffer, y's or one that
   overlaps neither.  */
GTC_INLINE sreg
r2gtc_55_div5 (mp_ptr dp, sreg x, sreg y, int s, mp_size_t w)
{
  x = reg_add (dp, x, y, s, w);
  mpn_bdiv_dbm1c (dp, dp, w, GMP_NUMB_MAX / 5, 0);
  return x;
}
#endif

/* Interpolation at width w = 2k+1 into the result rp, q0 and q8 already
   there.  With Re_p + i Im_p = Q(1+i), Re_m + i Im_m = Q(-1+i), Re_i + i
   Im_i = Q(i), M = Q(-1), S = Im_p + Im_m and d = Re_p - Re_m:
     A0 = (Re_p + Re_m)/2 = q0 - 4 q4 + 16 q8,  q4 = (q0 - A0)/4 + 4 q8,
     C2 = q0 + q4 + q8 - Re_i = q2 + q6,
     c1 = (Re_i - M + Im_i)/2 + C2 = q1 + q5,  c3 = c1 - Im_i = q3 + q7,
     q6 = (C2 - (Im_p - Im_m)/4)/5,  q2 = C2 - q6,
     q5 = (c1 - (S + d)/4)/5,  q7 = (c3 - (S - d)/8)/5,
     q1 = c1 - q5,  q3 = c3 - q7.
   T is scratch of w limbs.  */
#ifdef GTC_ACC_ASM
/* q6 with q2, and q5 and q7 each with q1 or q3, in one pass, the
   quarters and eighths never stored.  */
GTC_INLINE void
r2gtc_55_interp (mp_ptr rp, mp_size_t n, mp_size_t k, mp_size_t h, rrows5 *r,
	     mp_ptr T)
{
  mp_size_t w = 2 * k + 1;
  sreg q0w = { rp, 0 }, s, d, a0, q4, c2, hh, c1, c3;
  mp_srcptr q8 = rp + 8 * k;
  mp_limb_t t2, t4, t6, s8, cy;

  rp[2 * k] = 0;
  s = reg_add (T, r->ip, r->im, 1, w);
  /* d over Re_m, then A0 = (2 Re_p - d)/2 over Re_p */
  d = reg_add (r->rm.p, r->rp, r->rm, -1, w);
  a0 = reg_add3_half (r->rp.p, r->rp, r->rp, 1, d, -1, w);
  /* q4 into its place, its top limb over q6's low one */
  q4 = r2gtc_55_quarter (rp + 4 * k, q0w, a0, -1, w);
#if GTC_RTC5_Q8LSH2
  cy = GTC_ADDLSH2_N (q4.p, q4.p, q8, 2 * h);
#else
  cy = GTC_ADDMUL_1 (q4.p, q8, 2 * h, 4);
#endif
  mpn_add_1 (q4.p + 2 * h, q4.p + 2 * h, w - 2 * h, cy);
  /* C2 into A0's buffer, its flag clear */
  c2 = reg_add3 (r->rp.p, q0w, q4, 1, r->ri, -1, w);
  cy = GTC_ADD_N (c2.p, c2.p, q8, 2 * h);
  mpn_add_1 (c2.p + 2 * h, c2.p + 2 * h, w - 2 * h, cy);
  /* c1 into Re_i's buffer, c3 into Im_i's */
  hh = reg_add3_half_any (r->m.p, r->ri, r->m, -1, r->ii, 1, w);
  c1 = reg_add (r->ri.p, hh, c2, 1, w);
  c3 = reg_add (r->ii.p, c1, r->ii, -1, w);
  /* q6 and q2 into their places: q4's top and low limbs and q8's low
     limb saved */
  t4 = rp[6 * k];
  s8 = rp[8 * k];
  {
    mp_limb_t l4 = rp[4 * k];
    gtc_tc5s_n (rp + 6 * k, rp + 2 * k, r->ip.p, r->im.p, c2.p, w,
		(r->ip.neg == r->im.neg) | (r->ip.neg ? 0 : 4));
    t2 = rp[4 * k];
    rp[4 * k] = l4;
  }
  t6 = rp[8 * k];
  rp[8 * k] = s8;
  acc_limb (rp + 4 * k, 2 * n - 4 * k, t2);
  acc_limb (rp + 6 * k, 2 * n - 6 * k, t4);
  acc_limb (rp + 8 * k, 2 * n - 8 * k, t6);
  /* q5 = (c1 - (S + d)/4)/5 at 5k, c1 - q5 at k; q7 = (c3 - (S - d)/8)/5
     at 7k, c3 - q7 at 3k */
  rtc5_qacc (rp, 2 * n, 5 * k, k, w, s, d, 1, 2, c1, r->m.p);
  rtc5_qacc (rp, 2 * n, 7 * k, 3 * k, w, s, d, -1, 3, c3, r->m.p);
}
#else
/* The quarters and eighths stored, the divisions by GMP's, q1 and q3
   added as their two terms.  */
GTC_INLINE void
r2gtc_55_interp (mp_ptr rp, mp_size_t n, mp_size_t k, mp_size_t h, rrows5 *r,
	     mp_ptr T)
{
  mp_size_t w = 2 * k + 1;
  sreg q0w = { rp, 0 }, s, a1, a2, a3, a0, q4, c2, hh, c1, c3, q5, q6, q7;
  mp_srcptr q8 = rp + 8 * k;
  mp_limb_t t2, t4, t6, s8, cy;

  rp[2 * k] = 0;
  s = reg_add (T, r->ip, r->im, 1, w);
  a2 = r2gtc_55_quarter (r->im.p, r->ip, r->im, -1, w);
  a1 = r2gtc_55_add3_shr (r->ip.p, s, r->rp, 1, r->rm, -1, 2, w);
  a3 = r2gtc_55_add3_shr (T, s, r->rp, -1, r->rm, 1, 3, w);
  /* q4 into its place, its top limb over q6's low one */
  a0 = reg_add_half (r->rp.p, r->rp, r->rm, 1, w);
  q4 = r2gtc_55_quarter (rp + 4 * k, q0w, a0, -1, w);
#if GTC_RTC5_Q8LSH2
  cy = GTC_ADDLSH2_N (q4.p, q4.p, q8, 2 * h);
#else
  cy = GTC_ADDMUL_1 (q4.p, q8, 2 * h, 4);
#endif
  mpn_add_1 (q4.p + 2 * h, q4.p + 2 * h, w - 2 * h, cy);
  /* C2 into Re_m's buffer, its flag clear */
  c2 = reg_add3 (r->rm.p, q0w, q4, 1, r->ri, -1, w);
  cy = GTC_ADD_N (c2.p, c2.p, q8, 2 * h);
  mpn_add_1 (c2.p + 2 * h, c2.p + 2 * h, w - 2 * h, cy);
  /* c1 into Re_i's buffer, c3 into Im_i's */
  hh = reg_add3_half_any (r->m.p, r->ri, r->m, -1, r->ii, 1, w);
  c1 = reg_add (r->ri.p, hh, c2, 1, w);
  c3 = reg_add (r->ii.p, c1, r->ii, -1, w);
  /* q6 into its place: q4's top limb and q8's low limb saved */
  t4 = rp[6 * k];
  s8 = rp[8 * k];
  q6 = r2gtc_55_div5 (rp + 6 * k, c2, a2, -1, w);
  /* q2 = C2 - q6, both flags clear, into its place but its top limb */
  cy = GTC_SUB_N (rp + 2 * k, c2.p, q6.p, 2 * k);
  t2 = c2.p[2 * k] - q6.p[2 * k] - cy;
  t6 = rp[8 * k];
  rp[8 * k] = s8;
  q5 = r2gtc_55_div5 (a1.p, c1, a1, -1, w);
  q7 = r2gtc_55_div5 (a3.p, c3, a3, -1, w);

  acc_limb (rp + 4 * k, 2 * n - 4 * k, t2);
  acc_limb (rp + 6 * k, 2 * n - 6 * k, t4);
  acc_limb (rp + 8 * k, 2 * n - 8 * k, t6);
  acc (rp + k, 2 * n - k, c1, w);
  acc (rp + k, 2 * n - k, (sreg) { q5.p, !q5.neg }, w);
  acc (rp + 3 * k, 2 * n - 3 * k, c3, w);
  acc (rp + 3 * k, 2 * n - 3 * k, (sreg) { q7.p, !q7.neg }, w);
  acc (rp + 5 * k, 2 * n - 5 * k, q5, w);
  acc (rp + 7 * k, 2 * n - 7 * k, q7, w);
}
#endif

/* One level of the Toom-Cook-5 style reduction to Gaussian
   multiplication of paper II, Section 4.6, with its rational point (1,1)
   mirrored to (-1,1), as r2gtc_44's: (0,1), (-1,1), (1,0) (rational) and
   (i,1), (1+i,1), (-1+i,1) (Gaussian; a set the mirror x -> -x maps to
   its conjugates).  Three rational products and three Gaussian ones of k
   = ceil(n/5) limbs.  n >= 17.  P(-1) has a top limb of 0 to 2 that is 0
   for 77-92% of factors, P(1) one of 0 to 4 that is 0 for 1-4%, so the
   product at (-1,1) is the k-limb one for 59-84% of operand pairs; with
   a top limb nonzero, the k-limb product and the corrections where
   r2gtc_top_rule says.  */
GTC_OUTLINE void
r2gtc_55 (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  mp_size_t k = (n + 4) / 5, h = n - 4 * k, k1 = k + 1;
  mp_size_t g = 2 * k + 3;
  int gap = 7 * k1 <= 2 * n;
  tmp_mark mk;
  mp_ptr ws = tmp_alloc (c, &mk, R2GTC_55_ITCH (k, gap));
  mp_ptr v1 = ws, G = v1 + 2 * k + 2, evb = G + 6 * g, eva, t, T;
  reval5 ea, eb;
  rrows5 r;

  eva = gap ? rp : evb + 7 * k1;
  t = G;
  T = evb;
  ea.s1 = eva; ea.xr = ea.s1 + k1; ea.xi = ea.xr + k1; ea.pr = ea.xi + k1;
  ea.pi = ea.pr + k1; ea.mr = ea.pi + k1; ea.mi = ea.mr + k1;
  eb.s1 = evb; eb.xr = eb.s1 + k1; eb.xi = eb.xr + k1;
  eb.pr = eb.xi + k1; eb.pi = eb.pr + k1; eb.mr = eb.pi + k1;
  eb.mi = eb.mr + k1;
  rat_eval5 (&ea, ap, k, h, t);
  rat_eval5 (&eb, bp, k, h, t);

  r2gtc_top (c, 5, v1, ea.s1, eb.s1, k, 0);
  r.m = (sreg) { v1, ea.s1_neg ^ eb.s1_neg };
  gtc_signed (c, 5, &r.ri, &r.ii, G, G + g, ea.xr, ea.xr_neg, ea.xi, ea.xi_neg,
	eb.xr, eb.xr_neg, eb.xi, eb.xi_neg, k1, 1);
  gtc_signed (c, 5, &r.rp, &r.ip, G + 2 * g, G + 3 * g, ea.pr, ea.pr_neg, ea.pi,
	ea.pi_neg, eb.pr, eb.pr_neg, eb.pi, eb.pi_neg, k1, GTC_RTC5_PTOP);
  gtc_signed (c, 5, &r.rm, &r.im, G + 4 * g, G + 5 * g, ea.mr, ea.mr_neg, ea.mi,
	ea.mi_neg, eb.mr, eb.mr_neg, eb.mi, eb.mi_neg, k1, GTC_RTC5_PTOP);
  r2gtc (c, 5, rp, ap, bp, k);
  r2gtc (c, 5, rp + 8 * k, ap + 4 * k, bp + 4 * k, h);
  r2gtc_55_interp (rp, n, k, h, &r, T);
  tmp_free (c, &mk);
}

/* {rp, an+bn} = {ap, an} * {bp, bn}, an >= bn >= 1.  */
static void
mul_any (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_size_t an,
	 mp_srcptr bp, mp_size_t bn)
{
  tmp_mark mk;
  mp_ptr t;
  mp_size_t done, r;
  mp_limb_t cy;

  if (an == bn)
    {
      r2gtc (c, 5, rp, ap, bp, an);
      return;
    }
  if (r2gtc_is_tb (c, 5, bn))
    {
      mpn_mul_basecase (rp, ap, an, bp, bn);
      return;
    }

  /* Unbalanced: multiply b by bn-limb chunks of a, then by the rest, each
     product straight into its place; the bn limbs of the one before that
     it overlaps are saved first and added back.  */
  t = tmp_alloc (c, &mk, bn);
  r2gtc (c, 5, rp, ap, bp, bn);
  for (done = bn; an - done >= bn; done += bn)
    {
      mpn_copyi (t, rp + done, bn);
      r2gtc (c, 5, rp + done, ap + done, bp, bn);
      cy = GTC_ADD_N (rp + done, rp + done, t, bn);
      mpn_add_1 (rp + done + bn, rp + done + bn, bn, cy);
    }
  r = an - done;
  if (r > 0)
    {
      mpn_copyi (t, rp + done, bn);
      mul_any (c, rp + done, bp, bn, ap + done, r);
      cy = GTC_ADD_N (rp + done, rp + done, t, bn);
      mpn_add_1 (rp + done + bn, rp + done + bn, r, cy);
    }
  tmp_free (c, &mk);
}
