/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   Gaussian multiplication: the base cases, the Gaussian Karatsuba (gtc_22)
   and the Gaussian Toom-Cook-3 (gtc_33).

   The products of the levels take first-quadrant factors (all components
   >= 0), P + iQ = (ar + i ai)(br + i bi), and give P and Q of 2n+1 limbs in
   two's complement.  */

/* The dispatchers are forced inline: a node then calls its children's
   nodes directly.  I is the cap: the products of gtc_ii and r2gtc_ii are
   dispatched with I = i, gtc's and r2gtc's with I = 5.  */
GTC_INLINE void r2gtc (gtc_ctx *, int, mp_ptr, mp_srcptr, mp_srcptr,
		       mp_size_t);
GTC_INLINE void gtc_q1 (gtc_ctx *, int, mp_ptr, mp_ptr, mp_srcptr, mp_srcptr,
			 mp_srcptr, mp_srcptr, mp_size_t);

/* Whether a rational n x n product under the cap I is the textbook
   product.  */
GTC_INLINE int
r2gtc_is_tb (const gtc_ctx *c, int I, mp_size_t n)
{
  return r2gtc_alg (c, I, n) == R2GTC_ALG_TB;
}

/* 1-limb components, four hardware multiplications; P and Q get 3 limbs.  */
GTC_INLINE void
gtc_1x1 (mp_ptr P, mp_ptr Q, mp_limb_t a, mp_limb_t b, mp_limb_t c,
	  mp_limb_t d)
{
  unsigned __int128 ac = (unsigned __int128) a * c;
  unsigned __int128 bd = (unsigned __int128) b * d;
  unsigned __int128 ad = (unsigned __int128) a * d;
  unsigned __int128 bc = (unsigned __int128) b * c;
  unsigned __int128 p = ac - bd, q = ad + bc;
  P[0] = (mp_limb_t) p;
  P[1] = (mp_limb_t) (p >> 64);
  P[2] = -(mp_limb_t) (ac < bd);
  Q[0] = (mp_limb_t) q;
  Q[1] = (mp_limb_t) (q >> 64);
  Q[2] = q < ad;
}

/* The textbook product, the four real products of the definition.  With
   the assembly it is a kernel per size (gtc_textbook_*.S); otherwise the C
   below: product scanning into a three-limb accumulator, unrolled, one
   function per size.  */

typedef unsigned __int128 gtc_u128;

/* (a2 a1 a0) += x, three limbs plus a 128-bit value.  */
#define GTC_ACC3(a0, a1, a2, x)						\
  do									\
    {									\
      gtc_u128 s_ = (gtc_u128) (a1) << 64 | (a0);			\
      (a2) += __builtin_add_overflow (s_, (gtc_u128) (x), &s_);		\
      (a0) = (mp_limb_t) s_;						\
      (a1) = (mp_limb_t) (s_ >> 64);					\
    }									\
  while (0)

/* {R, m+c+1} = a b + x y (sub = 0), or a b - x y in two's complement
   (sub = 1), for m-limb a, x and c-limb b, y.  */
GTC_INLINE void
textbook_part (mp_ptr R, mp_srcptr a, mp_srcptr b, mp_srcptr x,
	       mp_srcptr y, int m, int c, int sub)
{
  mp_limb_t r0 = 0, r1 = 0, r2 = 0;

#pragma GCC unroll 64
  for (int t = 0; t < m + c - 1; t++)
    {
      int lo = t < c ? 0 : t - c + 1, hi = t < m ? t : m - 1;
#pragma GCC unroll 32
      for (int i = lo; i <= hi; i++)
	{
	  gtc_u128 z = (gtc_u128) x[i] * y[t - i];
	  GTC_ACC3 (r0, r1, r2, (gtc_u128) a[i] * b[t - i]);
	  /* -z is added as ~z = 2^128 - 1 - z; the column's 1 - 2^128 per
	     product follows */
	  GTC_ACC3 (r0, r1, r2, sub ? ~z : z);
	}
      if (sub)
	{
	  GTC_ACC3 (r0, r1, r2, hi - lo + 1);
	  r2 -= hi - lo + 1;
	}
      /* the accumulator holds the exact signed sum: shift it arithmetically */
      R[t] = r0;
      r0 = r1;
      r1 = r2;
      r2 = sub ? (mp_limb_t) ((mp_limb_signed_t) r2 >> 63) : 0;
    }
  R[m + c - 1] = r0;
  R[m + c] = r1;
}

GTC_INLINE void
gtc_textbook_n (mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
		 mp_srcptr br, mp_srcptr bi, int n)
{
  textbook_part (P, ar, br, ai, bi, n, n, 1);	/* ar br - ai bi */
  textbook_part (Q, ar, bi, ai, br, n, n, 0);	/* ar bi + ai br */
}

#ifndef GTC_TEXTBOOK_ASM
#define GTC_TB_C(n)							\
  GTC_OUTLINE void							\
  gtc_tb_c##n (mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,		\
	       mp_srcptr br, mp_srcptr bi)				\
  {									\
    gtc_textbook_n (P, Q, ar, ai, br, bi, n);				\
  }
GTC_TB_C (2) GTC_TB_C (3) GTC_TB_C (4) GTC_TB_C (5) GTC_TB_C (6)
GTC_TB_C (7) GTC_TB_C (8) GTC_TB_C (9) GTC_TB_C (10) GTC_TB_C (11)
GTC_TB_C (12) GTC_TB_C (13) GTC_TB_C (14) GTC_TB_C (15) GTC_TB_C (16)
GTC_TB_C (17) GTC_TB_C (18) GTC_TB_C (19) GTC_TB_C (20)
#undef GTC_TB_C
#endif

/* 2 <= n <= GTC_TEXTBOOK_LIMIT.  */
GTC_INLINE void
gtc_textbook (mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai, mp_srcptr br,
	       mp_srcptr bi, mp_size_t n)
{
#ifdef GTC_TEXTBOOK_ASM
  gtc_tb_fn[n] (P, Q, ar, ai, br, bi, NULL);
#else
  typedef void (*tb_fn) (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
			 mp_srcptr);
  static const tb_fn tb[] = {
    NULL, NULL, gtc_tb_c2, gtc_tb_c3, gtc_tb_c4, gtc_tb_c5, gtc_tb_c6,
    gtc_tb_c7, gtc_tb_c8, gtc_tb_c9, gtc_tb_c10, gtc_tb_c11, gtc_tb_c12,
    gtc_tb_c13, gtc_tb_c14, gtc_tb_c15, gtc_tb_c16, gtc_tb_c17, gtc_tb_c18,
    gtc_tb_c19, gtc_tb_c20
  };
  tb[n] (P, Q, ar, ai, br, bi);
#endif
}

/* Products of factors of k+1 limbs whose top limbs are small (the
   evaluations at 1, -1, i, -i): the product is below 8 B^2k, so its 2k+1
   low limbs hold it (bit counting, not limb counting), and with
   X = A + x B^k, Y = C + y B^k,
     X Y = A C + (A y + C x) B^k + x y B^2k,
   the k-limb product and a linear pass per unit of the top limbs (up to
   eight) can stand in for the k+1 limb product.  GTC_TOP_CORR: always
   (where a limb product costs much more than a limb of a pass, as on
   aarch64); otherwise only where the k+1 limb product would begin a step
   of cost (gtc_top_rule), and where every top limb is 0.  P and Q get 2k+3
   limbs, valid in their low 2k+1.  */
#if defined (__aarch64__)
#define GTC_TOP_CORR 1
#else
#define GTC_TOP_CORR 0
#endif

/* R, of k+1 limbs, += s coef V, where V has k limbs, coef is 0, 1 or 2 and
   s is +1 (neg = 0) or -1; the carry out of R's top limb is dropped.  */
GTC_INLINE void
top_acc (mp_ptr R, mp_srcptr V, mp_size_t k, mp_limb_t coef, int neg)
{
  for (; coef != 0; coef--)
    {
      mp_limb_t cy = neg ? GTC_SUB_N (R, R, V, k) : GTC_ADD_N (R, R, V, k);
      R[k] = neg ? R[k] - cy : R[k] + cy;
    }
}

/* Whether to take the k-limb product and the corrections of top limbs of
   weight w: where k+1 begins a step of the k+1 limb product's cost -- k a
   multiple of 5 under a Toom-5, of 3 under a Toom-3, of 4 otherwise -- and
   below 100 limbs at small weights.  */
GTC_INLINE int
gtc_top_rule (const gtc_ctx *c, int I, mp_size_t k, mp_limb_t w)
{
  mp_size_t n = k + 1;
  int a = n < GTC_TABLE_N ? c->alg[GTC_CAP (I)][n]
    : gtc_alg_thresh (&c->cfg, I, n);
  if (a == GTC_ALG_55)
    return k % 5 == 0 && w <= 1 + (k < 100);
  if (a == GTC_ALG_33)
    return k % 3 == 0 && w <= 1 + (k < 100);
  return ((k & 3) == 0 && w <= (k < 100 ? 4 : 1 + (k < 100)))
    || (k < 100 && (k & 3) != 3 && w == 1);
}

GTC_INLINE void
gtc_q1_top (gtc_ctx *c, int I, mp_ptr P, mp_ptr Q, mp_srcptr ar,
	    mp_srcptr ai, mp_srcptr br, mp_srcptr bi, mp_size_t k)
{
  mp_limb_t xr = ar[k], xi = ai[k], yr = br[k], yi = bi[k];

#if GTC_TOP_CORR
  if (1)
#else
  mp_limb_t w = xr + xi + yr + yi;
  if (w != 0 && gtc_top_rule (c, I, k, w))
#endif
    {
      gtc_q1 (c, I, P, Q, ar, ai, br, bi, k);
      if ((xr | xi | yr | yi) == 0)
	return;
      /* Re: ar yr - ai yi + br xr - bi xi at B^k, xr yr - xi yi at B^2k */
      top_acc (P + k, ar, k, yr, 0);
      top_acc (P + k, ai, k, yi, 1);
      top_acc (P + k, br, k, xr, 0);
      top_acc (P + k, bi, k, xi, 1);
      P[2 * k] += xr * yr - xi * yi;
      /* Im: ar yi + ai yr + br xi + bi xr at B^k, xr yi + xi yr at B^2k */
      top_acc (Q + k, ar, k, yi, 0);
      top_acc (Q + k, ai, k, yr, 0);
      top_acc (Q + k, br, k, xi, 0);
      top_acc (Q + k, bi, k, xr, 0);
      Q[2 * k] += xr * yi + xi * yr;
    }
  else
    gtc_q1 (c, I, P, Q, ar, ai, br, bi,
	     (xr | xi | yr | yi) == 0 ? k : k + 1);
}

/* The Gaussian product of the general form, (xr + i xi)(yr + i yi) with
   n-limb components in sign-magnitude form, into buf0 and buf1 (2n+1
   limbs each); *re and *im point into them.  Every Gaussian integer is a
   unit times one in the closed first quadrant, x = (Ar + i Ai) i^e: both
   factors are rotated there, multiplied, and the product rotated back by
   i^(ex+ey).  No data moves: the rotations choose which component is which
   and set the flags, by indexing small arrays rather than by branches on
   the data.  sr: the components are evaluations with small top limbs, the
   product gtc_q1_top's.  */
GTC_INLINE void
gtc_signed (gtc_ctx *c, int I, sreg *re, sreg *im, mp_ptr buf0,
	    mp_ptr buf1, mp_srcptr xr, int xr_neg, mp_srcptr xi, int xi_neg,
	    mp_srcptr yr, int yr_neg, mp_srcptr yi, int yi_neg, mp_size_t n,
	    int sr)
{
  /* e with x = (Ar + i Ai) i^e, by the signs of (Re x, Im x) */
  static const unsigned char quadrant[2][2] = { {0, 3}, {1, 2} };
  int ex = quadrant[xr_neg != 0][xi_neg != 0];
  int ey = quadrant[yr_neg != 0][yi_neg != 0];
  int e = (ex + ey) & 3;
  mp_srcptr xs[2] = { xr, xi }, ys[2] = { yr, yi };
  mp_ptr bufs[2] = { buf0, buf1 };
  mp_srcptr ar = xs[ex & 1], ai = xs[(ex & 1) ^ 1];
  mp_srcptr br = ys[ey & 1], bi = ys[(ey & 1) ^ 1];

  if (sr)
    gtc_q1_top (c, I, buf0, buf1, ar, ai, br, bi, n - 1);
  else
    gtc_q1 (c, I, buf0, buf1, ar, ai, br, bi, n);

  /* (P + iQ) i^e with P in buf0, Q in buf1:
     e = 0: (P, Q);  1: (-Q, P);  2: (-P, -Q);  3: (Q, -P).  */
  *re = (sreg) { bufs[e & 1], (0x6 >> e) & 1 };
  *im = (sreg) { bufs[(e & 1) ^ 1], (0xc >> e) & 1 };
}

/* Evaluation of a Gaussian polynomial a0 + a1 x + a2 x^2 (all parts in
   the first quadrant; a0, a1 of k limbs, a2 of h limbs) at 1, -1 and i,
   paper I, (4.6): with t = a0 + a2,
     P(1) = t + a1,  P(-1) = t - a1,  P(i) = (a0 - a2) + i a1,
   k+1 limbs each, in sign-magnitude form.  */
typedef struct
{
  mp_ptr p1r, p1i, m1r, m1i, pir, pii;
  int m1r_neg, m1i_neg, pir_neg, pii_neg;
} geval;

GTC_OUTLINE void
gauss_eval3 (geval *e, mp_srcptr ar, mp_srcptr ai, mp_size_t k, mp_size_t h)
{
  mp_srcptr a0r = ar, a1r = ar + k, a2r = ar + 2 * k;
  mp_srcptr a0i = ai, a1i = ai + k, a2i = ai + 2 * k;

#ifdef GTC_ACC_ASM
  eval3 (e->p1r, a0r, 1, a1r, 1, a2r, 1, k, h);
  eval3 (e->p1i, a0i, 1, a1i, 1, a2i, 1, k, h);
  e->m1r_neg = eval3 (e->m1r, a0r, 1, a1r, -1, a2r, 1, k, h);
  e->m1i_neg = eval3 (e->m1i, a0i, 1, a1i, -1, a2i, 1, k, h);
  /* Re P(i) = a0r - a1i - a2r,  Im P(i) = a0i + a1r - a2i */
  e->pir_neg = eval3 (e->pir, a0r, 1, a1i, -1, a2r, -1, k, h);
  e->pii_neg = eval3 (e->pii, a0i, 1, a1r, 1, a2i, -1, k, h);
#else
  e->m1r[k] = gtc_add (e->m1r, a0r, k, a2r, h);		/* t */
  e->m1i[k] = gtc_add (e->m1i, a0i, k, a2i, h);
  e->p1r[k] = e->m1r[k] + GTC_ADD_N (e->p1r, e->m1r, a1r, k);
  e->p1i[k] = e->m1i[k] + GTC_ADD_N (e->p1i, e->m1i, a1i, k);
  e->m1r_neg = abs_sub (e->m1r, e->m1r, k + 1, a1r, k);
  e->m1i_neg = abs_sub (e->m1i, e->m1i, k + 1, a1i, k);
  /* Re P(i) = a0r - (a2r + a1i),  Im P(i) = (a0i + a1r) - a2i */
  e->pir[k] = gtc_add (e->pir, a1i, k, a2r, h);
  e->pir_neg = !abs_sub (e->pir, e->pir, k + 1, a0r, k);
  e->pii[k] = GTC_ADD_N (e->pii, a0i, a1r, k);
  e->pii_neg = abs_sub (e->pii, e->pii, k + 1, a2i, h);
#endif
}

/* One component of the Gaussian Toom-3 recomposition
     C = q0 + q1 X + q2 X^2 + q3 X^3 + q4 X^4,   X = B^k,
   with the even coefficients already in place: q0 in {C, 2k+1}, q4 in
   {C + 4k, 2h+1} and q2 between them, over q0's top limb c0 and q4's low
   limb e4, both saved by the interpolation.  The result is {C, 2n+1} in
   two's complement.  (The C build's; the tests check it everywhere.)  */
GTC_OUTLINE __attribute__ ((unused)) void
recompose (mp_ptr C, mp_size_t n, mp_size_t k, sreg q1, sreg q3,
	   mp_limb_t c0, mp_limb_t e4)
{
  mp_size_t len = 2 * n + 1, w = 2 * k + 1;
  mp_limb_t t2 = C[4 * k];		/* q2's top limb, in q4's low limb */

  C[4 * k] = e4;
  acc_limb (C + 4 * k, len - 4 * k, t2);
  acc_limb (C + 2 * k, len - 2 * k, c0);
  acc (C + k, len - k, q1, w);
  acc (C + 3 * k, len - 3 * k, q3, w);
}

/* One component of the Gaussian Karatsuba recomposition
     C = v0 + (v0 + vinf - vm1) X + vinf X^2,  X = B^k,  modulo B^len,
   len = 2k + 2s + 1, shared as in GMP's toom22.  On entry {C, 2k} holds
   the low limbs of v0, whose top (sign) limb c0 has been saved, and {C +
   2k, 2s+1} holds vinf, top limb ti; vm1 has width 2k+1.  In blocks of k
   limbs, v0 = L0 + H0 X + c0 X^2 and vinf = Li + Hi X (+ ti B^(2s)), so
   the result has the blocks
     X^1: L0 + H0 + Li,  X^2: H0 + Li + Hi + c0,  X^3: Hi + c0 + (ti),
   and with T = H0 + Li shared, three k-limb additions; then the signed
   limbs and vm1.  (The C build's.)  */
GTC_OUTLINE __attribute__ ((unused)) void
kara_recompose (mp_ptr C, mp_size_t len, mp_size_t k, mp_size_t s,
		mp_limb_t c0, sreg vm1)
{
  mp_limb_t ti = C[2 * k + 2 * s], cT, c1, c2;

  cT = GTC_ADD_N (C + 2 * k, C + k, C + 2 * k, k);		/* T = H0 + Li */
  c1 = cT + GTC_ADD_N (C + k, C + 2 * k, C, k);			/* X^1 */
  c2 = cT + gtc_add (C + 2 * k, C + 2 * k, k, C + 3 * k, 2 * s - k);	/* X^2 */
  acc_limb (C + 2 * k, len - 2 * k, c1 + c0);
  /* ti is vinf's sign in vinf X, at B^(k+2s): X^3 when s = k */
  if (s == k)
    acc_limb (C + 3 * k, len - 3 * k, c2 + c0 + ti);
  else
    {
      acc_limb (C + 3 * k, len - 3 * k, c2 + c0);
      acc_limb (C + k + 2 * s, len - k - 2 * s, ti);
    }
  acc (C + k, len - k, (sreg) { vm1.p, !vm1.neg }, 2 * k + 1);	/* - vm1 X */
}

#ifdef GTC_ACC_ASM
/* The same with vm1 subtracted inside the passes over the blocks X^1 and
   X^2 (gtc_add3_n); vm1's top limb, signed, joins the limbs at X^3.  */
GTC_OUTLINE void
kara_recompose3 (mp_ptr C, mp_size_t len, mp_size_t k, mp_size_t s,
		 mp_limb_t c0, sreg vm1)
{
  mp_limb_t ti = C[2 * k + 2 * s], cT, vt;
  unsigned sub = !vm1.neg;		/* vm1 is subtracted */
  mp_size_t m = 2 * s - k, j;
  mp_limb_signed_t c1, c2;

  cT = GTC_ADD_N (C + 2 * k, C + k, C + 2 * k, k);		/* T = H0 + Li */
  /* X^1: L0 + T -+ vm1, over H0 */
  c1 = (mp_limb_signed_t) gtc_add3_n (C + k, C, C + 2 * k, vm1.p, k, sub << 1)
    - sub;
  /* X^2: T + Hi -+ vm1 X^-1, over T; Hi has 2s-k limbs */
  c2 = (mp_limb_signed_t) gtc_add3_n (C + 2 * k, C + 2 * k, C + 3 * k,
				     vm1.p + k, m, sub << 1) - sub;
  for (j = m; j < k; j++)
    {
      __int128 t = (__int128) c2 + C[2 * k + j];
      t += sub ? -(__int128) vm1.p[k + j] : (__int128) vm1.p[k + j];
      C[2 * k + j] = (mp_limb_t) t;
      c2 = (mp_limb_signed_t) (t >> GMP_NUMB_BITS);
    }
  vt = sub ? -vm1.p[2 * k] : vm1.p[2 * k];	/* signed, at X^3 */
  acc_limb_bf (C + 2 * k, len - 2 * k, cT + c1 + c0);
  if (s == k)
    acc_limb_bf (C + 3 * k, len - 3 * k, cT + c2 + c0 + ti + vt);
  else
    {
      acc_limb_bf (C + 3 * k, len - 3 * k, cT + c2 + c0 + vt);
      acc_limb_bf (C + k + 2 * s, len - k - 2 * s, ti);
    }
}

/* Both components' passes of kara_recompose in one call (fewer calls,
   which matters for small k): gtc_linear_*.S, up to half sizes of
   GTC_KARA_RECOMP2_MAX (per microarchitecture), kara_recompose3 above.  */
#if defined (__aarch64__)
#define GTC_KARA_RECOMP2_MAX ((mp_size_t) (~(mp_limb_t) 0 >> 1))
#else
#define GTC_KARA_RECOMP2_MAX 20
#endif

void gtc_kara_recomp2 (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t,
		       mp_limb_t *);

/* The signed limbs after gtc_kara_recomp2's passes.  */
GTC_INLINE void
kara_fix (mp_ptr C, mp_size_t len, mp_size_t k, mp_size_t s, mp_limb_t c0,
	  mp_limb_t ti, mp_limb_t ca, mp_limb_t cb, mp_limb_t cc,
	  mp_limb_t cd, mp_limb_t utop)
{
  mp_limb_t top = ca + cc + cd + c0 + utop;

  acc_limb_bf (C + 2 * k, len - 2 * k, ca + cb + c0);
  if (s == k)
    top += ti;
  acc_limb_bf (C + 3 * k, len - 3 * k, top);
  if (s != k)
    acc_limb_bf (C + k + 2 * s, len - k - 2 * s, ti);
}
#endif

/* Gaussian Karatsuba: GMP's mpn_toom22_mul (points 0, -1, inf) over Z[i],
   n >= 2.  With x = x0 + x1 X, X = B^k, k = ceil(n/2), x1 of s = floor(n/2)
   limbs:
     v0 = x0 y0,  vinf = x1 y1,  vm1 = (x0 - x1)(y0 - y1),
     x y = v0 + (v0 + vinf - vm1) X + vinf X^2.
   As in GMP, x0 - x1 fits in k limbs, and v0 and vinf are computed in
   their final places.  */
GTC_OUTLINE void
gtc_22 (gtc_ctx *c, mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
	      mp_srcptr br, mp_srcptr bi, mp_size_t n)
{
  mp_size_t s = n >> 1, k = n - s, w = 2 * k + 1;
  tmp_mark mk;
  mp_ptr M0 = tmp_alloc (c, &mk, 2 * w), M1 = M0 + w;
  mp_ptr dxr = P, dyr = P + k, dxi = Q, dyi = Q + k;
  mp_limb_t p0, q0;
  int sxr, sxi, syr, syi;
  sreg mre, mim;

  /* evaluation at -1, in sign-magnitude form, in the result area */
  sxr = abs_sub (dxr, ar, k, ar + k, s);
  sxi = abs_sub (dxi, ai, k, ai + k, s);
  syr = abs_sub (dyr, br, k, br + k, s);
  syi = abs_sub (dyi, bi, k, bi + k, s);

  /* vm1 in scratch; v0, then vinf in place over v0's top limbs */
  gtc_signed (c, 2, &mre, &mim, M0, M1, dxr, sxr, dxi, sxi, dyr, syr, dyi, syi, k, 0);
  gtc_q1 (c, 2, P, Q, ar, ai, br, bi, k);
  p0 = P[2 * k];
  q0 = Q[2 * k];
  gtc_q1 (c, 2, P + 2 * k, Q + 2 * k, ar + k, ai + k, br + k, bi + k, s);

#ifdef GTC_ACC_ASM
  if (k <= GTC_KARA_RECOMP2_MAX)
  {
    mp_limb_t io[19];
    mp_limb_t tp = P[2 * k + 2 * s], tq = Q[2 * k + 2 * s];
    io[0] = 2 * s - k;
    io[1] = mre.neg ? 0 : ~(mp_limb_t) 0;	/* ~0: subtract vm1 */
    io[2] = mim.neg ? 0 : ~(mp_limb_t) 0;
    gtc_kara_recomp2 (P, Q, mre.p, mim.p, k, io);
    kara_fix (P, 2 * n + 1, k, s, p0, tp, io[3], io[5], io[7], io[9],
	      mre.p[2 * k] ^ io[1]);
    kara_fix (Q, 2 * n + 1, k, s, q0, tq, io[4], io[6], io[8], io[10],
	      mim.p[2 * k] ^ io[2]);
  }
  else
#endif
  {
#ifdef GTC_ACC_ASM
    kara_recompose3 (P, 2 * n + 1, k, s, p0, mre);
    kara_recompose3 (Q, 2 * n + 1, k, s, q0, mim);
#else
    kara_recompose (P, 2 * n + 1, k, s, p0, mre);
    kara_recompose (Q, 2 * n + 1, k, s, q0, mim);
#endif
  }
  tmp_free (c, &mk);
}

/* The Gaussian Toom-Cook-3 of paper I, Sections 4.2 and 4.3: points
   (0,1), (-1,1), (1,1), (i,1), (1,0).  A level in parts -- layout,
   evaluation, products, interpolation, recomposition -- each forced
   inline, so that gtc_33 compiles as one function.  */

/* What the interpolation hands to the recomposition: the odd coefficients
   q1 and q3 of both components, and the limbs of each component that the
   write of q2 covered (q0's top limb, q4's low limb).  With the kernels,
   q1 is handed over as the sum q1 + y1, the interpolation's last step left
   to the accumulation; without them y1.p is NULL.  */
typedef struct
{
  sreg q1re, q1im, q3re, q3im;
  mp_limb_t c0re, c0im, e4re, e4im;
  sreg y1re, y1im;
} gauss_odd;

#ifdef GTC_ACC_ASM
/* The recomposition of both components by the kernels: the limbs that q2
   covered, as in recompose, then q1 + y1 into each component in one pass
   (gtc_add3_n), and q3 into both in one pass (acc2).  */
GTC_OUTLINE void
gtc_33_recompose_k (mp_ptr P, mp_ptr Q, mp_size_t n, mp_size_t k,
		       const gauss_odd *o)
{
  mp_size_t len = 2 * n + 1, w = 2 * k + 1;
  mp_limb_t tp = P[4 * k], tq = Q[4 * k];	/* q2's top limbs */

  P[4 * k] = o->e4re;
  Q[4 * k] = o->e4im;
  acc_limb_bf (P + 4 * k, len - 4 * k, tp);
  acc_limb_bf (Q + 4 * k, len - 4 * k, tq);
  acc_limb_bf (P + 2 * k, len - 2 * k, o->c0re);
  acc_limb_bf (Q + 2 * k, len - 2 * k, o->c0im);
  acc3 (P + k, len - k, o->q1re, o->y1re, w);
  acc3 (Q + k, len - k, o->q1im, o->y1im, w);
  acc2 (P + 3 * k, Q + 3 * k, len - 3 * k, o->q3re, o->q3im, w);
}
#endif

GTC_INLINE void
gtc_33_recompose (mp_ptr P, mp_ptr Q, mp_size_t n, mp_size_t k,
		     const gauss_odd *o)
{
#ifdef GTC_ACC_ASM
  gtc_33_recompose_k (P, Q, n, k, o);
#else
  recompose (P, n, k, o->q1re, o->q3re, o->c0re, o->e4re);
  recompose (Q, n, k, o->q1im, o->q3im, o->c0im, o->e4im);
#endif
}

#ifdef GTC_ACC_ASM
/* q2 = R2 - R0 - R4 in one pass, into the gap {C + 2k, 2k+1} between q0 =
   R0 = {C, 2k+1}, whose top limb c0 the caller has saved, and q4 = R4 =
   {C + 4k, 2h+1}, whose low limb the gap's top limb covers: the pass reads
   each limb of R4 before it writes over it.  All three flags are 0.  */
GTC_INLINE sreg
q2_gap (mp_ptr C, mp_srcptr R2, mp_size_t k, mp_size_t h, mp_limb_t c0)
{
  mp_ptr d = C + 2 * k;
  mp_srcptr R4 = C + 4 * k;
  mp_size_t m = h == k ? 2 * k : 2 * h + 1, j;
  mp_limb_signed_t cy;
  mp_limb_t ext;
  __int128 t;

  cy = (mp_limb_signed_t) gtc_add3_n (d, R2, C, R4, m, 3) - 2;
  /* above R4's 2h+1 limbs: its sign extension */
  ext = h == k ? 0 : -(R4[2 * h] >> (GMP_NUMB_BITS - 1));
  for (j = m; j < 2 * k; j++)
    {
      t = (__int128) cy + R2[j] - C[j] - ext;
      d[j] = (mp_limb_t) t;
      cy = (mp_limb_signed_t) (t >> GMP_NUMB_BITS);
    }
  t = (__int128) cy + R2[2 * k] - c0 - (h == k ? R4[2 * k] : ext);
  d[2 * k] = (mp_limb_t) t;
  return (sreg) { d, 0 };
}

/* The same where R2 already lies in the gap, q2 written over it: R2's low
   limb is where q0's top limb c0 belongs, and its top limb where q4's low
   limb e4 belongs, both of which the caller keeps apart.  */
GTC_INLINE sreg
q2_gap_in (mp_ptr C, mp_size_t k, mp_size_t h, mp_limb_t c0, mp_limb_t e4)
{
  mp_ptr d = C + 2 * k;
  mp_srcptr R4 = C + 4 * k;
  mp_size_t m = h == k ? 2 * k : 2 * h + 1, j;
  mp_limb_signed_t cy, c;
  mp_limb_t ext;
  __int128 t;

  t = (__int128) d[0] - C[0] - e4;
  d[0] = (mp_limb_t) t;
  c = (mp_limb_signed_t) (t >> GMP_NUMB_BITS);
  cy = (mp_limb_signed_t) gtc_add3_n (d + 1, d + 1, C + 1, R4 + 1, m - 1, 3)
    - 2;
  ext = h == k ? 0 : -(R4[2 * h] >> (GMP_NUMB_BITS - 1));
  for (j = m; j < 2 * k; j++)
    {
      t = (__int128) cy + d[j] - C[j] - ext;
      d[j] = (mp_limb_t) t;
      cy = (mp_limb_signed_t) (t >> GMP_NUMB_BITS);
    }
  t = (__int128) cy + d[2 * k] - c0 - (h == k ? R4[2 * k] : ext);
  d[2 * k] = (mp_limb_t) t;
  acc_limb_bf (d + 1, 2 * k, (mp_limb_t) c);
  return (sreg) { d, 0 };
}
#endif

/* Where a level keeps its evaluations and products.  As in GMP's Toom-3,
   the evaluations go into the result area, which holds nothing until the
   products at 0 and infinity, taken last.
   Gap mode (with the kernels, where 5k+4 <= 2n+1): the product at 1 goes
   into the gap between q0 and q4, as GMP's Toom-3 keeps its v1, and the
   scratch holds the products at -1 and i and four limbs, through which the
   products at 0 and infinity trade the limbs of R2 they cover.  The
   products come in the order i, -1, 1, 0, infinity.
   Otherwise six product buffers of 2k+3 limbs in scratch; where the result
   area is too small for the ten evaluations (5k+5 > 2n+1), they go into
   the scratch by point.  */
#define GTC_33_PW(k) (2 * (k) + 3)
#define GTC_33_INPLACE(n, k) (5 * ((k) + 1) <= 2 * (n) + 1)
#ifdef GTC_ACC_ASM
#define GTC_33_GAP 1
#else
#define GTC_33_GAP 0
#endif
#define GTC_33_GAPMODE(n, k) (GTC_33_GAP && 2 * ((n) - 2 * (k)) >= (k) + 3)
#define GTC_33_ITCH(n, k) \
  (GTC_33_GAPMODE (n, k) ? 4 * GTC_33_PW (k) + 4			\
   : 6 * GTC_33_PW (k) + (GTC_33_INPLACE (n, k) ? 0 : 4 * ((k) + 1)))

/* Places the evaluations in P, Q and the scratch ws of GTC_33_ITCH (n, k)
   limbs; returns the buffers of the product at 1.  */
GTC_INLINE mp_ptr
gtc_33_layout (mp_ptr P, mp_ptr Q, mp_ptr ws, mp_size_t n, mp_size_t k,
		  geval *ea, geval *eb)
{
  mp_size_t pw = GTC_33_PW (k), k1 = k + 1;

  if (GTC_33_GAPMODE (n, k))
    {
      ea->p1r = P; ea->m1r = P + 2 * k; eb->m1r = P + 3 * k + 1;
      eb->p1r = P + 4 * k + 3;
      ea->p1i = Q; ea->m1i = Q + 2 * k; eb->m1i = Q + 3 * k + 1;
      eb->p1i = Q + 4 * k + 3;
      ea->pir = ws; ea->pii = ws + k1; eb->pir = ws + 2 * k1;
      eb->pii = ws + 3 * k1;
      return ws;
    }
  else if (GTC_33_INPLACE (n, k))
    {
      ea->p1r = P; ea->m1r = P + k1; ea->pir = P + 2 * k1;
      eb->p1r = P + 3 * k1; eb->m1r = P + 4 * k1;
      ea->p1i = Q; ea->m1i = Q + k1; ea->pii = Q + 2 * k1;
      eb->p1i = Q + 3 * k1; eb->m1i = Q + 4 * k1;
      eb->pir = ws + 2 * pw; eb->pii = ws + 3 * pw;
      return ws;
    }
  else
    {
      mp_ptr R0 = ws, R1 = ws + 2 * pw, R2 = ws + 4 * pw;
      ea->p1r = R0; ea->p1i = R0 + k1; eb->p1r = R0 + 2 * k1; eb->p1i = R0 + 3 * k1;
      ea->m1r = R1; ea->m1i = R1 + k1; eb->m1r = R1 + 2 * k1; eb->m1i = R1 + 3 * k1;
      ea->pir = R2; ea->pii = R2 + k1; eb->pir = R2 + 2 * k1; eb->pii = R2 + 3 * k1;
      return R2 + 4 * k1;
    }
}

/* The products, each once its buffers are free: Q(0,1) = q0 and Q(1,0) =
   q4 into their places in P and Q, Q(1,1) to W (both factors in the first
   quadrant: no signs), Q(-1,1) and Q(i,1) to the signed registers r1 and
   r3.  */
GTC_INLINE void
gtc_33_products (gtc_ctx *c, mp_ptr P, mp_ptr Q, mp_srcptr ar,
		    mp_srcptr ai, mp_srcptr br, mp_srcptr bi, mp_size_t n,
		    mp_size_t k, mp_size_t h, const geval *ea,
		    const geval *eb, mp_ptr W, sreg *r1re, sreg *r1im,
		    sreg *r3re, sreg *r3im)
{
  mp_size_t pw = GTC_33_PW (k);
  mp_ptr B1, B3;

  if (GTC_33_GAPMODE (n, k))
    {
      mp_ptr sl = W + 4 * pw;
      mp_limb_t t;
      B1 = W;
      B3 = W + 2 * pw;
      gtc_signed (c, 3, r3re, r3im, B3, B3 + pw,
	    ea->pir, ea->pir_neg, ea->pii, ea->pii_neg,
	    eb->pir, eb->pir_neg, eb->pii, eb->pii_neg, k + 1, 1);
      gtc_signed (c, 3, r1re, r1im, B1, B1 + pw,
	    ea->m1r, ea->m1r_neg, ea->m1i, ea->m1i_neg,
	    eb->m1r, eb->m1r_neg, eb->m1i, eb->m1i_neg, k + 1, 1);
      gtc_q1_top (c, 3, P + 2 * k, Q + 2 * k, ea->p1r, ea->p1i, eb->p1r,
		   eb->p1i, k);
      sl[0] = P[2 * k]; sl[1] = Q[2 * k];	/* R2's low limbs */
      sl[2] = P[4 * k]; sl[3] = Q[4 * k];	/* and its top limbs */
      gtc_q1 (c, 3, P, Q, ar, ai, br, bi, k);
      t = P[2 * k]; P[2 * k] = sl[0]; sl[0] = t;	/* q0's top limbs */
      t = Q[2 * k]; Q[2 * k] = sl[1]; sl[1] = t;
      gtc_q1 (c, 3, P + 4 * k, Q + 4 * k, ar + 2 * k, ai + 2 * k,
	       br + 2 * k, bi + 2 * k, h);
      t = P[4 * k]; P[4 * k] = sl[2]; sl[2] = t;	/* q4's low limbs */
      t = Q[4 * k]; Q[4 * k] = sl[3]; sl[3] = t;
    }
  else if (GTC_33_INPLACE (n, k))
    {
      B1 = W + 2 * pw;
      B3 = W + 4 * pw;
      gtc_q1_top (c, 3, W, W + pw, ea->p1r, ea->p1i, eb->p1r, eb->p1i, k);
      gtc_signed (c, 3, r3re, r3im, B3, B3 + pw,
	    ea->pir, ea->pir_neg, ea->pii, ea->pii_neg,
	    eb->pir, eb->pir_neg, eb->pii, eb->pii_neg, k + 1, 1);
      gtc_signed (c, 3, r1re, r1im, B1, B1 + pw,
	    ea->m1r, ea->m1r_neg, ea->m1i, ea->m1i_neg,
	    eb->m1r, eb->m1r_neg, eb->m1i, eb->m1i_neg, k + 1, 1);
      gtc_q1 (c, 3, P, Q, ar, ai, br, bi, k);
      gtc_q1 (c, 3, P + 4 * k, Q + 4 * k, ar + 2 * k, ai + 2 * k,
	       br + 2 * k, bi + 2 * k, h);
    }
  else
    {
      B1 = ea->p1r;
      B3 = ea->m1r;
      gtc_q1 (c, 3, P, Q, ar, ai, br, bi, k);
      gtc_q1 (c, 3, P + 4 * k, Q + 4 * k, ar + 2 * k, ai + 2 * k,
	       br + 2 * k, bi + 2 * k, h);
      gtc_q1_top (c, 3, W, W + pw, ea->p1r, ea->p1i, eb->p1r, eb->p1i, k);
      gtc_signed (c, 3, r1re, r1im, B1, B1 + pw,
	    ea->m1r, ea->m1r_neg, ea->m1i, ea->m1i_neg,
	    eb->m1r, eb->m1r_neg, eb->m1i, eb->m1i_neg, k + 1, 1);
      gtc_signed (c, 3, r3re, r3im, B3, B3 + pw,
	    ea->pir, ea->pir_neg, ea->pii, ea->pii_neg,
	    eb->pir, eb->pir_neg, eb->pii, eb->pii_neg, k + 1, 1);
    }
}

/* The interpolation, the inversion sequence (4.4) of paper I, on the rows
   R0..R4 = Q(0,1), Q(-1,1), Q(1,1), Q(i,1), Q(1,0), with R0 and R4 in
   place in P and Q.  i(x + iy) = -y + ix only swaps components and flips a
   flag.  q0, q2 and q4 end in the result, q1 and q3 in *o.  */
GTC_INLINE void
gtc_33_interp_coeffs (mp_ptr P, mp_ptr Q, mp_size_t n, mp_size_t k,
			 mp_size_t h, mp_ptr W, sreg r1re, sreg r1im,
			 sreg r3re, sreg r3im, gauss_odd *o)
{
  mp_size_t w = 2 * k + 1;		/* every coefficient q_j fits */
  int gap = GTC_33_GAPMODE (n, k);
  sreg r0re = { P, 0 }, r0im = { Q, 0 };
  sreg r2re = { gap ? P + 2 * k : W, 0 };
  sreg r2im = { gap ? Q + 2 * k : W + GTC_33_PW (k), 0 };
  sreg r4re = { P + 4 * k, 0 }, r4im = { Q + 4 * k, 0 };
  sreg t;
  mp_limb_t c0re, c0im, e4re, e4im;

  /* R1 <- (R2 - R1)/2 = q1 + q3 */
  r1re = reg_add_half (r1re.p, r2re, r1re, -1, w);
  r1im = reg_add_half (r1im.p, r2im, r1im, -1, w);
  /* R2 <- R2 - R1 = q0 + q2 + q4 */
  reg_add2 (&r2re, &r2im, r2re.p, r2im.p, r2re, r1re, -1, r2im, r1im, -1, w);
#ifdef GTC_ACC_ASM
  /* R3 <- R3 - i R1 - R2, then R3 <- i R3 / 2, one pass per component:
     = q3 - i q2 */
  t = reg_add3_half (r3im.p, r3im, r1re, -1, r2im, -1, w);
  r3im = reg_add3_half (r3re.p, r3re, r1im, +1, r2re, -1, w);
#else
  /* R3 <- R3 - i R1 = q0 - q2 + q4 - 2i q3 */
  reg_add2 (&r3re, &r3im, r3re.p, r3im.p, r3re, r1im, +1, r3im, r1re, -1, w);
  /* R3 <- R3 - R2, then R3 <- i R3 / 2: = q3 - i q2 */
  t = reg_add_half (r3im.p, r3im, r2im, -1, w);
  r3im = reg_add_half (r3re.p, r3re, r2re, -1, w);
#endif
  r3re = (sreg) { t.p, !t.neg };
  /* R2 <- R2 - R0 - R4 = q2, into the gap between q0 and q4 in the
     result; the two limbs it covers, q0's top limb and q4's low limb, are
     saved for the recomposition.  R2's flag is 0 all through (the product
     at (1,1) has first-quadrant factors), so its limbs are its value.  */
#ifdef GTC_ACC_ASM
  if (gap)
    {
      mp_srcptr sl = W + 4 * GTC_33_PW (k);
      c0re = sl[0]; c0im = sl[1];
      e4re = sl[2]; e4im = sl[3];
      r2re = q2_gap_in (P, k, h, c0re, e4re);
      r2im = q2_gap_in (Q, k, h, c0im, e4im);
    }
  else
    {
      c0re = P[2 * k]; c0im = Q[2 * k];
      e4re = P[4 * k]; e4im = Q[4 * k];
      r2re = q2_gap (P, r2re.p, k, h, c0re);
      r2im = q2_gap (Q, r2im.p, k, h, c0im);
    }
  (void) r0re; (void) r0im; (void) r4re; (void) r4im;
#else
  (void) gap;
  reg_add2 (&r2re, &r2im, r2re.p, r2im.p, r2re, r0re, -1, r2im, r0im, -1, w);
  c0re = P[2 * k]; c0im = Q[2 * k];
  e4re = P[4 * k]; e4im = Q[4 * k];
  r2re = reg_sub_ext_gap (P + 2 * k, r2re, r4re, 2 * h + 1, w, k);
  r2im = reg_sub_ext_gap (Q + 2 * k, r2im, r4im, 2 * h + 1, w, k);
#endif
  /* R3 <- R3 + i R2 = q3 */
  reg_add2 (&r3re, &r3im, r3re.p, r3im.p, r3re, r2im, -1, r3im, r2re, +1, w);
#ifdef GTC_ACC_ASM
  /* q1 = R1 - R3, left to the recomposition */
  *o = (gauss_odd) { .q1re = r1re, .q1im = r1im,
		     .y1re = { r3re.p, !r3re.neg },
		     .y1im = { r3im.p, !r3im.neg },
		     .q3re = r3re, .q3im = r3im,
		     .c0re = c0re, .c0im = c0im,
		     .e4re = e4re, .e4im = e4im };
#else
  /* R1 <- R1 - R3 = q1 */
  r1re = reg_add (r1re.p, r1re, r3re, -1, w);
  r1im = reg_add (r1im.p, r1im, r3im, -1, w);

  *o = (gauss_odd) { r1re, r1im, r3re, r3im, c0re, c0im, e4re, e4im,
		     { NULL, 0 }, { NULL, 0 } };
#endif
}

GTC_INLINE void
gtc_33_interp (mp_ptr P, mp_ptr Q, mp_size_t n, mp_size_t k, mp_size_t h,
		  mp_ptr W, sreg r1re, sreg r1im, sreg r3re, sreg r3im)
{
  gauss_odd o;

  gtc_33_interp_coeffs (P, Q, n, k, h, W, r1re, r1im, r3re, r3im, &o);
  gtc_33_recompose (P, Q, n, k, &o);
}

/* One level of the Gaussian Toom-Cook-3, n = 3 or n >= 5.  */
GTC_OUTLINE void
gtc_33 (gtc_ctx *c, mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
	   mp_srcptr br, mp_srcptr bi, mp_size_t n)
{
  mp_size_t k = (n + 2) / 3, h = n - 2 * k;	/* parts of k, k and h limbs */
  tmp_mark mk;
  mp_ptr W;
  geval ea, eb;
  sreg r1re, r1im, r3re, r3im;
  mp_ptr ws = tmp_alloc (c, &mk, GTC_33_ITCH (n, k));

  W = gtc_33_layout (P, Q, ws, n, k, &ea, &eb);
  gauss_eval3 (&ea, ar, ai, k, h);
  gauss_eval3 (&eb, br, bi, k, h);
  gtc_33_products (c, P, Q, ar, ai, br, bi, n, k, h, &ea, &eb, W,
		      &r1re, &r1im, &r3re, &r3im);
  gtc_33_interp (P, Q, n, k, h, W, r1re, r1im, r3re, r3im);
  tmp_free (c, &mk);
}
