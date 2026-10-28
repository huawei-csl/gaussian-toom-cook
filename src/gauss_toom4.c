/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   The Gaussian Toom-Cook-4 (gtc_44): points 0, 1, -1, i, -i, 2 and
   infinity, the evaluation of paper I, (4.9), and the interpolation of
   paper II, Section 5.1.  Seven Gaussian products of n/4 limbs.  */

/* One operand's evaluations, k+1 limbs per component in sign-magnitude
   form: at 1 and 2 in the first quadrant (no signs), at -1, i and -i with
   a sign per component.  */
typedef struct
{
  mp_ptr p1r, p1i, m1r, m1i, pir, pii, mir, mii, p2r, p2i;
  int m1r_neg, m1i_neg, pir_neg, pii_neg, mir_neg, mii_neg;
} geval4;

/* {ds, n} = {u, n} + {v, n} and {dd, n} = |{u, n} - {v, n}|, a butterfly;
   returns the carry of the sum plus 2 if u < v.  The difference's operand
   order is chosen by masks: the sign is data.  */
GTC_INLINE mp_limb_t
bfly (mp_ptr ds, mp_ptr dd, mp_srcptr u, mp_srcptr v, mp_size_t n)
{
  mp_limb_t c;
  int lt;

  /* the top two limbs at once (at 1 and -1 the top limbs are carries,
     equal half the time); a loop only where both pairs are equal */
  {
    mp_limb_t u1 = u[n - 1], v1 = v[n - 1], u0 = u[n - 2], v0 = v[n - 2];
    lt = (u1 < v1) | ((u1 == v1) & (u0 < v0));
    if (((u1 ^ v1) | (u0 ^ v0)) == 0)
      {
	mp_size_t i = n - 3;
	while (i > 0 && u[i] == v[i])
	  i--;
	lt = u[i] < v[i];
      }
  }
  {
    mp_srcptr x = gtc_sel (lt, v, u), y = gtc_sel (lt, u, v);
#ifdef GTC_PAIR_ASM
    c = gtc_aors2x_n (ds, dd, x, x, y, y, n, 2) & 1;
#else
    c = GTC_ADD_N (ds, x, y, n);
    GTC_SUB_N (dd, x, y, n);
#endif
  }
  return c | (mp_limb_t) lt << 1;
}

#ifdef GTC_ACC_ASM
/* {d, n} = a + 2 (b + 2 (c + 2 d)): the value at 2 in one pass.  */
mp_limb_t gtc_addlsh123_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
			   mp_srcptr, mp_size_t);
#endif

/* {d, k+1} = a0 + 2 a1 + 4 a2 + 8 a3, a0, a1, a2 of k limbs, a3 of h <= k
   limbs: the value at 2, in one pass.  */
GTC_INLINE void
eval_p2 (mp_ptr d, mp_srcptr a0, mp_srcptr a1, mp_srcptr a2, mp_srcptr a3,
	 mp_size_t k, mp_size_t h)
{
  mp_limb_t c = 0, p1 = 0, p2 = 0, p3 = 0;
  mp_size_t i;

  for (i = 0; i < k; i++)
    {
      mp_limb_t v1 = a1[i], v2 = a2[i], v3 = i < h ? a3[i] : 0;
      gtc_u128 s = (gtc_u128) a0[i] + c
	+ ((v1 << 1) | (p1 >> (GMP_NUMB_BITS - 1)))
	+ ((v2 << 2) | (p2 >> (GMP_NUMB_BITS - 2)))
	+ ((v3 << 3) | (p3 >> (GMP_NUMB_BITS - 3)));
      d[i] = (mp_limb_t) s;
      c = (mp_limb_t) (s >> GMP_NUMB_BITS);
      p1 = v1; p2 = v2; p3 = v3;
    }
  d[k] = c + (p1 >> (GMP_NUMB_BITS - 1)) + (p2 >> (GMP_NUMB_BITS - 2))
    + (p3 >> (GMP_NUMB_BITS - 3));
}

/* {d + h, k - h + 1} = c + a0 + 2 a1 + 4 a2 over the limbs h..k-1 of the
   parts, c the limb above the kernel's {d, h}: the value at 2 where a3 is
   shorter.  */
GTC_INLINE void
p2_tail (mp_ptr d, mp_srcptr a0, mp_srcptr a1, mp_srcptr a2, mp_size_t k,
	 mp_size_t h, mp_limb_t c)
{
  for (mp_size_t i = h; i < k; i++)
    {
      gtc_u128 s = (gtc_u128) a0[i] + c + 2 * (gtc_u128) a1[i]
	+ 4 * (gtc_u128) a2[i];
      d[i] = (mp_limb_t) s;
      c = (mp_limb_t) (s >> GMP_NUMB_BITS);
    }
  d[k] = c;
}

/* GTC_TC4_E3: the values at 1 and -1 from O = a1 + a3 by two gtc_add3_n
   passes, the sign of a0 + a2 - O from the top limbs (eval4_pm1); else
   E = a0 + a2, O and a butterfly.  Per microarchitecture.  */
#if defined (GTC_ACC_ASM) && !defined (__aarch64__)
#define GTC_TC4_E3 1
#else
#define GTC_TC4_E3 0
#endif

#if GTC_TC4_E3
/* {p1, k+1} = a0 + a2 + O and {m1, k+1} = |a0 + a2 - O|, O of k+1 limbs
   (its top limb 0 or 1); returns 1 if a0 + a2 - O < 0.  With T the value of
   a0 + a2 - O from limb k-1 up, the rest lies strictly between -B^(k-1)
   and 2 B^(k-1): the value is positive for T >= 1, negative for T <= -2,
   and only for T = -1 or 0 made as it comes and negated after.  */
GTC_INLINE int
eval4_pm1 (mp_ptr p1, mp_ptr m1, mp_srcptr a0, mp_srcptr a2, mp_srcptr O,
	   mp_size_t k)
{
  __int128 t = (__int128) a0[k - 1] + a2[k - 1] - O[k - 1]
    - ((__int128) O[k] << GMP_NUMB_BITS);
  mp_limb_t cy;

  p1[k] = gtc_add3_n (p1, a0, a2, O, k, 0) + O[k];
  if (t <= -2)
    {
      m1[k] = gtc_add3_n (m1, O, a0, a2, k, 3) - 2 + O[k];
      return 1;
    }
  cy = gtc_add3_n (m1, a0, a2, O, k, 2) - 1 - O[k];
  m1[k] = cy;
  if (t >= 1 || (mp_limb_signed_t) cy >= 0)
    return 0;
  mpn_neg (m1, m1, k + 1);
  return 1;
}
#endif

/* Evaluation of a Gaussian polynomial a0 + a1 x + a2 x^2 + a3 x^3 (all
   parts in the first quadrant; a0, a1, a2 of k limbs, a3 of h limbs) at
   1, -1, i, -i and 2, paper I, (4.9).  With E = a0 + a2, O = a1 + a3,
   D0 = a0 - a2 and D1 = a1 - a3:
     P(1) = E + O,  P(-1) = E - O,  P(+-i) = D0 +- i D1,
   each pair a butterfly per component: Re P(+-i) = D0r -+ D1i and
   Im P(+-i) = D0i +- D1r take the sum and the difference of the
   magnitudes, the signs of D0 and D1 deciding which point gets which.
   P(2) by Horner's rule.  T is scratch of 6k + 2 limbs.  */
GTC_OUTLINE void
gauss_eval4 (geval4 *e, mp_srcptr ar, mp_srcptr ai, mp_size_t k,
	     mp_size_t h, mp_ptr T)
{
  mp_srcptr a0r = ar, a1r = ar + k, a2r = ar + 2 * k, a3r = ar + 3 * k;
  mp_srcptr a0i = ai, a1i = ai + k, a2i = ai + 2 * k, a3i = ai + 3 * k;
  mp_ptr E = T, O = T + k + 1, D0r = T + 2 * k + 2, D0i = D0r + k;
  mp_ptr D1r = D0i + k, D1i = D1r + k;
  int s0r, s0i, s1r, s1i, eq;
  mp_limb_t f;

  /* at 1 and -1 */
#if GTC_TC4_E3
  (void) E;
  O[k] = gtc_add (O, a1r, k, a3r, h);
  e->m1r_neg = eval4_pm1 (e->p1r, e->m1r, a0r, a2r, O, k);
  O[k] = gtc_add (O, a1i, k, a3i, h);
  e->m1i_neg = eval4_pm1 (e->p1i, e->m1i, a0i, a2i, O, k);
#else
  E[k] = GTC_ADD_N (E, a0r, a2r, k);
  O[k] = gtc_add (O, a1r, k, a3r, h);
  e->m1r_neg = bfly (e->p1r, e->m1r, E, O, k + 1) >> 1;
  E[k] = GTC_ADD_N (E, a0i, a2i, k);
  O[k] = gtc_add (O, a1i, k, a3i, h);
  e->m1i_neg = bfly (e->p1i, e->m1i, E, O, k + 1) >> 1;
#endif

  /* at i and -i */
  s0r = abs_sub (D0r, a0r, k, a2r, k);
  s0i = abs_sub (D0i, a0i, k, a2i, k);
  s1r = abs_sub (D1r, a1r, k, a3r, h);
  s1i = abs_sub (D1i, a1i, k, a3i, h);
  /* Re P(i) = D0r - D1i, Re P(-i) = D0r + D1i: with equal signs the
     magnitudes' sum goes to -i, else to i */
  {
    eq = s0r == s1i;
    mp_ptr dS = (mp_ptr) gtc_sel (eq, e->mir, e->pir);
    mp_ptr dD = (mp_ptr) gtc_sel (eq, e->pir, e->mir);
    f = bfly (dS, dD, D0r, D1i, k);
    dS[k] = f & 1;
    dD[k] = 0;
    e->pir_neg = s0r ^ (eq & (int) (f >> 1));
    e->mir_neg = s0r ^ ((!eq) & (int) (f >> 1));
  }
  /* Im P(i) = D0i + D1r, Im P(-i) = D0i - D1r */
  {
    eq = s0i == s1r;
    mp_ptr dS = (mp_ptr) gtc_sel (eq, e->pii, e->mii);
    mp_ptr dD = (mp_ptr) gtc_sel (eq, e->mii, e->pii);
    f = bfly (dS, dD, D0i, D1r, k);
    dS[k] = f & 1;
    dD[k] = 0;
    e->pii_neg = s0i ^ ((!eq) & (int) (f >> 1));
    e->mii_neg = s0i ^ (eq & (int) (f >> 1));
  }

  /* at 2 */
#ifdef GTC_ACC_ASM
  if (h < k)
    {
      p2_tail (e->p2r, a0r, a1r, a2r, k, h,
	       gtc_addlsh123_n (e->p2r, a0r, a1r, a2r, a3r, h));
      p2_tail (e->p2i, a0i, a1i, a2i, k, h,
	       gtc_addlsh123_n (e->p2i, a0i, a1i, a2i, a3i, h));
    }
  else
    {
      e->p2r[k] = gtc_addlsh123_n (e->p2r, a0r, a1r, a2r, a3r, k);
      e->p2i[k] = gtc_addlsh123_n (e->p2i, a0i, a1i, a2i, a3i, k);
    }
#else
  eval_p2 (e->p2r, a0r, a1r, a2r, a3r, k, h);
  eval_p2 (e->p2i, a0i, a1i, a2i, a3i, k, h);
#endif
}

/* Where a level keeps its evaluations and products: five regions of two
   product buffers each (2k+3 limbs a buffer, plus the pad of
   GTC_TC4_PAD); they hold the evaluations (a and b, both components, k+1
   limbs each) at 1, -1, i, -i and 2, and each product goes over the
   evaluations of the one before it: -1 over those at 1, i over those at
   -1, and so on.  The product at 1 goes into the result, between q0 and
   q6, where q2 ends; two limbs of scratch let the product at 0 trade its
   top limbs with its low limbs.  The result area holds the evaluation's
   scratch and, last, the products at 0 and infinity.  */
#define GTC_44_PW0(k) (2 * (k) + 3)
#define GTC_44_ITCH(n, k) (10 * (GTC_44_PW0 (k) + 31 * GTC_TC4_PAD) + 2)

/* The buffer width of the Toom-4's scratch for parts of k limbs.  */
GTC_INLINE mp_size_t
gtc_44_pw (const gtc_ctx *c, mp_size_t k)
{
  mp_size_t pw = GTC_44_PW0 (k);
#if GTC_TC4_PAD
  pw += c->tc4pad[pw & 511];
#else
  (void) c;
#endif
  return pw;
}

/* d = x + sy y + sz z at width w, one pass with the kernels, two without.
   dp may equal x.p.  */
GTC_INLINE sreg
reg_add3 (mp_ptr dp, sreg x, sreg y, int sy, sreg z, int sz, mp_size_t w)
{
#ifdef GTC_ACC_ASM
  int my = (sy > 0) != (x.neg == y.neg), mz = (sz > 0) != (x.neg == z.neg);
  gtc_add3_n (dp, x.p, y.p, z.p, w, my | mz << 1);
  return (sreg) { dp, x.neg };
#else
  x = reg_add (dp, x, y, sy, w);
  return reg_add (dp, x, z, sz, w);
#endif
}

#ifdef GTC_ACC_ASM
/* q5 = (R5 - 4 q3 - 8 q4)/15 in one pass: the sums and GMP's exact
   division by 15 fused.  GTC_TC4_DIV15N: -q5, the division by two carry
   chains, the flag flipped after.  Per microarchitecture.  */
mp_limb_t gtc_div15_lsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
			   mp_size_t, unsigned);
#if defined (GTC_ZEN)
#define GTC_TC4_DIV15N 1
mp_limb_t gtc_div15n_lsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
			    mp_size_t, unsigned);
#define GTC_DIV15_LSH_N gtc_div15n_lsh_n
#else
#define GTC_TC4_DIV15N 0
#define GTC_DIV15_LSH_N gtc_div15_lsh_n
#endif
#endif

/* x +- c u by gtc_aorsmul_1 on x86-64 (one call site, the Toom-4 style
   reduction's), by GMP's mpn_addmul_1 or mpn_submul_1 elsewhere.  */
#ifdef GTC_ASM_X86
mp_limb_t gtc_aorsmul_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, unsigned);
#define GTC_ADDMUL_1(r, u, n, c) gtc_aorsmul_1 (r, u, n, c, 0)
#define GTC_SUBMUL_1(r, u, n, c) gtc_aorsmul_1 (r, u, n, c, 1)
#else
#define GTC_ADDMUL_1 mpn_addmul_1
#define GTC_SUBMUL_1 mpn_submul_1
#endif

/* x += s c y at width w in place, c a small positive limb, y of width
   yw <= w sign-extended (mod B^w): GMP's mpn_addmul_1 or mpn_submul_1 over
   y's limbs, the high limb and y's sign extension times c added above.  */
GTC_INLINE sreg
reg_aorsmul (sreg x, sreg y, mp_limb_t c, int s, mp_size_t yw, mp_size_t w)
{
  mp_limb_t e = c & -(y.p[yw - 1] >> (GMP_NUMB_BITS - 1)), hi;

  if ((s > 0) == (x.neg == y.neg))
    {
      hi = mpn_addmul_1 (x.p, y.p, yw, c);
      if (yw < w)
	acc_limb (x.p + yw, w - yw, hi - e);
    }
  else
    {
      hi = mpn_submul_1 (x.p, y.p, yw, c);
      if (yw < w)
	acc_limb (x.p + yw, w - yw, e - hi);
    }
  return x;
}

/* GTC_TC4_NEG5: R5 - R1 computed as R1 - R5 with R5's flag flipped, so
   that the division by 15 adds 4 q3 and 8 q4, the kernel's cheaper form;
   per microarchitecture.  */
#if defined (GTC_ASM_ARM)
#define GTC_TC4_NEG5 0
#else
#define GTC_TC4_NEG5 1
#endif

/* x +- 2^s y on x86-64: gtc_aorslsh1_n (s = 1) and gtc_aorslsh_n, one
   carry chain each, where they beat GMP's mpn_addlsh1_n, mpn_sublsh1_n,
   mpn_rsblsh1_n, mpn_rsblsh2_n and mpn_addlsh2_n (per microarchitecture:
   GTC_TC4_SLK for the first two, GTC_LSHK_ADD2 for the last).  */
#if defined (GTC_ASM_X86) && !defined (__AVX512F__)
#define GTC_TC4_SLK 1
#define GTC_LSHK_ADD2 1
#else
#define GTC_TC4_SLK 0
#define GTC_LSHK_ADD2 0
#endif
#if GTC_TC4_SLK
mp_limb_t gtc_aorslsh1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t,
			  unsigned);
#endif
#ifdef GTC_ASM_X86
mp_limb_t gtc_aorslsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned,
			 unsigned);
#endif
#if GTC_TC4_SLK
#define GTC_ADDLSH1_N(r, u, v, n) gtc_aorslsh1_n (r, u, v, n, 0)
#define GTC_SUBLSH1_N(r, u, v, n) gtc_aorslsh1_n (r, u, v, n, 1)
#else
#define GTC_ADDLSH1_N mpn_addlsh1_n
#define GTC_SUBLSH1_N mpn_sublsh1_n
#endif
#ifdef GTC_ASM_X86
#define GTC_RSBLSH1_N(r, u, v, n) \
  ((mp_limb_signed_t) gtc_aorslsh_n (r, u, v, n, 1, 2))
#define GTC_RSBLSH2_N(r, u, v, n) \
  ((mp_limb_signed_t) gtc_aorslsh_n (r, u, v, n, 2, 2))
#else
#define GTC_RSBLSH1_N mpn_rsblsh1_n
#define GTC_RSBLSH2_N mpn_rsblsh2_n
#endif
#if GTC_LSHK_ADD2
#define GTC_ADDLSH2_N(r, u, v, n) gtc_aorslsh_n (r, u, v, n, 2, 0)
#else
#define GTC_ADDLSH2_N mpn_addlsh2_n
#endif

/* x += s 2 y at width w in place, y of width yw <= w sign-extended (mod
   B^w): the shifted sum over y's limbs, then what is left at limb yw, the
   carry or borrow and twice y's sign, as one signed limb.  */
GTC_INLINE sreg
reg_aorslsh1 (sreg x, sreg y, int s, mp_size_t yw, mp_size_t w)
{
  mp_limb_t e = 2 * (y.p[yw - 1] >> (GMP_NUMB_BITS - 1)), cy;

  if ((s > 0) == (x.neg == y.neg))
    {
      cy = GTC_ADDLSH1_N (x.p, x.p, y.p, yw);
      if (yw < w)
	acc_limb (x.p + yw, w - yw, cy - e);
    }
  else
    {
      cy = GTC_SUBLSH1_N (x.p, x.p, y.p, yw);
      if (yw < w)
	acc_limb (x.p + yw, w - yw, e - cy);
    }
  return x;
}

/* Places the evaluations in the scratch ws of GTC_44_ITCH (n, k) limbs:
   region j = 0..4 holds those at 1, -1, i, -i and 2.  */
GTC_INLINE void
gtc_44_layout (mp_ptr ws, mp_size_t k, mp_size_t pw, geval4 *ea,
		  geval4 *eb)
{
  mp_size_t k1 = k + 1, rw = 2 * pw;
  mp_ptr G = ws;

  ea->p1r = G; ea->p1i = G + k1; eb->p1r = G + 2 * k1; eb->p1i = G + 3 * k1;
  G += rw;
  ea->m1r = G; ea->m1i = G + k1; eb->m1r = G + 2 * k1; eb->m1i = G + 3 * k1;
  G += rw;
  ea->pir = G; ea->pii = G + k1; eb->pir = G + 2 * k1; eb->pii = G + 3 * k1;
  G += rw;
  ea->mir = G; ea->mii = G + k1; eb->mir = G + 2 * k1; eb->mii = G + 3 * k1;
  G += rw;
  ea->p2r = G; ea->p2i = G + k1; eb->p2r = G + 2 * k1; eb->p2i = G + 3 * k1;
}

/* The rows of the interpolation, Q at 1, -1, i, -i and 2 (paper II's R1
   to R5), real and imaginary parts, and what the result area's limbs
   trade with them.  */
typedef struct
{
  sreg re[6], im[6];
  mp_limb_t c0[2];		/* q0's top limbs */
  mp_ptr spare[2];		/* two free buffers */
  mp_limb_t q2top[2], q4top[2];
} gauss_rows4;

/* The products: each into the region of the point before it, the one at 1
   into the result; then q0 and q6 into the result.  */
GTC_INLINE void
gtc_44_products (gtc_ctx *c, mp_ptr P, mp_ptr Q, mp_srcptr ar,
		    mp_srcptr ai, mp_srcptr br, mp_srcptr bi, mp_size_t k,
		    mp_size_t h, mp_size_t pw, const geval4 *ea,
		    const geval4 *eb, mp_ptr ws, gauss_rows4 *r)
{
  mp_size_t rw = 2 * pw;
  mp_ptr G0 = ws, G1 = ws + rw, G2 = ws + 2 * rw, G3 = ws + 3 * rw;
  mp_ptr G5 = P + 2 * k, H5 = Q + 2 * k;

  gtc_q1 (c, 4, G5, H5, ea->p1r, ea->p1i, eb->p1r, eb->p1i, k + 1);
  r->re[1] = (sreg) { G5, 0 };
  r->im[1] = (sreg) { H5, 0 };
  gtc_signed (c, 4, &r->re[2], &r->im[2], G0, G0 + pw,
	ea->m1r, ea->m1r_neg, ea->m1i, ea->m1i_neg,
	eb->m1r, eb->m1r_neg, eb->m1i, eb->m1i_neg, k + 1, 1);
  gtc_signed (c, 4, &r->re[3], &r->im[3], G1, G1 + pw,
	ea->pir, ea->pir_neg, ea->pii, ea->pii_neg,
	eb->pir, eb->pir_neg, eb->pii, eb->pii_neg, k + 1, 1);
  gtc_signed (c, 4, &r->re[4], &r->im[4], G2, G2 + pw,
	ea->mir, ea->mir_neg, ea->mii, ea->mii_neg,
	eb->mir, eb->mir_neg, eb->mii, eb->mii_neg, k + 1, 1);
  gtc_q1 (c, 4, G3, G3 + pw, ea->p2r, ea->p2i, eb->p2r, eb->p2i, k + 1);
  r->re[5] = (sreg) { G3, 0 };
  r->im[5] = (sreg) { G3 + pw, 0 };
  r->spare[0] = ws + 4 * rw;	/* the evaluations at 2, consumed */
  r->spare[1] = ws + 4 * rw + pw;
  {
    /* q0's top limbs go where the product at 1 has its low limbs */
    mp_limb_t l1p = P[2 * k], l1q = Q[2 * k];
    gtc_q1 (c, 4, P, Q, ar, ai, br, bi, k);
    r->c0[0] = P[2 * k];
    r->c0[1] = Q[2 * k];
    P[2 * k] = l1p;
    Q[2 * k] = l1q;
  }
  gtc_q1 (c, 4, P + 6 * k, Q + 6 * k, ar + 3 * k, ai + 3 * k, br + 3 * k,
	   bi + 3 * k, h);
}

/* GTC_TC4_ORDER: from k = GTC_TC4_ORDER_K on, where a level outgrows L1,
   the interpolation's passes component by component (tc4_interp_comp),
   each pass reading what the one before it wrote; else alternating
   between the components.  Per microarchitecture.  */
#ifdef GTC_PAIR_ASM
#define GTC_TC4_ORDER 0
#else
#define GTC_TC4_ORDER 1
#endif
#define GTC_TC4_ORDER_K 112

#if GTC_TC4_ORDER
/* One component's interpolation from R2's halving on: x its rows, P its
   part of the result, c its index (0 real, 1 imaginary).  */
GTC_INLINE void
tc4_interp_comp (mp_ptr P, sreg *x, mp_size_t k, mp_size_t h,
		 gauss_rows4 *r, int c, mp_limb_t *c0)
{
  mp_size_t w = 2 * k + 1, w6 = 2 * h + 1;
  sreg r0 = { P, 0 }, r6 = { P + 6 * k, 0 };
  mp_limb_t l6, e4;
  mp_limb_t t;

  x[2] = reg_add_half (x[2].p, x[1], x[2], -1, w);
  x[2] = reg_add_half (x[2].p, x[2], x[4], -1, w);
  x[1] = reg_add3 (x[1].p, x[1], x[2], -1, x[3], -1, w);
  t = P[2 * k]; P[2 * k] = r->c0[c]; r->c0[c] = t;
  x[5] = reg_add_half (x[5].p, x[5], r0, -1, w);
  t = P[2 * k]; P[2 * k] = r->c0[c]; r->c0[c] = t;
#if GTC_TC4_NEG5
  x[5] = reg_add (x[5].p, x[1], x[5], -1, w);
  x[5].neg ^= 1;
#else
  x[5] = reg_add (x[5].p, x[5], x[1], -1, w);
#endif
  x[4] = reg_add (x[4].p, x[4], x[2], 1, w);
  x[1] = reg_add_half (r->spare[c], x[1], x[4], -1, w);
  P[2 * k] = r->c0[c];
  l6 = P[6 * k];
  x[3] = reg_add3 (P + 4 * k, x[1], r0, -1, x[3], 1, w);
#ifdef GTC_ACC_ASM
  GTC_DIV15_LSH_N (x[5].p, x[5].p, x[2].p, x[3].p, w,
		   (x[5].neg == x[2].neg) | (x[5].neg == x[3].neg) << 1);
  x[5].neg ^= GTC_TC4_DIV15N;
#else
  x[5] = reg_aorsmul (x[5], x[2], 4, -1, w, w);
  x[5] = reg_aorsmul (x[5], x[3], 8, -1, w, w);
  mpn_bdiv_dbm1c (x[5].p, x[5].p, w, GMP_NUMB_MAX / 15, 0);
#endif
  r->q4top[c] = P[6 * k];
  P[6 * k] = l6;
  e4 = P[4 * k];
  x[5] = reg_aorslsh1 (x[5], r6, -1, w6, w);
  c0[c] = r->c0[c];
  x[1] = reg_add_ext (P + 2 * k, x[1], r6, w6, -1, w);
  r->q2top[c] = P[4 * k];
  P[4 * k] = e4;
}
#endif

/* The interpolation (paper II, Section 5.1) of one product's rows, with q0
   in {C, 2k+1} and q6 in {C + 6k, 2h+1} for both components C = P, Q.
   The halvings are fused into the subtractions before them; i(x + iy) =
   -y + ix only swaps components and flips a flag.  q2 and q4 are written
   into the result, their top limbs saved in *r and q0's in *c0; the rows
   keep q3 (R2), q5 (R5) and q1 + q5 (R4), q1 left to the recomposition.  */
GTC_INLINE void
gtc_44_interp_coeffs (mp_ptr P, mp_ptr Q, mp_size_t k, mp_size_t h,
			 gauss_rows4 *r, mp_limb_t *c0)
{
  mp_size_t w = 2 * k + 1, w6 = 2 * h + 1;
  sreg r0re = { P, 0 }, r0im = { Q, 0 };
  sreg r6re = { P + 6 * k, 0 }, r6im = { Q + 6 * k, 0 };
  sreg *re = r->re, *im = r->im, tre, tim;
  mp_limb_t l6p, l6q, e4p, e4q;

#if GTC_TC4_ORDER
  if (k >= GTC_TC4_ORDER_K)
  {
  tre = reg_add_half (re[4].p, re[3], re[4], -1, w);
  tim = reg_add_half (im[4].p, im[3], im[4], -1, w);
  re[4] = tim;
  im[4] = (sreg) { tre.p, !tre.neg };
  re[3] = reg_add (re[3].p, re[3], tre, -1, w);
  im[3] = reg_add (im[3].p, im[3], tim, -1, w);
  tc4_interp_comp (P, re, k, h, r, 0, c0);
  tc4_interp_comp (Q, im, k, h, r, 1, c0);
  return;
  }
#endif
  /* R2 <- (R1 - R2)/2 = q1 + q3 + q5 */
  re[2] = reg_add_half (re[2].p, re[1], re[2], -1, w);
  im[2] = reg_add_half (im[2].p, im[1], im[2], -1, w);
  /* T <- (R3 - R4)/2 = i (q1 - q3 + q5), R4 <- T/i */
  tre = reg_add_half (re[4].p, re[3], re[4], -1, w);
  tim = reg_add_half (im[4].p, im[3], im[4], -1, w);
  re[4] = tim;
  im[4] = (sreg) { tre.p, !tre.neg };
  /* R3 <- R3 - T = q0 - q2 + q4 - q6 */
  reg_add2 (&re[3], &im[3], re[3].p, im[3].p, re[3], tre, -1, im[3], tim,
	    -1, w);
  /* R2 <- (R2 - R4)/2 = q3 */
  re[2] = reg_add_half (re[2].p, re[2], re[4], -1, w);
  im[2] = reg_add_half (im[2].p, im[2], im[4], -1, w);
  /* R1 <- R1 - R2 - R3 = q1 + 2 q2 + q5 + 2 q6 */
  re[1] = reg_add3 (re[1].p, re[1], re[2], -1, re[3], -1, w);
  im[1] = reg_add3 (im[1].p, im[1], im[2], -1, im[3], -1, w);
  /* R0's top limbs where R1 has its low limbs, for the halving */
  {
    mp_limb_t t;
    t = P[2 * k]; P[2 * k] = r->c0[0]; r->c0[0] = t;
    t = Q[2 * k]; Q[2 * k] = r->c0[1]; r->c0[1] = t;
  }
  /* R5 <- (R5 - R0)/2 - R1 = 4 q3 + 8 q4 + 15 q5 + 30 q6 */
  re[5] = reg_add_half (re[5].p, re[5], r0re, -1, w);
  im[5] = reg_add_half (im[5].p, im[5], r0im, -1, w);
  {
    mp_limb_t t;
    t = P[2 * k]; P[2 * k] = r->c0[0]; r->c0[0] = t;
    t = Q[2 * k]; Q[2 * k] = r->c0[1]; r->c0[1] = t;
  }
#if GTC_TC4_NEG5
  reg_add2 (&re[5], &im[5], re[5].p, im[5].p, re[1], re[5], -1, im[1],
	    im[5], -1, w);
  re[5].neg ^= 1;
  im[5].neg ^= 1;
#else
  reg_add2 (&re[5], &im[5], re[5].p, im[5].p, re[5], re[1], -1, im[5],
	    im[1], -1, w);
#endif
  /* R4 <- R4 + R2 = q1 + q5 */
  reg_add2 (&re[4], &im[4], re[4].p, im[4].p, re[4], re[2], 1, im[4], im[2],
	    1, w);
  /* R1 <- (R1 - R4)/2 = q2 + q6, into a spare buffer, which gives R0 its
     top limbs back */
  re[1] = reg_add_half (r->spare[0], re[1], re[4], -1, w);
  im[1] = reg_add_half (r->spare[1], im[1], im[4], -1, w);
  P[2 * k] = r->c0[0];
  Q[2 * k] = r->c0[1];
  /* R3 <- R1 - R0 + R3 = q4, into the result at 4k, over limbs that hold
     nothing by now and R6's low limb, which is saved.  R1's flag is 0, so
     the limbs are q4's own.  */
  l6p = P[6 * k];
  l6q = Q[6 * k];
  re[3] = reg_add3 (P + 4 * k, re[1], r0re, -1, re[3], 1, w);
  im[3] = reg_add3 (Q + 4 * k, im[1], r0im, -1, im[3], 1, w);
  /* R5 <- (R5 - 4 q3 - 8 q4)/15 = q5 + 2 q6: the division is exact, and
     GMP's mpn_bdiv_dbm1c with (B-1)/15 gives the exact quotient modulo
     B^w, right on two's complement */
#ifdef GTC_ACC_ASM
  /* the kernel subtracts 4 y (8 z) where the flags agree */
  GTC_DIV15_LSH_N (re[5].p, re[5].p, re[2].p, re[3].p, w,
		   (re[5].neg == re[2].neg)
		   | (re[5].neg == re[3].neg) << 1);
  GTC_DIV15_LSH_N (im[5].p, im[5].p, im[2].p, im[3].p, w,
		   (im[5].neg == im[2].neg)
		   | (im[5].neg == im[3].neg) << 1);
  re[5].neg ^= GTC_TC4_DIV15N;
  im[5].neg ^= GTC_TC4_DIV15N;
#else
  re[5] = reg_aorsmul (re[5], re[2], 4, -1, w, w);
  im[5] = reg_aorsmul (im[5], im[2], 4, -1, w, w);
  re[5] = reg_aorsmul (re[5], re[3], 8, -1, w, w);
  im[5] = reg_aorsmul (im[5], im[3], 8, -1, w, w);
  mpn_bdiv_dbm1c (re[5].p, re[5].p, w, GMP_NUMB_MAX / 15, 0);
  mpn_bdiv_dbm1c (im[5].p, im[5].p, w, GMP_NUMB_MAX / 15, 0);
#endif
  /* q4's top limb out, R6's low limb back */
  r->q4top[0] = P[6 * k];
  r->q4top[1] = Q[6 * k];
  P[6 * k] = l6p;
  Q[6 * k] = l6q;
  e4p = P[4 * k];
  e4q = Q[4 * k];
  /* R5 <- R5 - 2 R6 = q5 */
  re[5] = reg_aorslsh1 (re[5], r6re, -1, w6, w);
  im[5] = reg_aorslsh1 (im[5], r6im, -1, w6, w);
  /* R1 <- R1 - R6 = q2, into the result */
  c0[0] = r->c0[0];
  c0[1] = r->c0[1];
  re[1] = reg_add_ext (P + 2 * k, re[1], r6re, w6, -1, w);
  im[1] = reg_add_ext (Q + 2 * k, im[1], r6im, w6, -1, w);
  /* q2's top limb out, q4's low limb back */
  r->q2top[0] = P[4 * k];
  r->q2top[1] = Q[4 * k];
  P[4 * k] = e4p;
  Q[4 * k] = e4q;
}

/* The recomposition, C = sum q_j X^j modulo B^(2n+1), with q0, q2, q4 and
   q6 in place: their top limbs, which the next one's low limbs took, go
   back in as signed limbs; then q1 = (q1 + q5) - q5, q3 and q5.  */
GTC_INLINE void
gtc_44_recompose (mp_ptr P, mp_ptr Q, mp_size_t n, mp_size_t k,
		     const gauss_rows4 *r, const mp_limb_t *c0)
{
  mp_size_t len = 2 * n + 1, w = 2 * k + 1;
  const sreg *re = r->re, *im = r->im;

  acc_limb (P + 6 * k, len - 6 * k, r->q4top[0]);
  acc_limb (Q + 6 * k, len - 6 * k, r->q4top[1]);
  acc_limb (P + 4 * k, len - 4 * k, r->q2top[0]);
  acc_limb (Q + 4 * k, len - 4 * k, r->q2top[1]);
  acc_limb (P + 2 * k, len - 2 * k, c0[0]);
  acc_limb (Q + 2 * k, len - 2 * k, c0[1]);
#ifdef GTC_ACC_ASM
  acc3 (P + k, len - k, re[4], (sreg) { re[5].p, !re[5].neg }, w);
  acc3 (Q + k, len - k, im[4], (sreg) { im[5].p, !im[5].neg }, w);
  acc2 (P + 3 * k, Q + 3 * k, len - 3 * k, re[2], im[2], w);
  acc2 (P + 5 * k, Q + 5 * k, len - 5 * k, re[5], im[5], w);
#else
  acc (P + k, len - k, re[4], w);
  acc (Q + k, len - k, im[4], w);
  acc (P + k, len - k, (sreg) { re[5].p, !re[5].neg }, w);
  acc (Q + k, len - k, (sreg) { im[5].p, !im[5].neg }, w);
  acc (P + 3 * k, len - 3 * k, re[2], w);
  acc (Q + 3 * k, len - 3 * k, im[2], w);
  acc (P + 5 * k, len - 5 * k, re[5], w);
  acc (Q + 5 * k, len - 5 * k, im[5], w);
#endif
}

/* One level of the Gaussian Toom-Cook-4, n >= 13.  */
GTC_OUTLINE void
gtc_44 (gtc_ctx *c, mp_ptr P, mp_ptr Q, mp_srcptr ar, mp_srcptr ai,
	   mp_srcptr br, mp_srcptr bi, mp_size_t n)
{
  mp_size_t k = (n + 3) / 4, h = n - 3 * k;	/* parts of k, k, k and h */
  mp_size_t pw = gtc_44_pw (c, k);
  tmp_mark mk;
  geval4 ea, eb;
  gauss_rows4 r;
  mp_limb_t c0[2];
  mp_ptr ws = tmp_alloc (c, &mk, GTC_44_ITCH (n, k));

  gtc_44_layout (ws, k, pw, &ea, &eb);
  gauss_eval4 (&ea, ar, ai, k, h, P);
  gauss_eval4 (&eb, br, bi, k, h, P);
  gtc_44_products (c, P, Q, ar, ai, br, bi, k, h, pw, &ea, &eb, ws, &r);
  gtc_44_interp_coeffs (P, Q, k, h, &r, c0);
  gtc_44_recompose (P, Q, n, k, &r, c0);
  tmp_free (c, &mk);
}
