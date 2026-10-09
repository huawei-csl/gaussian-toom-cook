/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   libgtc: Gaussian Toom-Cook, and rational multiplication through
   Gaussian integers (see gtc.h).  One translation unit: the parts below
   are included, so that the compiler sees every call.

   Notation: B = 2^64 is the limb base, X = B^k the splitting point;
   numbers are little-endian limb vectors, as in GMP's mpn layer.  */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gtc.h"
#include "gmp_decls.h"
#ifdef __x86_64__
#include <x86intrin.h>
#endif

#define HIGHBIT (GMP_NUMB_MAX ^ (GMP_NUMB_MAX >> 1))

/* The small per-node helpers are forced inline, so that a node compiles to
   straight-line code around its calls; the larger steps stay out of line,
   which keeps the hot code small.  */
#define GTC_INLINE static inline __attribute__ ((always_inline))
#define GTC_OUTLINE static __attribute__ ((noinline))

#include "config.h"
#include "ctx.c"
#include "linear.c"
#include "gauss.c"
#include "gauss_toom4.c"
#include "gauss_toom5.c"
#include "rational.c"

/* The public interface (gtc.h).  */

void
r2gtc_mul_n_ctx (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
		 mp_size_t n)
{
  r2gtc (c, 5, rp, ap, bp, n);
  arena_fit (c);
}

/* One level of r2gtc_ii at the top where n allows it, else r2gtc_ii's
   products' choice.  */
void
r2gtc_22_ctx (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  if (n >= 2)
    r2gtc_22 (c, rp, ap, bp, n);
  else
    r2gtc (c, 2, rp, ap, bp, n);
  arena_fit (c);
}

void
r2gtc_33_ctx (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  if (n == 3 || n >= 5)
    r2gtc_33 (c, rp, ap, bp, n);
  else
    r2gtc (c, 3, rp, ap, bp, n);
  arena_fit (c);
}

void
r2gtc_44_ctx (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  if (n >= GTC_44_MIN)
    r2gtc_44 (c, rp, ap, bp, n);
  else
    r2gtc (c, 4, rp, ap, bp, n);
  arena_fit (c);
}

void
r2gtc_55_ctx (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  if (n >= GTC_55_MIN)
    r2gtc_55 (c, rp, ap, bp, n);
  else
    r2gtc (c, 5, rp, ap, bp, n);
  arena_fit (c);
}

void
r2gtc_mul_ctx (gtc_ctx *c, mp_ptr rp, mp_srcptr ap, mp_size_t an,
	       mp_srcptr bp, mp_size_t bn)
{
  mul_any (c, rp, ap, an, bp, bn);
  arena_fit (c);
}

void
r2gtc_mpz_mul_ctx (gtc_ctx *c, mpz_ptr r, mpz_srcptr a, mpz_srcptr b)
{
  mp_size_t an = mpz_size (a), bn = mpz_size (b), rn;
  int neg = (mpz_sgn (a) < 0) != (mpz_sgn (b) < 0);
  int alias = r == a || r == b;
  mpz_t t;
  mp_ptr tp;

  if (an == 0 || bn == 0)
    {
      mpz_set_ui (r, 0);
      return;
    }
  if (an < bn)
    {
      mpz_srcptr s = a;
      a = b;
      b = s;
      rn = an;
      an = bn;
      bn = rn;
    }
  /* Into r itself unless it is an operand, whose limbs the product
     would overwrite: then into a temporary, as GMP's mpz_mul does.  */
  if (!alias)
    {
      tp = mpz_limbs_write (r, an + bn);
      r2gtc_mul_ctx (c, tp, mpz_limbs_read (a), an, mpz_limbs_read (b),
		     bn);
      rn = an + bn - (tp[an + bn - 1] == 0);
      mpz_limbs_finish (r, neg ? -rn : rn);
      return;
    }
  mpz_init (t);
  tp = mpz_limbs_write (t, an + bn);
  r2gtc_mul_ctx (c, tp, mpz_limbs_read (a), an, mpz_limbs_read (b), bn);
  rn = an + bn - (tp[an + bn - 1] == 0);
  mpz_limbs_finish (t, neg ? -rn : rn);
  mpz_swap (r, t);
  mpz_clear (t);
}

static _Thread_local gtc_ctx *default_ctx;

static gtc_ctx *
get_default_ctx (void)
{
  if (default_ctx == NULL)
    default_ctx = gtc_ctx_new (NULL);
  return default_ctx;
}

void
r2gtc_mul_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  r2gtc_mul_n_ctx (get_default_ctx (), rp, ap, bp, n);
}

void
r2gtc_mul (mp_ptr rp, mp_srcptr ap, mp_size_t an, mp_srcptr bp,
	   mp_size_t bn)
{
  r2gtc_mul_ctx (get_default_ctx (), rp, ap, an, bp, bn);
}

void
r2gtc_mpz_mul (mpz_ptr r, mpz_srcptr a, mpz_srcptr b)
{
  r2gtc_mpz_mul_ctx (get_default_ctx (), r, a, b);
}

/* Two's complement with a negation flag to sign-magnitude.  */
static void
to_sign_magnitude (mp_ptr z, int *z_neg, sreg v, mp_size_t w)
{
  int tc_neg = (v.p[w - 1] & HIGHBIT) != 0;
  if (tc_neg)
    mpn_neg (z, v.p, w);
  else
    mpn_copyi (z, v.p, w);
  *z_neg = (tc_neg != v.neg) && !mpn_zero_p (z, w);
}

/* The Gaussian product of the public functions: as gtc_signed, the
   factors rotated into the first quadrant and the product back, with
   one level of gtc_ii at the top where top = i and n allows it, else
   the algorithm of gtc_ii's products (gtc: top = 0, I = 5).  */
static void
gtc_public (gtc_ctx *c, int top, int I,
	    mp_ptr zr, int *zr_neg, mp_ptr zi, int *zi_neg,
	    mp_srcptr xr, int xr_neg, mp_srcptr xi, int xi_neg,
	    mp_srcptr yr, int yr_neg, mp_srcptr yi, int yi_neg, mp_size_t n)
{
  static const unsigned char quadrant[2][2] = { {0, 3}, {1, 2} };
  int ex = quadrant[xr_neg != 0][xi_neg != 0];
  int ey = quadrant[yr_neg != 0][yi_neg != 0];
  int e = (ex + ey) & 3;
  mp_srcptr xs[2] = { xr, xi }, ys[2] = { yr, yi };
  mp_srcptr ar = xs[ex & 1], ai = xs[(ex & 1) ^ 1];
  mp_srcptr br = ys[ey & 1], bi = ys[(ey & 1) ^ 1];
  tmp_mark mk;
  mp_ptr buf = tmp_alloc (c, &mk, 2 * (2 * n + 1));
  mp_ptr bufs[2] = { buf, buf + 2 * n + 1 };
  sreg re, im;

  if (top == 2 && n >= 2)
    gtc_22 (c, bufs[0], bufs[1], ar, ai, br, bi, n);
  else if (top == 3 && (n == 3 || n >= 5))
    gtc_33 (c, bufs[0], bufs[1], ar, ai, br, bi, n);
  else if (top == 4 && n >= GTC_44_MIN)
    gtc_44 (c, bufs[0], bufs[1], ar, ai, br, bi, n);
  else if (top == 5 && n >= GTC_55_MIN)
    gtc_55 (c, bufs[0], bufs[1], ar, ai, br, bi, n);
  else
    gtc_q1 (c, I, bufs[0], bufs[1], ar, ai, br, bi, n);
  /* (P + iQ) i^e, as gtc_signed */
  re = (sreg) { bufs[e & 1], (0x6 >> e) & 1 };
  im = (sreg) { bufs[(e & 1) ^ 1], (0xc >> e) & 1 };
  to_sign_magnitude (zr, zr_neg, re, 2 * n + 1);
  to_sign_magnitude (zi, zi_neg, im, 2 * n + 1);
  tmp_free (c, &mk);
  arena_fit (c);
}

#define GTC_PUBLIC(name, top, I)					\
  void									\
  name (gtc_ctx *c, mp_ptr zr, int *zr_neg, mp_ptr zi, int *zi_neg,	\
	mp_srcptr xr, int xr_neg, mp_srcptr xi, int xi_neg,		\
	mp_srcptr yr, int yr_neg, mp_srcptr yi, int yi_neg, mp_size_t n)	\
  {									\
    gtc_public (c, top, I, zr, zr_neg, zi, zi_neg, xr, xr_neg, xi,	\
		xi_neg, yr, yr_neg, yi, yi_neg, n);			\
  }
GTC_PUBLIC (gtc_mul_n_ctx, 0, 5)
GTC_PUBLIC (gtc_22_ctx, 2, 2)
GTC_PUBLIC (gtc_33_ctx, 3, 3)
GTC_PUBLIC (gtc_44_ctx, 4, 4)
GTC_PUBLIC (gtc_55_ctx, 5, 5)
#undef GTC_PUBLIC
