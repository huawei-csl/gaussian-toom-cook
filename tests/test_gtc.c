/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   Correctness tests for libgtc: rational products (r2gtc and r2gtc_22 to
   r2gtc_55) against mpn_mul_n / mpn_mul / mpz_mul, Gaussian products (gtc
   and gtc_22 to gtc_55) against an mpz reference, over many sizes,
   thresholds, per-cap tables and adversarial inputs.
   Usage: test_gtc [seed] [--long]  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gtc.h"

static unsigned long long rng_state;

static mp_limb_t
rnd (void)
{
  /* xorshift64* */
  rng_state ^= rng_state >> 12;
  rng_state ^= rng_state << 25;
  rng_state ^= rng_state >> 27;
  return (mp_limb_t) (rng_state * 2685821657736338717ULL);
}

static mp_size_t
rnd_range (mp_size_t lo, mp_size_t hi)	/* inclusive */
{
  return lo + (mp_size_t) (rnd () % (mp_limb_t) (hi - lo + 1));
}

enum { PAT_RANDOM, PAT_ONES, PAT_RUNS, PAT_SPARSE, PAT_MIXED, PAT_REPEAT,
       PAT_HIGH, NPAT };
static const char *pat_name[NPAT] = { "random", "all-ones", "runs", "sparse",
  "mixed", "repeat", "high" };

/* Fill {p, n} with test data of the given pattern.  */
static void
fill (mp_ptr p, mp_size_t n, int pat)
{
  mp_size_t i, k;
  switch (pat)
    {
    case PAT_RANDOM:
      for (i = 0; i < n; i++)
	p[i] = rnd ();
      break;
    case PAT_ONES:
      for (i = 0; i < n; i++)
	p[i] = GMP_NUMB_MAX;
      break;
    case PAT_RUNS:			/* long runs of 0s and 1s */
      mpn_random2 (p, n);
      break;
    case PAT_SPARSE:
      for (i = 0; i < n; i++)
	p[i] = (rnd () % 8 == 0) ? rnd () : 0;
      if (n > 0)
	p[n - 1] |= 1;
      break;
    case PAT_MIXED:			/* each limb 0, ~0, 1, top bit or random */
      for (i = 0; i < n; i++)
	{
	  mp_limb_t r = rnd () % 5;
	  p[i] = r == 0 ? 0 : r == 1 ? GMP_NUMB_MAX : r == 2 ? 1
	    : r == 3 ? (mp_limb_t) 1 << (GMP_NUMB_BITS - 1) : rnd ();
	}
      break;
    case PAT_REPEAT:
      /* Equal thirds make a0 - a2 vanish at the top level (when 3 | n) and
	 produce many equal parts further down.  */
      k = (n + 2) / 3;
      for (i = 0; i < n; i++)
	p[i] = i < k ? rnd () : p[i % k];
      break;
    default:				/* PAT_HIGH: small low part, max top */
      for (i = 0; i < n; i++)
	p[i] = i < n / 2 ? (rnd () & 3) : GMP_NUMB_MAX - (rnd () & 3);
      break;
    }
}

#define CANARY ((mp_limb_t) 0x5a5a5a5a5a5a5a5aULL)

static long failures, checks;

/* One line of the summary: a label and the checks since before/fails.  */
static void
summary (const char *label, long before, long fails)
{
  printf ("%-64s%7ld checks, %ld failures\n", label, checks - before,
	  failures - fails);
}

static void
report (const char *what, const gtc_config *cfg, mp_size_t n, int pat)
{
  failures++;
  if (failures > 20)
    return;
  if (cfg == NULL)
    printf ("FAIL %s: n=%ld pattern=%s\n", what, (long) n, pat_name[pat]);
  else
    printf ("FAIL %s: n=%ld pattern=%s gtc22..55=%ld,%ld,%ld,%ld "
	    "km=%ld r2gtc22..55=%ld,%ld,%ld,%ld gmp22=%ld tables=%s%s%s%s/"
	    "%s%s%s%s\n", what, (long) n, pat_name[pat],
	    (long) cfg->gtc22_thresh, (long) cfg->gtc33_thresh,
	    (long) cfg->gtc44_thresh, (long) cfg->gtc55_thresh,
	    (long) cfg->gtc_km_thresh, (long) cfg->r2gtc22_thresh,
	    (long) cfg->r2gtc33_thresh, (long) cfg->r2gtc44_thresh,
	    (long) cfg->r2gtc55_thresh, (long) cfg->gmp22_thresh,
	    cfg->gtc_table[0] ? "2" : "-", cfg->gtc_table[1] ? "3" : "-",
	    cfg->gtc_table[2] ? "4" : "-", cfg->gtc_table[3] ? "5" : "-",
	    cfg->r2gtc_table[0] ? "2" : "-", cfg->r2gtc_table[1] ? "3" : "-",
	    cfg->r2gtc_table[2] ? "4" : "-", cfg->r2gtc_table[3] ? "5" : "-");
}

/* The rational products: r2gtc, then each r2gtc_ii with its level at the
   top (above 400 limbs one of them at random).  */
typedef void (*rat_fn) (gtc_ctx *, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
static const rat_fn rat_fns[5] = { r2gtc_mul_n_ctx, r2gtc_22_ctx,
  r2gtc_33_ctx, r2gtc_44_ctx, r2gtc_55_ctx };
static const char *const rat_names[5] = { "r2gtc", "r2gtc_22", "r2gtc_33",
  "r2gtc_44", "r2gtc_55" };

/* {r, 2n} by each rational algorithm against mpn_mul_n.  The whole
   result area and a margin around it are prefilled, so every result limb
   must be written and nothing outside; the inputs must be left alone.  */
static void
verify_mul_n (gtc_ctx *ctx, mp_srcptr a, mp_srcptr b, mp_size_t n, int pat)
{
  mp_ptr a2 = malloc (n * sizeof (mp_limb_t));
  mp_ptr b2 = malloc (n * sizeof (mp_limb_t));
  mp_ptr ref = malloc (2 * n * sizeof (mp_limb_t));
  mp_ptr buf = malloc ((2 * n + 4) * sizeof (mp_limb_t));
  mp_ptr r = buf + 2;
  const gtc_config *cfg = gtc_ctx_config (ctx);
  int one = n > 400 ? 1 + (int) (rnd () % 4) : 0;
  char what[64];

  mpn_copyi (a2, a, n);
  mpn_copyi (b2, b, n);
  mpn_mul_n (ref, a, b, n);
  for (int f = 0; f < 5; f++)
    {
      if (one && f && f != one)
	continue;
      for (mp_size_t i = 0; i < 2 * n + 4; i++)
	buf[i] = CANARY;
      rat_fns[f] (ctx, r, a, b, n);
      checks++;
      if (mpn_cmp (r, ref, 2 * n) != 0)
	snprintf (what, sizeof what, "%s result", rat_names[f]);
      else if (buf[0] != CANARY || buf[1] != CANARY
	       || buf[2 * n + 2] != CANARY || buf[2 * n + 3] != CANARY)
	snprintf (what, sizeof what, "%s wrote outside the result",
		  rat_names[f]);
      else if (mpn_cmp (a, a2, n) != 0 || mpn_cmp (b, b2, n) != 0)
	snprintf (what, sizeof what, "%s modified an input", rat_names[f]);
      else
	continue;
      report (what, cfg, n, pat);
    }

  free (a2); free (b2); free (ref); free (buf);
}

static void
check_mul_n (gtc_ctx *ctx, mp_size_t n, int pat_a, int pat_b)
{
  mp_ptr a = malloc (n * sizeof (mp_limb_t));
  mp_ptr b = malloc (n * sizeof (mp_limb_t));
  fill (a, n, pat_a);
  fill (b, n, pat_b);
  verify_mul_n (ctx, a, b, n, pat_a);
  free (a);
  free (b);
}

static void
limbs_to_mpz (mpz_t z, mp_srcptr p, mp_size_t n, int neg)
{
  mpz_t ro;
  mpz_set (z, mpz_roinit_n (ro, p, n));
  if (neg)
    mpz_neg (z, z);
}

/* The Gaussian products: gtc, then each gtc_ii with its level at the top
   (above 150 limbs one of them at random).  */
typedef void (*gauss_fn) (gtc_ctx *, mp_ptr, int *, mp_ptr, int *,
			  mp_srcptr, int, mp_srcptr, int, mp_srcptr, int,
			  mp_srcptr, int, mp_size_t);
static const gauss_fn gauss_fns[5] = { gtc_mul_n_ctx, gtc_22_ctx,
  gtc_33_ctx, gtc_44_ctx, gtc_55_ctx };
static const char *const gauss_names[5] = { "gtc", "gtc_22", "gtc_33",
  "gtc_44", "gtc_55" };

/* The Gaussian product of x[0] + i x[1] and x[2] + i x[3] (n limbs each,
   signs s[j]) by each Gaussian algorithm against (xr yr - xi yi) + i (xr yi
   + xi yr).  */
static void
verify_gauss (gtc_ctx *ctx, mp_ptr *x, const int *s, mp_size_t n, int pat)
{
  mp_ptr z[2];
  int zs[2], j;
  mpz_t v[4], ref_re, ref_im, got;
  const gtc_config *cfg = gtc_ctx_config (ctx);
  int one = n > 150 ? 1 + (int) (rnd () % 4) : 0;
  char what[64];

  for (j = 0; j < 4; j++)
    {
      mpz_init (v[j]);
      limbs_to_mpz (v[j], x[j], n, s[j]);
    }
  for (j = 0; j < 2; j++)
    z[j] = malloc ((2 * n + 3) * sizeof (mp_limb_t));
  mpz_inits (ref_re, ref_im, got, NULL);
  mpz_mul (ref_re, v[0], v[2]);
  mpz_submul (ref_re, v[1], v[3]);
  mpz_mul (ref_im, v[0], v[3]);
  mpz_addmul (ref_im, v[1], v[2]);

  for (int f = 0; f < 5; f++)
    {
      if (one && f && f != one)
	continue;
      for (j = 0; j < 2; j++)
	for (mp_size_t i = 0; i < 2 * n + 3; i++)
	  z[j][i] = CANARY;
      gauss_fns[f] (ctx, z[0] + 1, &zs[0], z[1] + 1, &zs[1],
		    x[0], s[0], x[1], s[1], x[2], s[2], x[3], s[3], n);
      checks++;

      limbs_to_mpz (got, z[0] + 1, 2 * n + 1, zs[0]);
      if (mpz_cmp (got, ref_re) != 0)
	{
	  snprintf (what, sizeof what, "%s real part", gauss_names[f]);
	  report (what, cfg, n, pat);
	}
      limbs_to_mpz (got, z[1] + 1, 2 * n + 1, zs[1]);
      if (mpz_cmp (got, ref_im) != 0)
	{
	  snprintf (what, sizeof what, "%s imaginary part", gauss_names[f]);
	  report (what, cfg, n, pat);
	}
      for (j = 0; j < 2; j++)
	if (z[j][0] != CANARY || z[j][2 * n + 2] != CANARY)
	  {
	    snprintf (what, sizeof what, "%s wrote outside the result",
		      gauss_names[f]);
	    report (what, cfg, n, pat);
	  }
      if ((zs[0] && mpz_sgn (ref_re) == 0) || (zs[1] && mpz_sgn (ref_im) == 0))
	{
	  snprintf (what, sizeof what, "%s negative zero", gauss_names[f]);
	  report (what, cfg, n, pat);
	}
    }

  for (j = 0; j < 4; j++)
    mpz_clear (v[j]);
  free (z[0]);
  free (z[1]);
  mpz_clears (ref_re, ref_im, got, NULL);
}

/* A random Gaussian product: mixed patterns, random signs, sometimes zero
   or equal components.  */
static void
check_gauss (gtc_ctx *ctx, mp_size_t n, int pat)
{
  mp_ptr x[4];
  int s[4], j;

  for (j = 0; j < 4; j++)
    {
      x[j] = malloc (n * sizeof (mp_limb_t));
      fill (x[j], n, (rnd () % 3 == 0) ? (int) (rnd () % NPAT) : pat);
      if (rnd () % 16 == 0)
	mpn_zero (x[j], n);
      if (j > 0 && rnd () % 16 == 0)
	mpn_copyi (x[j], x[j - 1], n);
      s[j] = rnd () & 1;
    }
  verify_gauss (ctx, x, s, n, pat);
  for (j = 0; j < 4; j++)
    free (x[j]);
}

/* r2gtc_mul (unbalanced) against mpn_mul.  */
static void
check_mul (gtc_ctx *ctx, mp_size_t an, mp_size_t bn, int pat)
{
  mp_ptr a = malloc (an * sizeof (mp_limb_t));
  mp_ptr b = malloc (bn * sizeof (mp_limb_t));
  mp_ptr ref = malloc ((an + bn) * sizeof (mp_limb_t));
  mp_ptr r = malloc ((an + bn) * sizeof (mp_limb_t));
  fill (a, an, pat);
  fill (b, bn, (int) (rnd () % NPAT));
  mpn_mul (ref, a, an, b, bn);
  r2gtc_mul_ctx (ctx, r, a, an, b, bn);
  checks++;
  if (mpn_cmp (r, ref, an + bn) != 0)
    report ("r2gtc_mul (unbalanced)", gtc_ctx_config (ctx), an, pat);
  free (a); free (b); free (ref); free (r);
}

/* r2gtc_mpz_mul with signs, zero and aliasing.  */
static void
check_mpz (gtc_ctx *ctx, mp_size_t an, mp_size_t bn)
{
  mpz_t a, b, ref, r;
  mp_ptr p;
  mpz_inits (a, b, ref, r, NULL);
  if (an > 0)
    {
      p = mpz_limbs_write (a, an);
      fill (p, an, (int) (rnd () % NPAT));
      p[an - 1] |= 1;
      mpz_limbs_finish (a, (rnd () & 1) ? -an : an);
    }
  if (bn > 0)
    {
      p = mpz_limbs_write (b, bn);
      fill (p, bn, (int) (rnd () % NPAT));
      p[bn - 1] |= 1;
      mpz_limbs_finish (b, (rnd () & 1) ? -bn : bn);
    }
  mpz_mul (ref, a, b);
  r2gtc_mpz_mul_ctx (ctx, r, a, b);
  checks++;
  if (mpz_cmp (r, ref) != 0)
    report ("r2gtc_mpz_mul", gtc_ctx_config (ctx), an, PAT_RANDOM);
  r2gtc_mpz_mul_ctx (ctx, a, a, b);	/* aliased output */
  checks++;
  if (mpz_cmp (a, ref) != 0)
    report ("r2gtc_mpz_mul aliased", gtc_ctx_config (ctx), an, PAT_RANDOM);
  mpz_clears (a, b, ref, r, NULL);
}

/* Exhaustive over extreme limb values: every n x n product with n <= 3
   through every algorithm's level at the top, and every Gaussian product
   of 1-limb components with all 16 sign patterns.  */
static void
check_tiny_exhaustive (gtc_ctx *c1, gtc_ctx *c2)
{
  static const mp_limb_t val[] = { 0, 1, 2, GMP_NUMB_MAX >> 1,
    (GMP_NUMB_MAX >> 1) + 1, GMP_NUMB_MAX - 1, GMP_NUMB_MAX };
  const long nv = sizeof val / sizeof val[0];
  mp_limb_t a[3], b[3], xs[4];
  mp_ptr x[4] = { &xs[0], &xs[1], &xs[2], &xs[3] };
  int s[4];

  for (mp_size_t n = 1; n <= 3; n++)
    {
      long total = 1;
      for (int i = 0; i < 2 * n; i++)
	total *= nv;
      for (long idx = 0; idx < total; idx++)
	{
	  long t = idx;
	  for (int i = 0; i < n; i++, t /= nv)
	    a[i] = val[t % nv];
	  for (int i = 0; i < n; i++, t /= nv)
	    b[i] = val[t % nv];
	  verify_mul_n (c1, a, b, n, PAT_MIXED);
	  verify_mul_n (c2, a, b, n, PAT_MIXED);
	}
    }
  for (long idx = 0; idx < nv * nv * nv * nv; idx++)
    for (int sg = 0; sg < 16; sg++)
      {
	long t = idx;
	for (int j = 0; j < 4; j++, t /= nv)
	  {
	    xs[j] = val[t % nv];
	    s[j] = (sg >> j) & 1;
	  }
	verify_gauss (c1, x, s, 1, PAT_MIXED);
	verify_gauss (c2, x, s, 1, PAT_MIXED);
      }
}

/* Nonnegative: make_ctx_k gives r2gtc22_thresh this value (below 2 it is
   clamped to 2).  */
static mp_size_t force_rat_kara = -1;
/* Nonnegative: make_ctx_k gives gtc_km_thresh this value (the Gaussian
   Karatsuba over {1, i} as the base case, where it is in assembly).  */
static mp_size_t force_km = -1;
/* Nonnegative: make_ctx_k gives gmp22_thresh this value (GMP's Karatsuba
   below the reductions).  */
static mp_size_t force_gmp_kara = -1;
/* Nonnegative: make_ctx_k gives gtc44_thresh this value.  */
static mp_size_t force_toom4 = -1;
/* Nonnegative: make_ctx_k gives gtc55_thresh this value.  */
static mp_size_t force_toom5 = -1;
/* Nonnegative: make_ctx_k gives r2gtc44_thresh this value.  */
static mp_size_t force_rat_toom4 = -1;
/* Nonnegative: make_ctx_k gives r2gtc55_thresh this value.  */
static mp_size_t force_rat_toom5 = -1;

/* rt: r2gtc33_thresh, gt: gtc33_thresh, gk: gtc22_thresh, rk:
   r2gtc22_thresh; 0 means the default.  */
static gtc_ctx *
make_ctx_k (mp_size_t rt, mp_size_t gt, mp_size_t gk, mp_size_t rk)
{
  gtc_config cfg;
  gtc_config_default (&cfg);
  if (rk)			/* 0: the default */
    cfg.r2gtc22_thresh = rk;
  if (force_rat_kara >= 0)
    cfg.r2gtc22_thresh = force_rat_kara;
  if (force_km >= 0)
    cfg.gtc_km_thresh = force_km;
  if (force_gmp_kara >= 0)
    cfg.gmp22_thresh = force_gmp_kara;
  if (force_toom4 >= 0)
    cfg.gtc44_thresh = force_toom4;
  if (force_toom5 >= 0)
    cfg.gtc55_thresh = force_toom5;
  if (force_rat_toom4 >= 0)
    cfg.r2gtc44_thresh = force_rat_toom4;
  if (force_rat_toom5 >= 0)
    cfg.r2gtc55_thresh = force_rat_toom5;
  /* Thresholds given: they decide at every size (the default tables
     would override them below 1024 limbs).  */
  if (rt || rk || force_rat_kara >= 0 || force_gmp_kara >= 0
      || force_rat_toom4 >= 0 || force_rat_toom5 >= 0)
    for (int j = 0; j < 4; j++)
      cfg.r2gtc_table[j] = NULL;
  if (gt || gk || force_toom4 >= 0 || force_toom5 >= 0)
    for (int j = 0; j < 4; j++)
      cfg.gtc_table[j] = NULL;
  if (gk)
    cfg.gtc22_thresh = gk;
  if (rt)
    cfg.r2gtc33_thresh = rt;
  if (gt)
    cfg.gtc33_thresh = gt;
  return gtc_ctx_new (&cfg);
}

static gtc_ctx *
make_ctx (mp_size_t rt, mp_size_t gt, mp_size_t rk)
{
  return make_ctx_k (rt, gt, 0, rk);
}

/* Gaussian products of nmin..nmax (<= 64) limbs whose limbs are extreme
   values, with random signs: with the Gaussian Karatsuba at every size (odd
   sizes give unequal halves), or with the Gaussian Toom-Cooks at every
   size, where the evaluations hit the bounds of their top limbs.  */
static void
check_gauss_extreme (gtc_ctx *ctx, mp_size_t nmin, mp_size_t nmax,
		     long iterations)
{
  static const mp_limb_t val[] = { 0, 1, 2, GMP_NUMB_MAX >> 1,
    (GMP_NUMB_MAX >> 1) + 1, GMP_NUMB_MAX - 1, GMP_NUMB_MAX };
  mp_limb_t xs[4][64];
  mp_ptr x[4] = { xs[0], xs[1], xs[2], xs[3] };
  int s[4];
  for (long it = 0; it < iterations; it++)
    {
      mp_size_t n = nmin + it % (nmax - nmin + 1);
      for (int j = 0; j < 4; j++)
	{
	  for (mp_size_t i = 0; i < n; i++)
	    xs[j][i] = val[rnd () % 7];
	  s[j] = rnd () & 1;
	}
      verify_gauss (ctx, x, s, n, PAT_MIXED);
    }
}

/* A random table for the cap i of the Gaussian side (rat = 0: of the
   characters t k 2 3 4 5 and others, as a string or a list) or of the
   rational side (rat = 1: a list of t g 2 3 4 5), choices above the cap
   and beyond what their sizes allow among them, now and then an item the
   parser stops at.  */
static void
random_table (char *tab, size_t size, int rat, int i)
{
  const char *chars = rat ? "tg2345" : "tk2345";
  (void) i;
  if (!rat && rnd () % 3)
    {
      int len = (int) rnd_range (2, (long) size - 1 < 399 ? (long) size - 1
				 : 399);
      for (int j = 0; j < len; j++)
	tab[j] = "tk2345-"[rnd () % 7];
      tab[len] = 0;
      return;
    }
  int m = (int) rnd_range (rat ? 0 : 1, 12), at = 0;
  tab[0] = 0;
  for (int j = 0; j < m && at < (int) size - 24; j++)
    {
      long lo = rnd_range (0, rat ? 1100 : 400);
      long hi = lo + rnd_range (0, rat ? 150 : 120);
      at += snprintf (tab + at, size - at, "%s%ld-%ld:%c",
		      j ? (rnd () & 1 ? " " : ",") : "", lo, hi,
		      rnd () % 16 ? chars[rnd () % 6] : 'x');
    }
}

int
main (int argc, char **argv)
{
  unsigned long long seed = 20260918;
  int longrun = 0, i, pat;
  mp_size_t n;

  for (i = 1; i < argc; i++)
    {
      if (strcmp (argv[i], "--long") == 0)
	longrun = 1;
      else
	seed = strtoull (argv[i], NULL, 0);
    }
  rng_state = seed ? seed : 1;
  printf ("libgtc %s, GMP %s, seed %llu\n", GTC_VERSION_STRING, gmp_version,
	  seed);

  /* Fixed configurations: from the deepest possible recursion (thresholds
     at the minimum, so rational and Gaussian levels interleave at tiny
     sizes) to the defaults.  0 means "default".  rk is r2gtc22_thresh: 2
     for r2gtc_22 down to 2 limbs, GTC_NEVER for the textbook product
     below r2gtc33_thresh.  Every product goes through r2gtc and each
     r2gtc_ii, gtc and each gtc_ii.  */
  static const struct { mp_size_t rt, gt, gk, rk; } cfgs[] = {
    { 5, 5, 0, 0 },
    { 5, 5, 0, 2 },
    { 7, 11, 0, 2 },
    { 11, 7, 0, 0 },
    { 13, 40, 0, 0 },
    { 40, 13, 0, 2 },
    { 5, 100000, 2, 0 },	/* Karatsuba */
    { 7, 60, 3, 2 },		/* T3 > K > base */
    { 60, 30, 9, GTC_NEVER },
    { 0, 0, 5, 0 },
    { 390, 0, 0, 0 },
    { 390, 0, 0, 45 },
    { 60, 0, 0, GTC_NEVER },
    { 0, 0, 0, 0 },
    { 0, 0, 0, 2 },
    { 40, 13, 4, GTC_NEVER },
    { 0, 0, 0, 45 },
    { 0, 0, 0, 90 },
    /* Karatsuba down to 2 limbs; the textbook product up to its largest
       size (the default Karatsuba threshold is lowered to it); Toom-3
       down to 5 limbs over the textbook product; all three.  */
    { 5, 100000, 0, 2 },
    { 7, 5, 0, 0 },
    { 0, 40, 12, 0 },
    { 0, 30, 15, 0 },
    { 130, 60, 14, 0 },
    /* recursing from 130 limbs, with the default Gaussian thresholds */
    { 130, 0, 0, 0 },
    /* no rational reduction below the top, no Gaussian Karatsuba */
    { GTC_NEVER, 68, GTC_NEVER, 0 },
  };
  const int ncfgs = sizeof cfgs / sizeof cfgs[0];

  for (i = 0; i < ncfgs; i++)
    {
      gtc_ctx *ctx = make_ctx_k (cfgs[i].rt, cfgs[i].gt, cfgs[i].gk,
				 cfgs[i].rk);
      const gtc_config *cfg = gtc_ctx_config (ctx);
      long before = checks, fails = failures;
      mp_size_t nmax = longrun ? 700 : 260;

      for (n = 1; n <= nmax; n++)
	for (pat = 0; pat < NPAT; pat++)
	  check_mul_n (ctx, n, pat, (pat + (int) n) % NPAT);
      for (int j = 0; j < (longrun ? 200 : 40); j++)
	{
	  n = rnd_range (nmax, longrun ? 6000 : 2500);
	  check_mul_n (ctx, n, (int) (rnd () % NPAT), (int) (rnd () % NPAT));
	}
      for (n = 1; n <= (longrun ? 300 : 120); n++)
	for (pat = 0; pat < NPAT; pat++)
	  check_gauss (ctx, n, pat);
      for (int j = 0; j < 30; j++)
	{
	  mp_size_t an = rnd_range (1, 3000), bn = rnd_range (1, an);
	  check_mul (ctx, an, bn, (int) (rnd () % NPAT));
	  check_mpz (ctx, rnd_range (0, 400), rnd_range (0, 400));
	}
      char label[224], rat[24], rk[24], gauss[24], kara[24], km[24], gk[24];
      if (cfg->r2gtc33_thresh >= GTC_NEVER)
	strcpy (rat, "never");
      else
	snprintf (rat, sizeof rat, "%ld", (long) cfg->r2gtc33_thresh);
      if (cfg->r2gtc22_thresh >= GTC_NEVER)
	strcpy (rk, "never");
      else
	snprintf (rk, sizeof rk, "%ld", (long) cfg->r2gtc22_thresh);
      if (cfg->gtc_table[GTC_CAP (5)] != NULL)
	strcpy (gauss, "table");
      else
	snprintf (gauss, sizeof gauss, "%ld", (long) cfg->gtc33_thresh);
      if (cfg->gtc_table[GTC_CAP (5)] != NULL)
	strcpy (kara, "-");
      else if (cfg->gtc22_thresh >= GTC_NEVER)
	strcpy (kara, "never");
      else
	snprintf (kara, sizeof kara, "%ld", (long) cfg->gtc22_thresh);
      if (cfg->gtc_km_thresh >= GTC_NEVER)
	strcpy (km, "never");
      else
	snprintf (km, sizeof km, "%ld", (long) cfg->gtc_km_thresh);
      if (cfg->gmp22_thresh >= GTC_NEVER)
	strcpy (gk, "never");
      else
	snprintf (gk, sizeof gk, "%ld", (long) cfg->gmp22_thresh);
      snprintf (label, sizeof label, "config r2gtc33/22=%-5s/%-5s "
		"gtc33=%-6s gtc22=%-5s km=%-5s gmp22=%-5s:", rat, rk, gauss,
		kara, km, gk);
      summary (label, before, fails);
      gtc_ctx_free (ctx);
    }

  /* GMP's Karatsuba below the reductions: from 4 limbs, its smallest
     (GTC_GMP22_MIN; every rational product from there up to them),
     from 9 and 25, with r2gtc_22 from 30 limbs or not at all.  */
  {
    long before = checks, fails = failures;
    static const mp_size_t xs[] = { 4, 9, 25 }, ts[] = { 30, GTC_NEVER };
    for (int xi = 0; xi < 3; xi++)
      for (int ti = 0; ti < 2; ti++)
	{
	  gtc_ctx *ctx;
	  force_gmp_kara = xs[xi];
	  ctx = make_ctx (41, 0, ts[ti]);
	  for (n = 1; n <= (longrun ? 400 : 160); n++)
	    for (pat = 0; pat < NPAT; pat++)
	      check_mul_n (ctx, n, pat, (pat + (int) n) % NPAT);
	  for (int j = 0; j < (longrun ? 60 : 12); j++)
	    {
	      mp_size_t an = rnd_range (1, 3000), bn = rnd_range (1, an);
	      check_mul (ctx, an, bn, (int) (rnd () % NPAT));
	      check_mul_n (ctx, rnd_range (160, 3000),
			   (int) (rnd () % NPAT), (int) (rnd () % NPAT));
	    }
	  gtc_ctx_free (ctx);
	}
    force_gmp_kara = -1;
    summary ("GMP's Karatsuba below the reductions:", before, fails);
  }

  /* gtc_44: from its smallest size, over gtc_33, gtc_22 and the textbook
     product at small and default thresholds, on every size and pattern,
     on extreme limbs (the evaluations' top limbs at their bounds), and
     under the reductions.  */
  {
    long before = checks, fails = failures;
    static const struct { mp_size_t g4, gt, gk; } t4[] = {
      { 13, 5, 0 }, { 13, 100000, 2 }, { 13, 100000, 100000 },
      { 20, 9, 4 }, { 40, 0, 0 }, { 13, 0, 0 }, { 100, 0, 0 },
      { 13, 30, 0 } };
    for (unsigned v = 0; v < sizeof t4 / sizeof t4[0]; v++)
      {
	gtc_ctx *ctx;
	force_toom4 = t4[v].g4;
	ctx = make_ctx_k (0, t4[v].gt, t4[v].gk, 0);
	for (n = 1; n <= (longrun ? 400 : 160); n++)
	  for (pat = 0; pat < NPAT; pat++)
	    check_gauss (ctx, n, pat);
	for (int j = 0; j < (longrun ? 40 : 10); j++)
	  check_gauss (ctx, rnd_range (160, 1500), (int) (rnd () % NPAT));
	check_gauss_extreme (ctx, 13, 64, longrun ? 40000 : 8000);
	for (n = 1; n <= (longrun ? 300 : 120); n += 7)
	  check_mul_n (ctx, n, (int) (rnd () % NPAT), (int) (rnd () % NPAT));
	for (int j = 0; j < (longrun ? 30 : 6); j++)
	  check_mul_n (ctx, rnd_range (120, 3000), (int) (rnd () % NPAT),
		       (int) (rnd () % NPAT));
	gtc_ctx_free (ctx);
      }
    force_toom4 = -1;
    summary ("gtc_44:", before, fails);
  }

  /* gtc_55: from its smallest size, over itself, gtc_44, gtc_33, gtc_22
     and the textbook product at small and default thresholds, on every
     size and pattern, on extreme limbs (the evaluations' top limbs at
     their bounds), and under the reductions.  */
  {
    long before = checks, fails = failures;
    static const struct { mp_size_t g5, g4, gt, gk; } t5[] = {
      { 17, 13, 5, 0 }, { 17, GTC_NEVER, 100000, 2 },
      { 17, GTC_NEVER, 100000, 100000 }, { 17, GTC_NEVER, 5, 0 },
      { 25, 13, 9, 4 }, { 40, 0, 0, 0 }, { 17, 0, 0, 0 },
      { 90, 20, 0, 0 }, { 200, 0, 0, 0 }, { 17, 30, 0, 0 } };
    for (unsigned v = 0; v < sizeof t5 / sizeof t5[0]; v++)
      {
	gtc_ctx *ctx;
	force_toom5 = t5[v].g5;
	force_toom4 = t5[v].g4 ? t5[v].g4 : -1;
	ctx = make_ctx_k (0, t5[v].gt, t5[v].gk, 0);
	for (n = 1; n <= (longrun ? 500 : 200); n++)
	  for (pat = 0; pat < NPAT; pat++)
	    check_gauss (ctx, n, pat);
	for (int j = 0; j < (longrun ? 40 : 10); j++)
	  check_gauss (ctx, rnd_range (200, 2500), (int) (rnd () % NPAT));
	check_gauss_extreme (ctx, 17, 64, longrun ? 40000 : 8000);
	for (n = 1; n <= (longrun ? 300 : 120); n += 7)
	  check_mul_n (ctx, n, (int) (rnd () % NPAT), (int) (rnd () % NPAT));
	for (int j = 0; j < (longrun ? 30 : 6); j++)
	  check_mul_n (ctx, rnd_range (120, 4000), (int) (rnd () % NPAT),
		       (int) (rnd () % NPAT));
	gtc_ctx_free (ctx);
      }
    force_toom5 = -1;
    force_toom4 = -1;
    summary ("gtc_55:", before, fails);
  }

  /* r2gtc_44: at the top of every product from its smallest size, and
     recursing from 13 limbs and more over r2gtc_33, r2gtc_22, GMP's
     Karatsuba and the textbook product, with the Gaussian products by
     gtc_44, gtc_33, gtc_22 and the textbook product; on every size and
     pattern, on extreme limbs (the evaluations' top limbs at their
     bounds), and unbalanced.  */
  {
    long before = checks, fails = failures;
    static const struct { mp_size_t r4, rt, rk, gk, g4, gt; }
      t4[] = {
      { 13, 5, 2, GTC_NEVER, 13, 5 },
      { 13, GTC_NEVER, GTC_NEVER, GTC_NEVER, GTC_NEVER, 0 },
      { 13, 5, 0, 0, 13, 0 },
      { 17, 40, 2, GTC_NEVER, GTC_NEVER, 100000 },
      { 26, 13, 0, 9, 40, 0 },
      { 40, 0, GTC_NEVER, 4, 13, 5 },
      { 100, 0, 0, 0, 0, 0 },
      { 300, 0, 0, 0, 0, 0 },
      { GTC_NEVER, 0, 0, 0, 0, 0 } };
    static const mp_limb_t val[] = { 0, 1, 2, GMP_NUMB_MAX >> 1,
      (GMP_NUMB_MAX >> 1) + 1, GMP_NUMB_MAX - 1, GMP_NUMB_MAX };
    for (unsigned v = 0; v < sizeof t4 / sizeof t4[0]; v++)
      {
	gtc_ctx *ctx;
	force_rat_toom4 = t4[v].r4;
	force_gmp_kara = t4[v].gk;
	force_toom4 = t4[v].g4;
	ctx = make_ctx (t4[v].rt, t4[v].gt, t4[v].rk);
	for (n = 1; n <= (longrun ? 500 : 200); n++)
	  for (pat = 0; pat < NPAT; pat++)
	    check_mul_n (ctx, n, pat, (pat + (int) n) % NPAT);
	for (int j = 0; j < (longrun ? 60 : 15); j++)
	  check_mul_n (ctx, rnd_range (200, 5000), (int) (rnd () % NPAT),
		       (int) (rnd () % NPAT));
	for (int j = 0; j < (longrun ? 20000 : 4000); j++)
	  {
	    mp_limb_t xa[80], xb[80];
	    n = rnd_range (13, 80);
	    for (mp_size_t i = 0; i < n; i++)
	      {
		xa[i] = val[rnd () % 7];
		xb[i] = val[rnd () % 7];
	      }
	    verify_mul_n (ctx, xa, xb, n, PAT_MIXED);
	  }
	for (int j = 0; j < (longrun ? 30 : 8); j++)
	  {
	    mp_size_t an = rnd_range (1, 4000), bn = rnd_range (1, an);
	    check_mul (ctx, an, bn, (int) (rnd () % NPAT));
	  }
	gtc_ctx_free (ctx);
      }
    force_rat_toom4 = -1;
    force_gmp_kara = -1;
    force_toom4 = -1;
    summary ("r2gtc_44:", before, fails);
  }

  /* r2gtc_55: at the top of every product from its smallest size, and
     recursing from 17 limbs and more over r2gtc_44, r2gtc_33, r2gtc_22,
     GMP's Karatsuba and the textbook product, with the Gaussian products
     by gtc_55 to gtc_22 and the textbook product; on every size and
     pattern, on extreme limbs (the evaluations' top limbs at their
     bounds), and unbalanced.  */
  {
    long before = checks, fails = failures;
    static const struct { mp_size_t r5, r4, rt, rk, gk, g5, g4, gt; }
      t5[] = {
      { 17, 13, 5, 2, GTC_NEVER, 17, 13, 5 },
      { 17, GTC_NEVER, GTC_NEVER, GTC_NEVER, GTC_NEVER, GTC_NEVER,
	GTC_NEVER, 0 },
      { 17, GTC_NEVER, 5, 0, 0, GTC_NEVER, 13, 0 },
      { 17, 13, 40, 2, GTC_NEVER, 17, GTC_NEVER, 100000 },
      { 21, 17, 13, 0, 9, 40, 40, 0 },
      { 30, GTC_NEVER, 0, GTC_NEVER, 4, 17, 13, 5 },
      { 60, 20, 0, 0, 0, 30, 0, 0 },
      { 150, 0, 0, 0, 0, 0, 0, 0 },
      { 400, 0, 0, 0, 0, 0, 0, 0 },
      { GTC_NEVER, 0, 0, 0, 0, 0, 0, 0 } };
    static const mp_limb_t val[] = { 0, 1, 2, GMP_NUMB_MAX >> 1,
      (GMP_NUMB_MAX >> 1) + 1, GMP_NUMB_MAX - 1, GMP_NUMB_MAX };
    for (unsigned v = 0; v < sizeof t5 / sizeof t5[0]; v++)
      {
	gtc_ctx *ctx;
	force_rat_toom5 = t5[v].r5;
	force_rat_toom4 = t5[v].r4 ? t5[v].r4 : -1;
	force_gmp_kara = t5[v].gk;
	force_toom5 = t5[v].g5 ? t5[v].g5 : -1;
	force_toom4 = t5[v].g4 ? t5[v].g4 : -1;
	ctx = make_ctx (t5[v].rt, t5[v].gt, t5[v].rk);
	for (n = 1; n <= (longrun ? 500 : 200); n++)
	  for (pat = 0; pat < NPAT; pat++)
	    check_mul_n (ctx, n, pat, (pat + (int) n) % NPAT);
	for (int j = 0; j < (longrun ? 60 : 15); j++)
	  check_mul_n (ctx, rnd_range (200, 6000), (int) (rnd () % NPAT),
		       (int) (rnd () % NPAT));
	for (int j = 0; j < (longrun ? 20000 : 4000); j++)
	  {
	    mp_limb_t xa[100], xb[100];
	    n = rnd_range (17, 100);
	    for (mp_size_t i = 0; i < n; i++)
	      {
		xa[i] = val[rnd () % 7];
		xb[i] = val[rnd () % 7];
	      }
	    verify_mul_n (ctx, xa, xb, n, PAT_MIXED);
	  }
	for (int j = 0; j < (longrun ? 30 : 8); j++)
	  {
	    mp_size_t an = rnd_range (1, 5000), bn = rnd_range (1, an);
	    check_mul (ctx, an, bn, (int) (rnd () % NPAT));
	  }
	gtc_ctx_free (ctx);
      }
    force_rat_toom5 = -1;
    force_rat_toom4 = -1;
    force_gmp_kara = -1;
    force_toom5 = -1;
    force_toom4 = -1;
    summary ("r2gtc_55:", before, fails);
  }

  /* Random configurations.  */
  {
    long before = checks, fails = failures;
    for (i = 0; i < (longrun ? 3000 : 500); i++)
      {
	/* Thresholds mostly small (deep interleaving), sometimes large (long
	   stretches of r2gtc_22 or the textbook product).  */
	gtc_ctx *ctx;
	force_rat_kara = (mp_size_t[]) { 2, 3, 5, 9, 21, 60, 400, GTC_NEVER }
	  [rnd () % 8];
	force_km = (mp_size_t[]) { 2, 3, 5, 7, 9, 11, GTC_NEVER, GTC_NEVER }
	  [rnd () % 8];
	force_gmp_kara = (mp_size_t[]) { 4, 5, 6, 9, 17, 30, GTC_NEVER,
					 GTC_NEVER } [rnd () % 8];
	force_toom4 = (mp_size_t[]) { 13, 14, 17, 25, 40, 90, GTC_NEVER,
				      GTC_NEVER } [rnd () % 8];
	force_toom5 = (mp_size_t[]) { 17, 18, 21, 30, 50, 120, GTC_NEVER,
				      GTC_NEVER } [rnd () % 8];
	force_rat_toom4 = (mp_size_t[]) { 13, 14, 19, 30, 60, 150, GTC_NEVER,
					  GTC_NEVER } [rnd () % 8];
	force_rat_toom5 = (mp_size_t[]) { 17, 18, 22, 35, 70, 200, GTC_NEVER,
					  GTC_NEVER } [rnd () % 8];
	ctx = make_ctx_k (rnd_range (5, (rnd () & 1) ? 70 : 360),
			  rnd_range (5, (rnd () & 1) ? 70 : 200),
			  rnd_range (2, 40), 0);
	n = rnd_range (1, (rnd () & 7) ? 800 : 5000);
	check_mul_n (ctx, n, (int) (rnd () % NPAT), (int) (rnd () % NPAT));
	check_gauss (ctx, rnd_range (1, 300), (int) (rnd () % NPAT));
	gtc_ctx_free (ctx);
      }
    force_rat_kara = -1;
    force_km = -1;
    force_gmp_kara = -1;
    force_toom4 = -1;
    force_toom5 = -1;
    force_rat_toom4 = -1;
    force_rat_toom5 = -1;
    summary ("random configurations:", before, fails);
  }

  /* Random per-cap tables, over random thresholds: each cap's table (or
     none) on either side, of every algorithm's character and characters
     that leave the size to the thresholds, so that each gtc_ii and
     r2gtc_ii hands its products to every algorithm it may, including
     choices above the cap and ones a size does not allow (they fall back
     to the thresholds or a smaller algorithm).  */
  {
    long before = checks, fails = failures;
    static char gtab[4][400], rtab[4][300];
    for (i = 0; i < (longrun ? 600 : 120); i++)
      {
	gtc_config cfg;
	gtc_ctx *ctx;
	gtc_config_default (&cfg);
	for (int I = 2; I <= 5; I++)
	  {
	    random_table (gtab[GTC_CAP (I)], sizeof gtab[0], 0, I);
	    random_table (rtab[GTC_CAP (I)], sizeof rtab[0], 1, I);
	    cfg.gtc_table[GTC_CAP (I)] = rnd () % 4 ? gtab[GTC_CAP (I)] : NULL;
	    cfg.r2gtc_table[GTC_CAP (I)] = rnd () % 4 ? rtab[GTC_CAP (I)]
	      : NULL;
	  }
	cfg.gtc33_thresh = rnd_range (5, 200);
	cfg.gtc22_thresh = rnd_range (2, 40);
	cfg.gtc44_thresh = (rnd () & 1) ? rnd_range (13, 300) : GTC_NEVER;
	cfg.gtc55_thresh = (rnd () & 1) ? rnd_range (17, 400) : GTC_NEVER;
	cfg.r2gtc22_thresh = (mp_size_t[]) { 2, 9, 40, GTC_NEVER }[rnd () % 4];
	cfg.gmp22_thresh = (mp_size_t[]) { 4, 9, 40, GTC_NEVER }[rnd () % 4];
	cfg.r2gtc33_thresh = rnd_range (5, 200);
	cfg.r2gtc44_thresh = (rnd () & 1) ? rnd_range (13, 600) : GTC_NEVER;
	cfg.r2gtc55_thresh = (rnd () & 1) ? rnd_range (17, 900) : GTC_NEVER;
	ctx = gtc_ctx_new (&cfg);
	for (int j = 0; j < 4; j++)
	  check_gauss (ctx, rnd_range (1, 450), (int) (rnd () % NPAT));
	check_mul_n (ctx, rnd_range (1, (rnd () & 3) ? 1400 : 4000),
		     (int) (rnd () % NPAT), (int) (rnd () % NPAT));
	for (int j = 0; j < 2; j++)
	  check_mul_n (ctx, rnd_range (1, 1100), (int) (rnd () % NPAT),
		       (int) (rnd () % NPAT));
	gtc_ctx_free (ctx);
      }
    summary ("random per-cap tables:", before, fails);
  }

  /* Exhaustive tiny sizes with extreme limb values.  */
  {
    long before = checks, fails = failures;
    gtc_ctx *c1 = make_ctx (5, 5, 0);
    gtc_ctx *c2 = make_ctx (5, 5, 2);
    check_tiny_exhaustive (c1, c2);
    gtc_ctx_free (c1);
    gtc_ctx_free (c2);
    {
      gtc_ctx *kara = make_ctx_k (5, 100000, 2, 0);
      check_gauss_extreme (kara, 2, 12, longrun ? 200000 : 30000);
      gtc_ctx_free (kara);
      /* The textbook product as the base case, up to its largest size
	 (with the assembly: 10 limbs flat, the four-way split from 11 to
	 20), and the Gaussian Karatsuba above.  */
      gtc_ctx *pure = make_ctx_k (5, 100000, 0, 0);
      check_gauss_extreme (pure, 1, 24, longrun ? 100000 : 30000);
      gtc_ctx_free (pure);
      /* A per-size table with gtc_33: the textbook product up to 16
	 limbs, gtc_33 from 17 to 40 (over the textbook product up to 50
	 limbs, as over gtc_22 and itself above), gtc_22 from 41 to 64
	 (over the textbook product, its products' choice under the cap
	 2), for every cap.  */
	{
	  static char tab[65];
	  gtc_config cfg;
	  gtc_ctx *t;
	  for (int j = 0; j < 65; j++)
	    tab[j] = j <= 16 ? 't' : j <= 40 ? '3' : 'k';
	  tab[64] = 0;
	  gtc_config_default (&cfg);
	  for (int j = 0; j < 4; j++)
	    cfg.gtc_table[j] = tab;
	  t = gtc_ctx_new (&cfg);
	  check_gauss_extreme (t, 1, 64, longrun ? 100000 : 20000);
	  gtc_ctx_free (t);
	}
      /* gtc_22 over the textbook product at every size from 4 to 40, and
	 over gtc_22 up to 64 (gtc_kara_x86.S with the assembly): tables
	 that make the sizes of one range gtc_22 and those below the
	 textbook product, for every cap.  */
      {
	static const int ranges[6][3] = {	/* textbook up to, kara from, to */
	  { 3, 4, 5 }, { 5, 6, 10 }, { 10, 11, 20 }, { 20, 21, 40 },
	  { 10, 11, 64 }, { 16, 33, 64 } };
	for (int v = 0; v < 6; v++)
	  {
	    static char tab[66];
	    gtc_config cfg;
	    gtc_ctx *t;
	    for (int j = 0; j <= 64; j++)
	      tab[j] = j <= ranges[v][0] ? 't' : j <= ranges[v][2] ? 'k' : '-';
	    tab[65] = 0;
	    gtc_config_default (&cfg);
	    for (int j = 0; j < 4; j++)
	      cfg.gtc_table[j] = tab;
	    t = gtc_ctx_new (&cfg);
	    check_gauss_extreme (t, ranges[v][1], ranges[v][2],
				 longrun ? 60000 : 12000);
	    gtc_ctx_free (t);
	  }
      }
      /* The Gaussian Karatsuba over {1, i} as the base case at every size
	 it covers (gtc_km_arm.S, 2 to 10 limbs; elsewhere the textbook
	 product), alone, under the gtc_22 levels from 11 limbs (in
	 assembly, through the entries without the callee-saved registers)
	 and, from 6 limbs, under gtc_33.  */
      for (int v = 0; v < 3; v++)
	{
	  gtc_ctx *t;
	  force_km = v == 2 ? 6 : 2;
	  t = make_ctx_k (5, v == 2 ? 5 : 100000, v == 1 ? 11 : 100000, 0);
	  force_km = -1;
	  check_gauss_extreme (t, 1, v ? 40 : 12, longrun ? 100000 : 30000);
	  for (n = 1; n <= 80; n++)
	    check_mul_n (t, n, PAT_RANDOM, (int) n % NPAT);
	  gtc_ctx_free (t);
	}
      /* gtc_33, from 5 limbs on (the base case below), where the
	 evaluations hit the bounds of their top limbs.  */
	{
	  gtc_ctx *h = make_ctx_k (5, 5, 100000, 0);
	  check_gauss_extreme (h, 5, 14, longrun ? 200000 : 30000);
	  gtc_ctx_free (h);
	}
    }
    summary ("extreme limbs (tiny sizes, Gaussian, Karatsuba, Toom-3):", before, fails);
  }

  /* A few large products with the defaults (default per-thread context).  */
  {
    long before = checks, fails = failures;
    static const mp_size_t big[] = { 10000, 33333, 100001 };
    for (i = 0; i < 3; i++)
      {
	mp_size_t bn = big[i];
	mp_ptr a = malloc (bn * sizeof (mp_limb_t));
	mp_ptr b = malloc (bn * sizeof (mp_limb_t));
	mp_ptr r = malloc (2 * bn * sizeof (mp_limb_t));
	mp_ptr ref = malloc (2 * bn * sizeof (mp_limb_t));
	fill (a, bn, i == 1 ? PAT_ONES : PAT_RANDOM);
	fill (b, bn, i == 2 ? PAT_RUNS : PAT_RANDOM);
	mpn_mul_n (ref, a, b, bn);
	r2gtc_mul_n (r, a, b, bn);
	checks++;
	if (mpn_cmp (r, ref, 2 * bn) != 0)
	  {
	    gtc_config cfg;
	    gtc_config_default (&cfg);
	    report ("large r2gtc_mul_n", &cfg, bn, PAT_RANDOM);
	  }
	free (a); free (b); free (r); free (ref);
      }
    /* The reductions recursing from 130 limbs, and applied once, up to
       and beyond the benchmarks' largest size (the defaults, above,
       recurse from 30 limbs with the assembly).  */
    static const mp_size_t bigr[] = { 6562, 20000, 19683, 59049 };
    {
      gtc_ctx *rec = make_ctx (130, 0, 0);
      gtc_ctx *one = make_ctx (GTC_NEVER, 0, 0);
      gtc_ctx *three = make_ctx_k (GTC_NEVER, 68, GTC_NEVER, 0);
      for (i = 0; i < (longrun ? 4 : 2); i++)
	{
	  check_mul_n (rec, bigr[i], i & 1 ? PAT_RUNS : PAT_RANDOM, PAT_ONES);
	  check_mul_n (one, 5 * bigr[i] + 1, PAT_RANDOM, PAT_HIGH);
	  check_mul_n (three, 3 * bigr[i] + 2, PAT_HIGH, PAT_RANDOM);
	}
      gtc_ctx_free (rec);
      gtc_ctx_free (one);
      gtc_ctx_free (three);
    }
    summary ("large sizes (default context; the reductions):", before,
	     fails);
  }

  printf ("%s: %ld checks, %ld failures\n", failures ? "FAILED" : "PASSED",
	  checks, failures);
  return failures != 0;
}
