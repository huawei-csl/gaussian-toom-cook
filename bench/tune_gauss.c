/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   tune_gauss -- where the Gaussian and the rational products should
   switch algorithms.

   The base case first: where the Gaussian Karatsuba over {1, i} is in
   assembly (aarch64), the textbook product against it: KM, where it
   starts to win for good, is the base case's switch (gtc_km_thresh),
   under which K is measured.  Then, as GMP's tuneup does, each threshold
   compares one level of the faster algorithm, whose children use what lies
   below it, with the slower algorithm, at every size n:
     K: gtc_22 (children: the base case) against the base case;
     G: gtc_33 (children: gtc_22 from K, else the base case) against gtc_22
        from K and the base case below;
     and gtc_33 over the base case alone against the base case alone;
     with --g4max, G4: gtc_44 over those; with --toom5-only, G5.

   Then, with --rational, the same for the rational products, over the
   Gaussian thresholds just found (or given):
     X: GMP's Karatsuba, mpn_toom22_mul (its children: GMP's thresholds)
        against the textbook product;
     T: r2gtc_22 (children: GMP's Karatsuba from X, else the textbook
        product) against what lies below it;
     R: r2gtc_33 (children: r2gtc_22 from T, else GMP's Karatsuba from X,
        else the textbook product) against what lies below it.
   --rational-only skips the Gaussian sweeps (give --kara and --toom).

   --table [N] --cap I decides gtc_ii's products (gtc_table[GTC_CAP (I)])
   size by size, bottom-up, below N (default 1024): at size n every
   algorithm the cap allows is timed with its products following the
   choices already made for the smaller sizes, and the fastest is kept.
   The other caps' tables are the defaults or --set-table's; the products
   of a gtc_22 level are the cap 2's, so its table comes first.

   --rat-table [LO:HI] --cap I does the same for r2gtc_ii's rational
   products (r2gtc_table[GTC_CAP (I)]) from LO (default 2) to HI (default
   1023): one level of every algorithm the cap allows (the textbook
   product, GMP's Karatsuba, r2gtc_22 to r2gtc_ii), three times
   interleaved, the minimum of each.  A size leaves the thresholds' choice
   only where the fastest beats it by more than --margin (default 0.003).

   Both print the table as gtc_config takes it.  Times are core cycles
   (perf_event_open) per product of n-limb first-quadrant factors, cycling
   through a pool of inputs, minimum over trials.  Run it on a quiet,
   pinned core:  ./tune_gauss --pin 3  */

#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <sched.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "gtc.c"

static int fdc = -1;
static gmp_randstate_t rs;
static int trials = 15;

/* Cycles per product of gtc_q1 (rational: r2gtc) under the cap cap with
   context c at size n.  */
static int rational;
static int cap = 5;

static double
cycles (gtc_ctx *c, mp_size_t n, mp_ptr in, int pool, mp_ptr P, mp_ptr Q)
{
  long reps = 200000 / (n * n + 10) + 20;
  double best = 1e300;
  for (int t = 0; t < trials; t++)
    {
      long long cy;
      ioctl (fdc, PERF_EVENT_IOC_RESET, 0);
      ioctl (fdc, PERF_EVENT_IOC_ENABLE, 0);
      for (long i = 0; i < reps; i++)
	{
	  mp_ptr x = in + (i & (pool - 1)) * 4 * n;
	  if (rational)
	    r2gtc (c, cap, P, x, x + n, n);
	  else
	    gtc_q1 (c, cap, P, Q, x, x + n, x + 2 * n, x + 3 * n, n);
	}
      ioctl (fdc, PERF_EVENT_IOC_DISABLE, 0);
      if (read (fdc, &cy, sizeof cy) == sizeof cy && (double) cy / reps < best)
	best = (double) cy / reps;
      arena_fit (c);
    }
  return best;
}

/* gtc_km_thresh of every context: the measured KM, or --km.  */
static mp_size_t km = GTC_DEFAULT_GTC_KM;

/* gtc44_thresh of every context: GTC_NEVER but in the G4 sweep
   (and --g4 in the G5 one).  */
static mp_size_t g4 = GTC_NEVER;
/* gtc55_thresh of every context: GTC_NEVER but in the G5 sweep.  */
static mp_size_t g5 = GTC_NEVER;

static gtc_ctx *
ctx (mp_size_t g, mp_size_t k)
{
  gtc_config cfg;
  gtc_config_default (&cfg);
  cfg.gtc44_thresh = g4;
  cfg.gtc55_thresh = g5;
  cfg.r2gtc33_thresh = GTC_NEVER;	/* the real products are textbook ones */
  cfg.r2gtc22_thresh = GTC_NEVER;
  cfg.gtc33_thresh = g;
  cfg.gtc22_thresh = k;
  cfg.gtc_km_thresh = km;
  for (int j = 0; j < 4; j++)	/* the thresholds decide */
    cfg.gtc_table[j] = cfg.r2gtc_table[j] = NULL;
  return gtc_ctx_new (&cfg);
}

/* --set-table I STRING, --set-rat-table I STRING: the tables of the caps
   the tuning takes as given, over the defaults.  */
static const char *set_gtab[4], *set_rtab[4];

static void
tables_given (gtc_config *cfg)
{
  for (int j = 0; j < 4; j++)
    {
      if (set_gtab[j])
	cfg->gtc_table[j] = *set_gtab[j] ? set_gtab[j] : NULL;
      if (set_rtab[j])
	cfg->r2gtc_table[j] = *set_rtab[j] ? set_rtab[j] : NULL;
    }
}

/* A table's list form from its choices per size (0: the thresholds),
   ranges of equal choices joined.  */
static void
table_string (char *out, size_t len, const char *ch, mp_size_t n)
{
  size_t at = 0;
  out[0] = 0;
  for (mp_size_t i = 0; i < n;)
    {
      mp_size_t j = i;
      if (!ch[i])
	{
	  i++;
	  continue;
	}
      while (j + 1 < n && ch[j + 1] == ch[i])
	j++;
      if (j > i)
	at += snprintf (out + at, len - at, "%s%ld-%ld:%c", at ? " " : "",
			(long) i, (long) j, ch[i]);
      else
	at += snprintf (out + at, len - at, "%s%ld:%c", at ? " " : "",
			(long) i, ch[i]);
      i = j + 1;
    }
}

/* --table --cap I: the algorithm of gtc_ii's products per size, decided
   bottom-up as GMP's tuneup decides its thresholds, but at every size: at
   size n each algorithm the cap allows is timed with its products
   following the choices already made for the smaller sizes, and the
   fastest is kept.  So a gtc_33 or gtc_22 level hands each product to
   what is fastest at the product's size (gtc_33 to the textbook product
   directly, say), which shows where the thresholds lose and by how much.
   The products of gtc_22 are the cap 2's: under a higher cap they follow
   the cap 2's table (the default, or --set-table 2).  "(table)" marks
   the sizes where the choice is not the thresholds'.  Prints the table
   as gtc_config.gtc_table takes it.  */
static void fill_pool (mp_ptr in, int pool, mp_size_t n);

static void
tune_table (mp_size_t nmax)
{
  const int pool = 64;
  /* by GTC_ALG_*: the base case, (the leaves), gtc_22, gtc_33, gtc_44,
     gtc_55 */
  const char name[] = "t-2345";
  enum { NA = 6 };
  static char ch[GTC_TABLE_N], str[16384];
  mp_ptr in = malloc (pool * 4 * nmax * sizeof (mp_limb_t));
  mp_ptr P = malloc ((2 * nmax + 4) * sizeof (mp_limb_t));
  mp_ptr Q = malloc ((2 * nmax + 4) * sizeof (mp_limb_t));
  gtc_config cfg;
  gtc_ctx *c;

  gtc_config_default (&cfg);
  tables_given (&cfg);
  cfg.gtc_table[GTC_CAP (cap)] = NULL;
  c = gtc_ctx_new (&cfg);
  memset (ch, 0, sizeof ch);
  printf ("\n# gtc_table[GTC_CAP (%d)]: the products of gtc_%d%d\n"
	  "%5s %10s %10s %10s %10s %10s  choice\n", cap, cap, cap, "n", "base",
	  "gtc_22", "gtc_33", "gtc_44", "gtc_55");
  for (mp_size_t n = 2; n < nmax && n < GTC_TABLE_N; n++)
    {
      double best[NA] = { 1e300, 1e300, 1e300, 1e300, 1e300, 1e300 };
      int ok[NA] = { 0 };
      ok[GTC_ALG_TB] = n <= GTC_TEXTBOOK_LIMIT;
      ok[GTC_ALG_22] = 1;
      ok[GTC_ALG_33] = cap >= 3 && (n == 3 || n >= 5);
      ok[GTC_ALG_44] = cap >= 4 && n >= GTC_44_MIN;
      ok[GTC_ALG_55] = cap >= 5 && n >= GTC_55_MIN;
      fill_pool (in, pool, n);
      /* Interleaved trials, the minimum per algorithm.  */
      for (int t = 0; t < trials; t++)
	for (int a = 0; a < NA; a++)
	  if (ok[a])
	    {
	      int save = trials;
	      double cy;
	      set_alg (c, cap, n, a);
	      trials = 1;
	      cy = cycles (c, n, in, pool, P, Q);
	      trials = save;
	      if (cy < best[a])
		best[a] = cy;
	    }
      int w = GTC_ALG_22, d;
      for (int a = 0; a < NA; a++)
	if (ok[a] && best[a] < best[w])
	  w = a;
      set_alg (c, cap, n, w);
      ch[n] = name[w];
      d = gtc_alg_thresh (&c->cfg, cap, n);
      printf ("%5ld", (long) n);
      for (int a = 0; a < NA; a++)
	if (a == GTC_ALG_LEAF)
	  continue;
	else if (ok[a])
	  printf (" %10.0f", best[a]);
	else
	  printf (" %10s", "-");
      printf ("  %c%s\n", name[w], w != d ? " (table)" : "");
      fflush (stdout);
    }
  table_string (str, sizeof str, ch, nmax < GTC_TABLE_N ? nmax : GTC_TABLE_N);
  printf ("# gtc_table[GTC_CAP (%d)] \"%s\"\n", cap, str);
  free (in);
  free (P);
  free (Q);
  gtc_ctx_free (c);
}

static void
fill_pool (mp_ptr in, int pool, mp_size_t n)
{
  for (int i = 0; i < pool * 4; i++)
    {
      mpz_t z;
      mpz_init (z);
      mpz_urandomb (z, rs, n * GMP_NUMB_BITS);
      mpn_zero (in + i * n, n);
      mpn_copyi (in + i * n, mpz_limbs_read (z), mpz_size (z));
      mpz_clear (z);
    }
}

/* The size after the last one at which the slower algorithm still wins
   (the conservative end of a noisy crossover), ns[0] if the faster one
   always wins, 0 if it does not win at the largest size.  */
static mp_size_t
crossover (const mp_size_t *ns, const double *ratio, int m)
{
  int i = m - 1;
  while (i >= 0 && ratio[i] < 1)
    i--;
  return i < 0 ? ns[0] : i == m - 1 ? 0 : ns[i + 1];
}

/* The rational thresholds over the Gaussian ones K and G.  */
static gtc_ctx *
ctx_rx (mp_size_t r, mp_size_t t, mp_size_t x, mp_size_t g, mp_size_t k)
{
  gtc_config cfg;
  gtc_config_default (&cfg);
  cfg.r2gtc33_thresh = r;
  cfg.r2gtc22_thresh = t;
  cfg.gmp22_thresh = x;
  cfg.gtc33_thresh = g;
  cfg.gtc22_thresh = k;
  cfg.gtc_km_thresh = km;
  cfg.gtc44_thresh = g4;
  cfg.gtc55_thresh = g5;
  for (int j = 0; j < 4; j++)
    cfg.gtc_table[j] = cfg.r2gtc_table[j] = NULL;
  return gtc_ctx_new (&cfg);
}

/* xfix: GMP's Karatsuba from X = xfix (GTC_NEVER: not at all), 0: measure
   it.  */
static mp_size_t xfix;

static void
tune_rational (mp_size_t G, mp_size_t K, mp_size_t tmax, mp_size_t rmax, mp_size_t tfix)
{
  const int pool = 64;
  mp_size_t nmax = tmax > rmax ? tmax : rmax, ns[4096];
  double rt[4096], rr[4096];
  int m = 0;
  mp_ptr in = malloc (pool * 4 * nmax * sizeof (mp_limb_t));
  mp_ptr P = malloc ((2 * nmax + 4) * sizeof (mp_limb_t));
  gtc_ctx *text = ctx_rx (GTC_NEVER, GTC_NEVER, GTC_NEVER, G, K);
  mp_size_t X = xfix;

  rational = 1;
  /* X: GMP's Karatsuba over the textbook product.  */
  if (X == 0)
    {
      printf ("\n%5s %10s %10s %7s   (rational; Gaussian K = %ld, G = %ld)\n",
	      "n", "textbook", "GMP K2", "ratio", (long) K, (long) G);
      for (mp_size_t n = 4; n <= tmax; n += n < 60 ? 1 : n < 150 ? 4 : 10)
	{
	  gtc_ctx *gk = ctx_rx (GTC_NEVER, GTC_NEVER, n, G, K);
	  fill_pool (in, pool, n);
	  double b = cycles (text, n, in, pool, P, NULL);
	  double k = cycles (gk, n, in, pool, P, NULL);
	  ns[m] = n;
	  rt[m++] = k / b;
	  printf ("%5ld %10.0f %10.0f %7.3f\n", (long) n, b, k, k / b);
	  fflush (stdout);
	  gtc_ctx_free (gk);
	}
      X = crossover (ns, rt, m);
      printf ("# GMP's Karatsuba threshold X = %ld%s\n", (long) X,
	      X ? " (measured)" : ": never wins");
      if (X == 0)
	X = GTC_NEVER;
      m = 0;
    }
  else
    printf ("\n# GMP's Karatsuba threshold X = %ld (given)\n",
	    (long) (X < GTC_NEVER ? X : 0));

  /* T: the Karatsuba-style reduction over what lies below it.  */
  gtc_ctx *belowt = ctx_rx (GTC_NEVER, GTC_NEVER, X, G, K);
  printf ("\n%5s %10s %10s %7s   (rational; Gaussian K = %ld, G = %ld)\n",
	  "n", "below T", "red. K2", "ratio", (long) K, (long) G);
  for (mp_size_t n = 4; n <= tmax; n += n < 60 ? 1 : n < 150 ? 4 : 10)
    {
      gtc_ctx *k2 = ctx_rx (GTC_NEVER, n, X, G, K);
      fill_pool (in, pool, n);
      double b = cycles (belowt, n, in, pool, P, NULL);
      double k = cycles (k2, n, in, pool, P, NULL);
      ns[m] = n;
      rt[m++] = k / b;
      printf ("%5ld %10.0f %10.0f %7.3f\n", (long) n, b, k, k / b);
      fflush (stdout);
      gtc_ctx_free (k2);
    }
  mp_size_t T = tfix ? tfix : crossover (ns, rt, m);
  printf ("# rational Karatsuba-style threshold T = %ld%s\n", (long) T,
	  tfix ? " (given)" : T ? " (measured)" : ": never wins");
  if (T == 0)
    T = GTC_NEVER;

  /* R: the Toom-3 style reduction over what lies below it.  */
  gtc_ctx *below = ctx_rx (GTC_NEVER, T, X, G, K);
  printf ("\n%5s %10s %10s %7s\n", "n", "below R", "red. TC3", "ratio");
  m = 0;
  for (mp_size_t n = 5; n <= rmax; n += n < 60 ? 1 : n < 150 ? 4 : n < 400
       ? 10 : 25)
    {
      gtc_ctx *tc3 = ctx_rx (n, T, X, G, K);
      fill_pool (in, pool, n);
      double b = cycles (below, n, in, pool, P, NULL);
      double t = cycles (tc3, n, in, pool, P, NULL);
      ns[m] = n;
      rr[m++] = t / b;
      printf ("%5ld %10.0f %10.0f %7.3f\n", (long) n, b, t, t / b);
      fflush (stdout);
      gtc_ctx_free (tc3);
    }
  mp_size_t R = crossover (ns, rr, m);
  if (R)
    printf ("# rational Toom-3 style threshold R = %ld over T = %ld, "
	    "X = %ld\n", (long) R, (long) (T < GTC_NEVER ? T : 0),
	    (long) (X < GTC_NEVER ? X : 0));
  else
    printf ("# rational Toom-3 style threshold R: never wins up to %ld "
	    "(over T = %ld, X = %ld)\n", (long) rmax,
	    (long) (T < GTC_NEVER ? T : 0), (long) (X < GTC_NEVER ? X : 0));
  rational = 0;
  gtc_ctx_free (belowt);
  gtc_ctx_free (below);
  gtc_ctx_free (text);
  free (in);
  free (P);
}

/* --rat-table: cycles per level of r2gtc_55, r2gtc_44, r2gtc_33,
   r2gtc_22 ('5' to '2'), or per product of GMP's Karatsuba ('g') or the
   textbook product ('t') at size n in context c.  */
static double
cycles_rlevel (gtc_ctx *c, int alg, mp_size_t n, mp_ptr in, int pool,
	       mp_ptr P)
{
  long reps = 200000 / (n * n + 10) + 20;
  double best = 1e300;
  for (int t = 0; t < trials; t++)
    {
      long long cy;
      ioctl (fdc, PERF_EVENT_IOC_RESET, 0);
      ioctl (fdc, PERF_EVENT_IOC_ENABLE, 0);
      for (long i = 0; i < reps; i++)
	{
	  mp_ptr x = in + (i & (pool - 1)) * 4 * n;
	  switch (alg)
	    {
	    case '5': r2gtc_55 (c, P, x, x + n, n); break;
	    case '4': r2gtc_44 (c, P, x, x + n, n); break;
	    case '3': r2gtc_33 (c, P, x, x + n, n); break;
	    case '2': r2gtc_22 (c, P, x, x + n, n); break;
	    case 'g': gmp_22 (c, P, x, x + n, n); break;
	    default: mpn_mul_basecase (P, x, n, x + n, n); break;
	    }
	}
      ioctl (fdc, PERF_EVENT_IOC_DISABLE, 0);
      if (read (fdc, &cy, sizeof cy) == sizeof cy && (double) cy / reps < best)
	best = (double) cy / reps;
      arena_fit (c);
    }
  return best;
}

static double margin = 0.003;

static void
tune_rat_table (mp_size_t lo, mp_size_t hi)
{
  const int pool = 64;
  /* the candidates, by R2GTC_ALG_* */
  static const char name[] = "tg2345";
  enum { NA = 6 };
  static char ch[GTC_TABLE_N], str[16384];
  mp_ptr in = malloc (pool * 4 * (hi + 1) * sizeof (mp_limb_t));
  mp_ptr P = malloc ((2 * hi + 4) * sizeof (mp_limb_t));
  gtc_config cfg;

  gtc_config_default (&cfg);
  tables_given (&cfg);
  if (lo < 2)
    lo = 2;
  if (hi >= GTC_TABLE_N)
    hi = GTC_TABLE_N - 1;
  memset (ch, 0, sizeof ch);
  printf ("\n# r2gtc_table[GTC_CAP (%d)] (the rational products of r2gtc_%d%d) "
	  "over X = %ld, T = %ld, R = %ld, R4 = %ld, R5 = %ld, margin %.4f\n"
	  "%5s %10s %10s %10s %10s %10s %10s  choice\n", cap, cap, cap,
	  (long) cfg.gmp22_thresh, (long) cfg.r2gtc22_thresh,
	  (long) cfg.r2gtc33_thresh, (long) cfg.r2gtc44_thresh,
	  (long) cfg.r2gtc55_thresh, margin, "n", "textbook", "GMP K2",
	  "r2gtc_22", "r2gtc_33", "r2gtc_44", "r2gtc_55");
  for (mp_size_t n = lo; n <= hi; n++)
    {
      gtc_ctx *c;
      double t[NA], x;
      int ok[NA], dflt, best = -1;
      ok[R2GTC_ALG_TB] = 1;
      ok[R2GTC_ALG_GMP22] = n >= GTC_GMP22_MIN;
      ok[R2GTC_ALG_22] = n >= 2;
      ok[R2GTC_ALG_33] = cap >= 3 && (n == 3 || n >= 5);
      ok[R2GTC_ALG_44] = cap >= 4 && n >= GTC_44_MIN;
      ok[R2GTC_ALG_55] = cap >= 5 && n >= GTC_55_MIN;
      table_string (str, sizeof str, ch, n);
      cfg.r2gtc_table[GTC_CAP (cap)] = str;
      c = gtc_ctx_new (&cfg);
      fill_pool (in, pool, n);
      for (int a = 0; a < NA; a++)
	t[a] = 1e300;
      for (int r = 0; r < 3; r++)
	for (int a = 0; a < NA; a++)
	  if (ok[a] && (x = cycles_rlevel (c, name[a], n, in, pool, P)) < t[a])
	    t[a] = x;
      gtc_ctx_free (c);
      dflt = r2gtc_alg_thresh (&cfg, cap, n);
      for (int a = 0; a < NA; a++)
	if (ok[a] && (best < 0 || t[a] < t[best]))
	  best = a;
      if (best != dflt && t[best] < t[dflt] * (1 - margin))
	ch[n] = name[best];
      printf ("%5ld", (long) n);
      for (int a = 0; a < NA; a++)
	if (ok[a])
	  printf (" %10.0f", t[a]);
	else
	  printf (" %10s", "-");
      printf ("  %c%s\n", ch[n] ? ch[n] : name[dflt], ch[n] ? " (table)" : "");
      fflush (stdout);
    }
  table_string (str, sizeof str, ch, hi + 1);
  printf ("# r2gtc_table[GTC_CAP (%d)] \"%s\"\n", cap, str);
  free (in);
  free (P);
}

/* G4: the Gaussian Toom-4 (its products: Toom-3 from G, Karatsuba from K,
   the base case) against what lies below it, up to g4max.  */
static mp_size_t
tune_toom4 (mp_size_t G, mp_size_t K, mp_size_t g4max)
{
  const int pool = 64;
  mp_size_t ns[4096];
  double r[4096];
  int m = 0;
  mp_ptr in = malloc (pool * 4 * g4max * sizeof (mp_limb_t));
  mp_ptr P = malloc ((2 * g4max + 4) * sizeof (mp_limb_t));
  mp_ptr Q = malloc ((2 * g4max + 4) * sizeof (mp_limb_t));
  gtc_ctx *below;

  g4 = GTC_NEVER;
  below = ctx (G, K);
  printf ("\n%5s %10s %10s %7s   (Toom-3 from G = %ld, Karatsuba from K = "
	  "%ld)\n", "n", "below G4", "Toom-4", "ratio", (long) G, (long) K);
  for (mp_size_t n = 40; n <= g4max; n += n < 120 ? 4 : n < 400 ? 10 : 25)
    {
      gtc_ctx *tc4;
      g4 = n;
      tc4 = ctx (G, K);
      g4 = GTC_NEVER;
      fill_pool (in, pool, n);
      double b = cycles (below, n, in, pool, P, Q);
      double t = cycles (tc4, n, in, pool, P, Q);
      ns[m] = n;
      r[m++] = t / b;
      printf ("%5ld %10.0f %10.0f %7.3f\n", (long) n, b, t, t / b);
      fflush (stdout);
      gtc_ctx_free (tc4);
    }
  mp_size_t G4 = crossover (ns, r, m);
  printf ("# Toom-4 threshold G4 = %ld%s\n", (long) G4,
	  G4 ? " (measured)" : ": Toom-4 never wins for good");
  gtc_ctx_free (below);
  free (in);
  free (P);
  free (Q);
  return G4;
}

/* G5: the Gaussian Toom-5 (its products: Toom-4 from G4, Toom-3 from G,
   Karatsuba from K, the base case) against what lies below it, up to
   g5max.  */
static mp_size_t
tune_toom5 (mp_size_t G4, mp_size_t G, mp_size_t K, mp_size_t g5max)
{
  const int pool = 64;
  mp_size_t ns[4096];
  double r[4096];
  int m = 0;
  mp_ptr in = malloc (pool * 4 * g5max * sizeof (mp_limb_t));
  mp_ptr P = malloc ((2 * g5max + 4) * sizeof (mp_limb_t));
  mp_ptr Q = malloc ((2 * g5max + 4) * sizeof (mp_limb_t));
  gtc_ctx *below;

  g4 = G4;
  g5 = GTC_NEVER;
  below = ctx (G, K);
  printf ("\n%5s %10s %10s %7s   (Toom-4 from G4 = %ld, Toom-3 from G = %ld, "
	  "Karatsuba from K = %ld)\n", "n", "below G5", "Toom-5", "ratio",
	  (long) G4, (long) G, (long) K);
  for (mp_size_t n = 40; n <= g5max; n += n < 120 ? 4 : n < 400 ? 10 : 25)
    {
      gtc_ctx *tc5;
      g5 = n;
      tc5 = ctx (G, K);
      g5 = GTC_NEVER;
      fill_pool (in, pool, n);
      double b = cycles (below, n, in, pool, P, Q);
      double t = cycles (tc5, n, in, pool, P, Q);
      ns[m] = n;
      r[m++] = t / b;
      printf ("%5ld %10.0f %10.0f %7.3f\n", (long) n, b, t, t / b);
      fflush (stdout);
      gtc_ctx_free (tc5);
    }
  mp_size_t G5 = crossover (ns, r, m);
  printf ("# Toom-5 threshold G5 = %ld%s\n", (long) G5,
	  G5 ? " (measured)" : ": Toom-5 never wins for good");
  gtc_ctx_free (below);
  free (in);
  free (P);
  free (Q);
  return G5;
}

int
main (int argc, char **argv)
{
  mp_size_t kmax = 60, gmax = 400, kfix = 0, gfix = 0, table_n = 0;
  mp_size_t tmax = 200, rmax = 1500, tfix = 0, kmfix = 0, g4max = 0;
  int pin = -1, do_rational = 0, toom4_only = 0, r2gtc_table = 0;
  int toom5_only = 0;
  mp_size_t g4fix = 0, g5max = 0;
  mp_size_t rt_lo = 0, rt_hi = GTC_TABLE_N - 1;
  for (int i = 1; i < argc; i++)
    {
      if (!strcmp (argv[i], "--pin") && i + 1 < argc)
	pin = atoi (argv[++i]);
      else if (!strcmp (argv[i], "--kmax") && i + 1 < argc)
	kmax = atol (argv[++i]);
      else if (!strcmp (argv[i], "--gmax") && i + 1 < argc)
	gmax = atol (argv[++i]);
      else if (!strcmp (argv[i], "--kara") && i + 1 < argc)
	kfix = atol (argv[++i]);
      else if (!strcmp (argv[i], "--toom") && i + 1 < argc)
	gfix = atol (argv[++i]);
      else if (!strcmp (argv[i], "--km") && i + 1 < argc)
	kmfix = km = atol (argv[++i]);
      else if (!strcmp (argv[i], "--rat-kara") && i + 1 < argc)
	tfix = atol (argv[++i]);
      else if (!strcmp (argv[i], "--gmp-kara") && i + 1 < argc)
	{
	  xfix = strcmp (argv[++i], "never") ? atol (argv[i]) : GTC_NEVER;
	  if (xfix <= 0)
	    xfix = GTC_NEVER;
	}
      else if (!strcmp (argv[i], "--tmax") && i + 1 < argc)
	tmax = atol (argv[++i]);
      else if (!strcmp (argv[i], "--rmax") && i + 1 < argc)
	rmax = atol (argv[++i]);
      else if (!strcmp (argv[i], "--g4max") && i + 1 < argc)
	g4max = atol (argv[++i]);
      else if (!strcmp (argv[i], "--toom4-only"))
	toom4_only = 1;
      else if (!strcmp (argv[i], "--g4") && i + 1 < argc)
	g4fix = atol (argv[++i]);
      else if (!strcmp (argv[i], "--g5max") && i + 1 < argc)
	g5max = atol (argv[++i]);
      else if (!strcmp (argv[i], "--toom5-only"))
	toom5_only = 1;
      else if (!strcmp (argv[i], "--cap") && i + 1 < argc)
	{
	  cap = atoi (argv[++i]);
	  if (cap < 2 || cap > 5)
	    {
	      puts ("--cap: 2 to 5");
	      return 2;
	    }
	}
      else if ((!strcmp (argv[i], "--set-table")
		|| !strcmp (argv[i], "--set-rat-table")) && i + 2 < argc)
	{
	  int I = atoi (argv[i + 1]);
	  if (I < 2 || I > 5)
	    {
	      puts ("--set-table, --set-rat-table: a cap of 2 to 5");
	      return 2;
	    }
	  (argv[i][6] == 'r' ? set_rtab : set_gtab)[GTC_CAP (I)] = argv[i + 2];
	  i += 2;
	}
      else if (!strcmp (argv[i], "--rational"))
	do_rational = 1;
      else if (!strcmp (argv[i], "--rational-only"))
	do_rational = 2;
      else if (!strcmp (argv[i], "--trials") && i + 1 < argc)
	trials = atoi (argv[++i]);
      else if (!strcmp (argv[i], "--table"))
	table_n = i + 1 < argc && argv[i + 1][0] != '-' ? atol (argv[++i])
	  : GTC_TABLE_N;
      else if (!strcmp (argv[i], "--rat-table"))
	{
	  r2gtc_table = 1;
	  if (i + 1 < argc && argv[i + 1][0] != '-')
	    {
	      char *e;
	      rt_lo = strtol (argv[++i], &e, 10);
	      if (*e == ':')
		rt_hi = strtol (e + 1, NULL, 10);
	    }
	}
      else if (!strcmp (argv[i], "--margin") && i + 1 < argc)
	margin = atof (argv[++i]);
      else
	{
	  puts ("usage: tune_gauss [--pin CPU] [--kmax N] [--gmax N] "
		"[--kara K] [--toom G] [--km M] [--trials T]\n"
		"                  [--rational | --rational-only] [--rat-kara T] "
		"[--gmp-kara X|never] [--tmax N] [--rmax N]\n"
		"                  [--g4max N] [--toom4-only] [--toom5-only] "
		"[--g4 G4] [--g5max N]\n"
		"                  [--table [N] | --rat-table [LO:HI]] [--cap I] "
		"[--margin M]\n"
		"                  [--set-table I STRING] [--set-rat-table I "
		"STRING]\n"
		"  --table [N]  the fastest algorithm per size of gtc_ii's "
		"products below N\n"
		"               (default and at most 1024), for "
		"gtc_config.gtc_table[GTC_CAP (I)]\n"
		"  --rat-table [LO:HI]  the fastest algorithm per size of "
		"r2gtc_ii's rational\n"
		"               products (default 2 to 1023), for "
		"gtc_config.r2gtc_table[GTC_CAP (I)]\n"
		"  --cap I      the cap i of those (2 to 5, default 5)\n"
		"  --set-table I STRING, --set-rat-table I STRING  the table of "
		"the cap I\n"
		"               to take instead of the default (\"\": none)\n"
		"  --margin M   --rat-table leaves the thresholds' choice only "
		"where another\n"
		"               is faster by more than M (default 0.003)\n"
		"  --g4max N    then the gtc_44 threshold G4 over K and G, up "
		"to N limbs\n"
		"  --toom4-only that alone (with --kara and --toom; --g4max, "
		"default 800)\n"
		"  --toom5-only the gtc_55 threshold G5 over K, G and G4 (--kara, "
		"--toom,\n"
		"               --g4, default the library's; --g5max, default "
		"1500)\n"
		"  --kara K     use K as the gtc_22 threshold instead of the "
		"measured one\n"
		"  --toom G     use G as the gtc_33 threshold instead of the "
		"measured one\n"
		"  --km M       use M as the threshold of the Gaussian Karatsuba "
		"over {1, i}\n"
		"               instead of the measured one\n"
		"  --rational   then the rational thresholds X, T and R over K "
		"and G\n"
		"  --rational-only  those alone (with --kara and --toom)\n"
		"  --rat-kara T use T instead of the measured one for R\n"
		"  --gmp-kara X use X (or never) instead of the measured "
		"threshold of GMP's\n"
		"               Karatsuba for T and R");
	  return 2;
	}
    }
  if (pin >= 0)
    {
      cpu_set_t set;
      CPU_ZERO (&set);
      CPU_SET (pin, &set);
      sched_setaffinity (0, sizeof set, &set);
    }
  {
    struct perf_event_attr pe;
    memset (&pe, 0, sizeof pe);
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof pe;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    fdc = (int) syscall (SYS_perf_event_open, &pe, 0, -1, -1, 0);
    if (fdc < 0)
      {
	perror ("perf_event_open");
	return 1;
      }
  }
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 20260918);
  printf ("# libgtc %s, GMP %s\n# cycles per product of n-limb "
	  "components (minimum of %d trials)\n", GTC_VERSION_STRING,
	  gmp_version, trials);
  printf ("# base case: the textbook product, up to %d limbs\n",
	  GTC_TEXTBOOK_LIMIT);
  if (r2gtc_table)
    {
      tune_rat_table (rt_lo, rt_hi);
      return 0;
    }
  if (toom5_only)
    {
      gtc_config d;
      gtc_config_default (&d);
      tune_toom5 (g4fix ? g4fix : d.gtc44_thresh,
		  gfix ? gfix : d.gtc33_thresh,
		  kfix ? kfix : d.gtc22_thresh, g5max ? g5max : 1500);
      return 0;
    }
  if (toom4_only)
    {
      if (!kfix || !gfix)
	{
	  puts ("--toom4-only needs --kara K and --toom G");
	  return 2;
	}
      tune_toom4 (gfix, kfix, g4max ? g4max : 800);
      return 0;
    }
  if (do_rational == 2)
    {
      if (!kfix || !gfix)
	{
	  puts ("--rational-only needs --kara K and --toom G");
	  return 2;
	}
      tune_rational (gfix, kfix, tmax, rmax, tfix);
      return 0;
    }

  if (table_n)
    {
      tune_table (table_n);
      return 0;
    }

  mp_size_t ns[4096];
  double rk[4096] = { 0 }, rg[4096] = { 0 }, rgb[4096] = { 0 };
  int m;
  const int pool = 64;
  mp_size_t nmax = kmax > gmax ? kmax : gmax;
  if (nmax < GTC_TEXTBOOK_LIMIT)	/* the textbook sweep's sizes */
    nmax = GTC_TEXTBOOK_LIMIT;
  mp_ptr in = malloc (pool * 4 * nmax * sizeof (mp_limb_t));
  mp_ptr P = malloc ((2 * nmax + 4) * sizeof (mp_limb_t));
  mp_ptr Q = malloc ((2 * nmax + 4) * sizeof (mp_limb_t));
  gtc_ctx *base = ctx (1 << 30, 1 << 30);

  (void) kmfix;			/* (only where there is the kernel) */
#ifdef GTC_KM_ASM
  /* KM: the Gaussian Karatsuba over {1, i} against the textbook product,
     both as the base case.  */
    {
      mp_size_t save = km, nk[GTC_KM_MAX + 1];
      double rm[GTC_KM_MAX + 1];
      int mk = 0;
      printf ("\n%5s %10s %10s %7s   (the base case)\n", "n", "textbook",
	      "{1,i} kara", "ratio");
      for (mp_size_t n = 2; n <= GTC_KM_MAX; n++)
	{
	  km = GTC_NEVER;
	  gtc_ctx *tb = ctx (1 << 30, 1 << 30);
	  km = n;
	  gtc_ctx *k3 = ctx (1 << 30, 1 << 30);
	  fill_pool (in, pool, n);
	  double b = cycles (tb, n, in, pool, P, Q);
	  double k = cycles (k3, n, in, pool, P, Q);
	  nk[mk] = n;
	  rm[mk++] = k / b;
	  printf ("%5ld %10.0f %10.0f %7.3f\n", (long) n, b, k, k / b);
	  fflush (stdout);
	  gtc_ctx_free (tb);
	  gtc_ctx_free (k3);
	}
      km = kmfix ? save : crossover (nk, rm, mk);
      printf ("# {1, i} Karatsuba threshold KM = %ld%s\n", (long) km,
	      kmfix ? " (given)" : km ? " (measured)" : ": never wins");
      if (km == 0)
	km = GTC_NEVER;
      gtc_ctx_free (base);
      base = ctx (1 << 30, 1 << 30);
    }
#endif

  /* K: Karatsuba over the base case, against the base case.  */
  printf ("\n%5s %10s %10s %7s\n", "n", "base", "Karatsuba", "ratio");
  m = 0;
  for (mp_size_t n = 2; n <= (kmax > GTC_TEXTBOOK_LIMIT
			      ? GTC_TEXTBOOK_LIMIT : kmax); n++)
    {
      gtc_ctx *kara = ctx (1 << 30, n);
      fill_pool (in, pool, n);
      double b = cycles (base, n, in, pool, P, Q);
      double k = cycles (kara, n, in, pool, P, Q);
      ns[m] = n;
      rk[m++] = k / b;
      printf ("%5ld %10.0f %10.0f %7.3f\n", (long) n, b, k, k / b);
      fflush (stdout);
      gtc_ctx_free (kara);
    }
  mp_size_t K = kfix ? kfix : crossover (ns, rk, m);
  if (K == 0)
    {
      K = GTC_TEXTBOOK_LIMIT + 1;
      printf ("# Karatsuba threshold K = %ld: Karatsuba never wins up to the "
	      "textbook product's largest size\n", (long) K);
    }
  else
    printf ("# Karatsuba threshold K = %ld%s\n", (long) K,
	    kfix ? " (given)" : K ? " (measured)"
	    : ": Karatsuba never wins; base case only");
  if (K == 0)
    K = 1 << 30;

  /* G: Toom-3 over what lies below, against it; and over the base case
     alone.  */
  printf ("\n%5s %10s %10s %7s %10s %7s\n", "n", "below G", "Toom-3", "ratio",
	  "Toom-3/b", "ratio");
  gtc_ctx *below = ctx (1 << 30, K);
  m = 0;
  for (mp_size_t n = 5; n <= gmax; n += n < 40 ? 1 : n < 120 ? 4 : 10)
    {
      gtc_ctx *tc3 = ctx (n, K), *tc3b = ctx (n, 1 << 30);
      fill_pool (in, pool, n);
      double bl = cycles (below, n, in, pool, P, Q);
      double t = cycles (tc3, n, in, pool, P, Q);
      double bb = cycles (base, n, in, pool, P, Q);
      double tb = cycles (tc3b, n, in, pool, P, Q);
      ns[m] = n;
      rg[m] = t / bl;
      rgb[m++] = tb / bb;
      printf ("%5ld %10.0f %10.0f %7.3f %10.0f %7.3f\n", (long) n, bl, t, t / bl,
	      tb, tb / bb);
      fflush (stdout);
      gtc_ctx_free (tc3);
      gtc_ctx_free (tc3b);
    }
  mp_size_t G = crossover (ns, rg, m), GB = crossover (ns, rgb, m);
  printf ("# Toom-3 threshold G = %ld over Karatsuba (K = %ld); %ld over the "
	  "base case alone\n", (long) G, (long) (K < (1 << 30) ? K : 0), (long) GB);
  if (g4max)
    tune_toom4 (gfix ? gfix : G ? G : GTC_NEVER, K, g4max);
  if (do_rational)
    tune_rational (gfix ? gfix : G ? G : GTC_NEVER, K, tmax, rmax, tfix);
  return 0;
}
