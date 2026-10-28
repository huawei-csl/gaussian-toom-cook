/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   Benchmark: libgtc's Gaussian multiplication against three real products
   by GMP.

   The comparisons (--compare), each algorithm at the top and its own
   algorithms below it:

     g33    gtc_33 against three products by GMP's Toom-3, mpn_toom33_mul,
            up to 3,000 limbs per component;
     g44    gtc_44 against three products by GMP's Toom-4, mpn_toom44_mul,
            26 to 4,000;
     g55    gtc_55 against three products by GMP's Toom-4 and by its
            Toom-6.5, mpn_toom6h_mul (GMP has no Toom-5), 46 to 5,000.

   One level of gtc_ii at the top, then its products below it by gtc's
   tables and thresholds within the cap i (gtc_jj, j <= i, and the Gaussian
   base case).  GMP's side is the 3M formula with differences,

     (a + bi)(c + di) = ac - bd + (ac + bd - (a - b)(c - d)) i,

   on components made non-negative by a unit first (x = i^e (a + bi) with
   a, b >= 0, a sign and a swap of the components, free): three n x n
   products, each one level of mpn_toomii_mul with GMP's algorithms up to
   itself below it by GMP's thresholds, and GMP's mpn additions.  Inputs and
   outputs as gtc_mul_n_ctx's: sign-magnitude components, n limbs in, 2n+1
   out, random signs.

   --methods names the methods instead, the first the reference of the
   ratios: gmp3m_toom33, gmp3m_toom44, gmp3m_toom6h, gtc_33, gtc_44,
   gtc_55.  Run bench_gauss --help for the options.  */

#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gtc.h"
#include "gmp_decls.h"

/* GMP's scratch for mpn_toom6h_mul depends on its tuned thresholds; this
   bounds it for any threshold up to 2000 limbs.  */
#define MPN_TOOM6H_MUL_ITCH_BOUND(n) (4 * (n) + 8192)

#ifndef GTC_BENCH_SKEW_DEFAULT
#define GTC_BENCH_SKEW_DEFAULT 0
#endif

enum
{ M_3M33, M_3M44, M_3M6H, M_GTC_33, M_GTC_44, M_GTC_55, M_N };

/* By kind: the name, the smallest size it is measured at, whether it is
   gtc's.  GMP's Toom-3 from 17 limbs, its Toom-6.5 from 46 (the smallest
   its algorithm allows), its Toom-4 from 26, where its recursion's last
   part has at least 5 limbs.  */
static const struct
{
  const char *name;
  long min_n;
  int gtc;
} KIND[M_N] = {
  { "gmp3m_toom33", 17, 0 }, { "gmp3m_toom44", 26, 0 },
  { "gmp3m_toom6h", 46, 0 }, { "gtc_33", 5, 1 }, { "gtc_44", 13, 1 },
  { "gtc_55", 17, 1 } };

/* The presets of --compare: the methods (the first the reference) and the
   sizes, geometric from MIN to MAX.  */
static const struct
{
  const char *name, *methods;
  long min, max;
} COMPARE[] = {
  { "g33", "gmp3m_toom33,gtc_33", 20, 3000 },
  { "g44", "gmp3m_toom44,gtc_44", 26, 4000 },
  { "g55", "gmp3m_toom44,gmp3m_toom6h,gtc_55", 46, 5000 } };

typedef struct
{
  int kind;
  const char *name;
  gtc_ctx *ctx;
  double *trial;		/* per-call time of each trial, ns */
  long iters;
} method;

/* The inputs of one size: a pool of POOL Gaussian pairs, four components of
   n limbs each and their signs, cycled through by the timing loops (one
   repeated input would let the branch predictor learn its branches).  The
   pool stays within 256 KB.  */
static mp_ptr XPOOL;
static unsigned char *SPOOL;
static long POOL, CUR;
static mp_size_t PN;
static mp_srcptr XR, XI, YR, YI;
static int XRN, XIN, YRN, YIN;

/* Output and GMP's work areas.  */
static mp_ptr ZR, ZI, P, Q, R, U, V, SCRATCH;
static int ZRN, ZIN;

static void
use_input (long j)
{
  CUR = j;
  XR = XPOOL + 4 * j * PN;
  XI = XR + PN;
  YR = XI + PN;
  YI = YR + PN;
  XRN = SPOOL[4 * j] & 1;
  XIN = SPOOL[4 * j + 1] & 1;
  YRN = SPOOL[4 * j + 2] & 1;
  YIN = SPOOL[4 * j + 3] & 1;
}

static long
pool_size (mp_size_t n)
{
  long p = 1;
  while (p < 256 && 2 * p * 4 * n * (long) sizeof (mp_limb_t) <= 256 * 1024)
    p *= 2;
  return p;
}

static double
now_ns (void)
{
  struct timespec ts;
  clock_gettime (CLOCK_MONOTONIC_RAW, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* x = xr + xi i = i^e (a + b i) with a, b >= 0: the magnitudes, possibly
   swapped, and e.  */
static int
unit_split (mp_srcptr *a, mp_srcptr *b, mp_srcptr xr, int xrn, mp_srcptr xi,
	    int xin)
{
  if (xrn == xin)
    {
      *a = xr;
      *b = xi;
      return xrn ? 2 : 0;
    }
  /* -|r| + |i| i = i (|i| + |r| i), |r| - |i| i = -i (|i| + |r| i) */
  *a = xi;
  *b = xr;
  return xrn ? 1 : 3;
}

/* {r, n} = |{u, n} - {v, n}|; returns 1 if u < v.  */
static int
abs_sub_n (mp_ptr r, mp_srcptr u, mp_srcptr v, mp_size_t n)
{
  if (mpn_cmp (u, v, n) >= 0)
    {
      mpn_sub_n (r, u, v, n);
      return 0;
    }
  mpn_sub_n (r, v, u, n);
  return 1;
}

static void
real_mul (int kind, mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  switch (kind)
    {
    case M_3M33: mpn_toom33_mul (rp, ap, n, bp, n, SCRATCH); break;
    case M_3M44: mpn_toom44_mul (rp, ap, n, bp, n, SCRATCH); break;
    default: mpn_toom6h_mul (rp, ap, n, bp, n, SCRATCH); break;
    }
}

/* The Gaussian product by three real products: into ZR, ZI (2n+1 limbs),
   signs ZRN, ZIN.  */
static void
gmp3m (int kind, mp_size_t n)
{
  mp_srcptr a, b, c, d;
  int e = (unit_split (&a, &b, XR, XRN, XI, XIN)
	   + unit_split (&c, &d, YR, YRN, YI, YIN)) & 3;
  /* (a + b i)(c + d i) = re + im i, im = ad + bc >= 0; times i^e:
     e = 0: re + im i, 1: -im + re i, 2: -re - im i, 3: im - re i.  */
  mp_ptr re = e & 1 ? ZI : ZR, im = e & 1 ? ZR : ZI;
  int ren, s;

  real_mul (kind, P, a, c, n);
  real_mul (kind, Q, b, d, n);
  s = abs_sub_n (U, a, b, n) ^ abs_sub_n (V, c, d, n);
  real_mul (kind, R, U, V, n);	/* (a - b)(c - d) = (-1)^s R */

  ren = mpn_cmp (P, Q, 2 * n) < 0;
  if (ren)
    mpn_sub_n (re, Q, P, 2 * n);
  else
    mpn_sub_n (re, P, Q, 2 * n);
  re[2 * n] = 0;
  im[2 * n] = mpn_add_n (im, P, Q, 2 * n);
  if (s)
    mpn_add (im, im, 2 * n + 1, R, 2 * n);
  else
    mpn_sub (im, im, 2 * n + 1, R, 2 * n);

  switch (e)
    {
    case 0: ZRN = ren; ZIN = 0; break;
    case 1: ZRN = 1; ZIN = ren; break;
    case 2: ZRN = !ren; ZIN = 1; break;
    default: ZRN = 0; ZIN = !ren; break;
    }
}

static void
run (method *m, mp_size_t n)
{
  switch (m->kind)
    {
    case M_GTC_33:
      gtc_33_ctx (m->ctx, ZR, &ZRN, ZI, &ZIN, XR, XRN, XI, XIN, YR, YRN, YI,
		  YIN, n);
      break;
    case M_GTC_44:
      gtc_44_ctx (m->ctx, ZR, &ZRN, ZI, &ZIN, XR, XRN, XI, XIN, YR, YRN, YI,
		  YIN, n);
      break;
    case M_GTC_55:
      gtc_55_ctx (m->ctx, ZR, &ZRN, ZI, &ZIN, XR, XRN, XI, XIN, YR, YRN, YI,
		  YIN, n);
      break;
    default:
      gmp3m (m->kind, n);
    }
}

static double
time_iters (method *m, mp_size_t n, long iters)
{
  double t0 = now_ns ();
  for (long i = 0; i < iters; i++)
    {
      run (m, n);
      use_input ((CUR + 1) & (POOL - 1));
    }
  return (now_ns () - t0) / iters;
}

/* Iterations such that one trial lasts about min_ns.  */
static long
calibrate (method *m, mp_size_t n, double min_ns)
{
  long iters = 1;
  double t;
  while ((t = time_iters (m, n, iters) * iters) < min_ns / 4
	 && iters < (1L << 30))
    iters *= 2;
  iters = (long) (iters * min_ns / (t > 0 ? t : 1)) + 1;
  return iters;
}

static int
cmp_double (const void *a, const void *b)
{
  double x = *(const double *) a, y = *(const double *) b;
  return (x > y) - (x < y);
}

static double
median (double *v, int k)
{
  double *s = malloc (k * sizeof *s);
  memcpy (s, v, k * sizeof *s);
  qsort (s, k, sizeof *s, cmp_double);
  double m = k % 2 ? s[k / 2] : (s[k / 2 - 1] + s[k / 2]) / 2;
  free (s);
  return m;
}

static double
minimum (double *v, int k)
{
  double m = v[0];
  for (int i = 1; i < k; i++)
    if (v[i] < m)
      m = v[i];
  return m;
}

static int
col_width (const method *m)
{
  int w = (int) strlen (m->name);
  return w > 20 ? w : 20;
}

/* Parse "a,b,c" into v (at most max entries); returns the count.  */
static int
parse_list (const char *s, long *v, int max)
{
  int k = 0;
  while (*s && k < max)
    {
      char *end;
      v[k++] = strtol (s, &end, 10);
      if (end == s)
	return 0;
      s = *end == ',' ? end + 1 : end;
    }
  return k;
}

static void
usage (void)
{
  puts ("usage: bench_gauss [options]\n"
	"gtc's Gaussian multiplication against three real products by GMP\n"
	"(ac - bd + (ac + bd - (a-b)(c-d)) i): each method at the top, its own\n"
	"algorithms below it.  Times are per Gaussian multiplication of n-limb\n"
	"components; ratio = time / time(first column).  The minimum over\n"
	"trials is the headline number, the median is also given.\n"
	"  --compare C           g33 | g44 | g55: gtc_33 against three\n"
	"                        mpn_toom33_mul up to 3,000 limbs, gtc_44\n"
	"                        against three mpn_toom44_mul up to 4,000, gtc_55\n"
	"                        against three mpn_toom44_mul and three\n"
	"                        mpn_toom6h_mul up to 5,000 (sizes growing by 10%)\n"
	"  --methods M1,M2,...   the methods instead, the first the reference:\n"
	"                        gmp3m_toom33, gmp3m_toom44, gmp3m_toom6h,\n"
	"                        gtc_33, gtc_44, gtc_55\n"
	"  --range MIN:MAX:K     K sizes (limbs), geometric in [MIN, MAX]\n"
	"  --sizes N1,N2,...     explicit sizes in limbs\n"
	"  --trials T            timing trials per method and size (default 9)\n"
	"  --min-time MS         length of one trial in ms (default 25)\n"
	"  --csv FILE            also write the results as CSV\n"
	"  --pin CPU             pin the process to one CPU\n"
	"  --quick               smoke test only: tiny trials, few sizes");
}

/* aarch64's /proc/cpuinfo has no model name: name the chip by its
   implementer and part, and the server from the DMI tables.  */
static void
aarch64_cpu_name (char *cpu, size_t size)
{
  char line[256], prod[128] = "";
  unsigned impl = 0, part = 0;
  FILE *f = fopen ("/proc/cpuinfo", "r");
  if (f)
    {
      while (fgets (line, sizeof line, f))
	{
	  if (!impl)
	    sscanf (line, "CPU implementer : %x", &impl);
	  if (!part)
	    sscanf (line, "CPU part : %x", &part);
	}
      fclose (f);
    }
  f = fopen ("/sys/class/dmi/id/product_name", "r");
  if (f)
    {
      if (fgets (prod, sizeof prod, f))
	prod[strcspn (prod, "\n")] = 0;
      fclose (f);
    }
  const char *sep = *prod ? " · " : "";
  if (impl == 0x48 && part == 0xd01)
    snprintf (cpu, size, "Kunpeng-920%s%s", sep, prod);
  else if (impl)
    snprintf (cpu, size, "CPU implementer 0x%x part 0x%x%s%s", impl, part,
	      sep, prod);
  else if (*prod)
    snprintf (cpu, size, "%s", prod);
}

static void
print_header (void)
{
  char line[256], cpu[256] = "unknown";
  FILE *f = fopen ("/proc/cpuinfo", "r");
  if (f)
    {
      while (fgets (line, sizeof line, f))
	if (strncmp (line, "model name", 10) == 0)
	  {
	    char *p = strchr (line, ':');
	    if (p)
	      {
		snprintf (cpu, sizeof cpu, "%s", p + 2);
		cpu[strcspn (cpu, "\n")] = 0;
	      }
	    break;
	  }
      fclose (f);
    }
  if (strcmp (cpu, "unknown") == 0)
    aarch64_cpu_name (cpu, sizeof cpu);
  printf ("# libgtc %s, GMP %s on %s\n", GTC_VERSION_STRING, gmp_version,
	  cpu);
}

/* gtc's defaults for the Gaussian products, one line: the thresholds and
   which caps have tables.  */
static void
print_config (const gtc_config *c)
{
  char t[4][24];
  const mp_size_t v[4] = { c->gtc22_thresh, c->gtc33_thresh, c->gtc44_thresh,
    c->gtc55_thresh };
  for (int i = 0; i < 4; i++)
    if (v[i] >= GTC_NEVER)
      strcpy (t[i], "never");
    else
      snprintf (t[i], sizeof t[i], "%ld", (long) v[i]);
  printf ("# gtc: gtc_22..55 from %s, %s, %s, %s; tables for the caps",
	  t[0], t[1], t[2], t[3]);
  for (int i = 2; i <= 5; i++)
    if (c->gtc_table[GTC_CAP (i)])
      printf (" %d", i);
  printf (" (elsewhere the thresholds)\n");
}

static int
kind_of (const char *name, size_t len)
{
  for (int k = 0; k < M_N; k++)
    if (strlen (KIND[k].name) == len && !strncmp (name, KIND[k].name, len))
      return k;
  return -1;
}

/* The component {p, 2n+1} with sign neg as an mpz (read-only view).  */
static mpz_srcptr
as_mpz (mpz_ptr t, mp_srcptr p, mp_size_t len, int neg)
{
  mpz_roinit_n (t, p, len);
  if (neg)
    mpz_neg (t, t);
  return t;
}

int
main (int argc, char **argv)
{
  long sizes[4096];
  int nsizes = 0, trials = 9, quick = 0, pin = -1;
  double min_ms = 25;
  const char *methods = NULL, *compare = NULL, *csv_name = NULL;
  long rmin = 0, rmax = 0, rk = 0;

  /* GTC_BENCH_SKEW bytes, allocated first and kept: every later heap
     block moves by them.  */
  {
    const char *sk = getenv ("GTC_BENCH_SKEW");
    long skew = sk != NULL ? atol (sk) : GTC_BENCH_SKEW_DEFAULT;
    if (skew > 0)
      {
	volatile char *pad = malloc ((size_t) skew);
	if (pad != NULL)
	  pad[0] = 0;
      }
  }

  for (int i = 1; i < argc; i++)
    {
      const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
      if (strcmp (a, "--help") == 0 || strcmp (a, "-h") == 0)
	{
	  usage ();
	  return 0;
	}
      else if (strcmp (a, "--quick") == 0)
	quick = 1;
      else if (v == NULL)
	{
	  fprintf (stderr, "missing value for %s\n", a);
	  return 2;
	}
      else if (strcmp (a, "--compare") == 0 && i++)
	compare = v;
      else if (strcmp (a, "--methods") == 0 && i++)
	methods = v;
      else if (strcmp (a, "--range") == 0 && i++)
	{
	  if (sscanf (v, "%ld:%ld:%ld", &rmin, &rmax, &rk) != 3 || rmin < 1
	      || rmax < rmin || rk < 1)
	    {
	      fprintf (stderr, "bad --range %s\n", v);
	      return 2;
	    }
	}
      else if (strcmp (a, "--sizes") == 0 && i++)
	nsizes = parse_list (v, sizes, 4096);
      else if (strcmp (a, "--trials") == 0 && i++)
	trials = atoi (v);
      else if (strcmp (a, "--min-time") == 0 && i++)
	min_ms = atof (v);
      else if (strcmp (a, "--csv") == 0 && i++)
	csv_name = v;
      else if (strcmp (a, "--pin") == 0 && i++)
	pin = atoi (v);
      else
	{
	  fprintf (stderr, "unknown option %s\n", a);
	  usage ();
	  return 2;
	}
    }

  if (pin >= 0)
    {
      cpu_set_t set;
      CPU_ZERO (&set);
      CPU_SET (pin, &set);
      if (sched_setaffinity (0, sizeof set, &set) != 0)
	perror ("sched_setaffinity");
    }

  if (compare)
    {
      unsigned c;
      for (c = 0; c < sizeof COMPARE / sizeof COMPARE[0]; c++)
	if (!strcmp (compare, COMPARE[c].name))
	  break;
      if (c == sizeof COMPARE / sizeof COMPARE[0])
	{
	  fprintf (stderr, "bad --compare %s (g33, g44 or g55)\n", compare);
	  return 2;
	}
      if (!methods)
	methods = COMPARE[c].methods;
      if (rk == 0 && nsizes == 0)
	{
	  /* sizes growing by 10% from MIN to MAX */
	  rmin = COMPARE[c].min;
	  rmax = COMPARE[c].max;
	  rk = lround (log ((double) rmax / rmin) / log (1.1)) + 1;
	}
    }
  if (!methods)
    methods = "gmp3m_toom33,gtc_33";
  if (quick)
    {
      trials = 2;
      min_ms = 1;
      if (nsizes == 0 && rk == 0)
	{
	  rmin = 100;
	  rmax = 3000;
	  rk = 3;
	}
      else if (rk > 4)
	rk = 4;
    }
  if (rk == 0)
    {
      rmin = 100;
      rmax = 3000;
      rk = 12;
    }
  if (trials < 1)
    trials = 1;
  if (nsizes == 0)
    {
      double ratio = rk > 1 ? pow ((double) rmax / rmin, 1.0 / (rk - 1)) : 1;
      double x = rmin;
      for (long j = 0; j < rk && nsizes < 4096; j++, x *= ratio)
	{
	  long s = j == rk - 1 ? rmax : lround (x);
	  if (nsizes == 0 || s > sizes[nsizes - 1])
	    sizes[nsizes++] = s;
	}
    }

  gtc_config cfg;
  gtc_config_default (&cfg);
  int nm = 0;
  method ms[16];
  long min_n = 1;
  for (const char *p = methods; *p;)
    {
      const char *e = strchr (p, ',');
      size_t len = e ? (size_t) (e - p) : strlen (p);
      int k = kind_of (p, len);
      if (k < 0 || nm == 16)
	{
	  fprintf (stderr, "unknown method %.*s\n", (int) len, p);
	  return 2;
	}
      ms[nm].kind = k;
      ms[nm].name = KIND[k].name;
      ms[nm].ctx = KIND[k].gtc ? gtc_ctx_new (&cfg) : NULL;
      ms[nm].trial = malloc (trials * sizeof (double));
      nm++;
      if (KIND[k].min_n > min_n)
	min_n = KIND[k].min_n;
      p += len;
      if (*p == ',')
	p++;
    }
  for (int j = 0; j < nsizes; j++)
    if (sizes[j] < min_n)
      {
	fprintf (stderr, "sizes must be at least %ld limbs for these "
		 "methods\n", min_n);
	return 2;
      }

  FILE *csv = NULL;
  if (csv_name)
    {
      csv = fopen (csv_name, "w");
      if (csv == NULL)
	{
	  perror (csv_name);
	  return 1;
	}
      fprintf (csv, "limbs,method,config,min_ns,median_ns,ratio_min,"
	       "ratio_median\n");
    }

  print_header ();
  print_config (&cfg);
  printf ("# one level of each method at the top, its own algorithms below "
	  "it:\n# gtc_ii's products by gtc_jj (j <= i) and the Gaussian base "
	  "case; gmp3m_toomii:\n# ac - bd + (ac + bd - (a-b)(c-d)) i, three "
	  "mpn_toomii_mul, their products by\n# GMP's algorithms up to "
	  "itself\n");
  if (quick)
    printf ("# --quick: smoke test, the timings are not meaningful\n");
  printf ("# %d trial(s) of >= %.0f ms per method and size; min time per "
	  "Gaussian multiplication, (ratio to %s)\n# each size cycles "
	  "through up to 256 different inputs (as many as fit in 256 KB), "
	  "random signs\n", trials, min_ms, ms[0].name);
  printf ("%8s", "limbs");
  for (int j = 0; j < nm; j++)
    printf ("  %*s", col_width (&ms[j]), ms[j].name);
  printf ("\n");

  gmp_randstate_t rs;
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 20261015);
  mpz_t z, refr, refi, t, tr, ti;
  mpz_inits (z, refr, refi, t, NULL);

  for (int si = 0; si < nsizes; si++)
    {
      mp_size_t n = sizes[si];
      POOL = pool_size (n);
      PN = n;
      XPOOL = malloc (POOL * 4 * n * sizeof (mp_limb_t));
      SPOOL = malloc (POOL * 4);
      mp_size_t itch = mpn_toom33_mul_itch (n, n);
      if (itch < mpn_toom44_mul_itch (n, n))
	itch = mpn_toom44_mul_itch (n, n);
      if (itch < MPN_TOOM6H_MUL_ITCH_BOUND (n))
	itch = MPN_TOOM6H_MUL_ITCH_BOUND (n);
      SCRATCH = malloc ((itch + 1) * sizeof (mp_limb_t));
      ZR = malloc ((2 * n + 1) * sizeof (mp_limb_t));
      ZI = malloc ((2 * n + 1) * sizeof (mp_limb_t));
      P = malloc (2 * n * sizeof (mp_limb_t));
      Q = malloc (2 * n * sizeof (mp_limb_t));
      R = malloc (2 * n * sizeof (mp_limb_t));
      U = malloc (n * sizeof (mp_limb_t));
      V = malloc (n * sizeof (mp_limb_t));
      for (long q = 0; q < 4 * POOL; q++)
	{
	  mpz_urandomb (z, rs, n * GMP_NUMB_BITS);
	  mpz_setbit (z, n * GMP_NUMB_BITS - 1);
	  mpn_copyi (XPOOL + q * n, mpz_limbs_read (z), n);
	  SPOOL[q] = gmp_urandomb_ui (rs, 1);
	}

      /* Check every method on every input of the pool against mpz, then
	 warm up and calibrate.  The result areas are overwritten first, so
	 that a method that writes nothing cannot inherit the previous
	 method's correct product.  */
      for (long q = 0; q < POOL; q++)
	{
	  use_input (q);
	  mpz_t xr, xi, yr, yi;
	  as_mpz (xr, XR, n, XRN);
	  as_mpz (xi, XI, n, XIN);
	  as_mpz (yr, YR, n, YRN);
	  as_mpz (yi, YI, n, YIN);
	  mpz_mul (refr, xr, yr);
	  mpz_mul (t, xi, yi);
	  mpz_sub (refr, refr, t);
	  mpz_mul (refi, xr, yi);
	  mpz_mul (t, xi, yr);
	  mpz_add (refi, refi, t);
	  for (int j = 0; j < nm; j++)
	    {
	      for (mp_size_t i = 0; i < 2 * n + 1; i++)
		ZR[i] = ZI[i] = 0xdeadbeefcafef00dULL ^ i;
	      ZRN = ZIN = 2;
	      run (&ms[j], n);
	      if (mpz_cmp (as_mpz (tr, ZR, 2 * n + 1, ZRN), refr) != 0
		  || mpz_cmp (as_mpz (ti, ZI, 2 * n + 1, ZIN), refi) != 0)
		{
		  fprintf (stderr, "WRONG RESULT: %s at n=%ld\n", ms[j].name,
			   (long) n);
		  return 1;
		}
	    }
	}
      use_input (0);
      for (int j = 0; j < nm; j++)
	ms[j].iters = calibrate (&ms[j], n, min_ms * 1e6);
      /* Interleave the methods, rotating the order, so that slow drifts in
	 the machine's state affect all of them alike.  */
      for (int tt = 0; tt < trials; tt++)
	for (int jj = 0; jj < nm; jj++)
	  {
	    int j = (jj + tt) % nm;
	    ms[j].trial[tt] = time_iters (&ms[j], n, ms[j].iters);
	  }

      double base_min = minimum (ms[0].trial, trials);
      double base_med = median (ms[0].trial, trials);
      printf ("%8ld", (long) n);
      for (int j = 0; j < nm; j++)
	{
	  char cell[64];
	  double mn = minimum (ms[j].trial, trials);
	  if (j == 0)
	    snprintf (cell, sizeof cell, "%.2fus", mn / 1e3);
	  else
	    snprintf (cell, sizeof cell, "%.2fus (%.3f)", mn / 1e3,
		      mn / base_min);
	  printf ("  %*s", col_width (&ms[j]), cell);
	}
      printf ("\n");
      fflush (stdout);

      if (csv)
	for (int j = 0; j < nm; j++)
	  {
	    double mn = minimum (ms[j].trial, trials);
	    double md = median (ms[j].trial, trials);
	    fprintf (csv, "%ld,%s,\"\",%.1f,%.1f,%.4f,%.4f\n", (long) n,
		     ms[j].name, mn, md, mn / base_min, md / base_med);
	  }

      free (XPOOL);
      free (SPOOL);
      free (SCRATCH);
      free (ZR);
      free (ZI);
      free (P);
      free (Q);
      free (R);
      free (U);
      free (V);
    }

  if (csv)
    fclose (csv);
  mpz_clears (z, refr, refi, t, NULL);
  gmp_randclear (rs);
  for (int j = 0; j < nm; j++)
    {
      gtc_ctx_free (ms[j].ctx);
      free (ms[j].trial);
    }
  return 0;
}
