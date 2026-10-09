/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   Benchmark: libgtc's rational multiplication against GMP's.

   The comparisons (--compare), each algorithm at the top and its own
   algorithms below it, as a production implementation would run them:

     22     r2gtc_22 against GMP's Karatsuba, mpn_toom22_mul, up to 2,000
            limbs;
     33     r2gtc_33 against GMP's Toom-3, mpn_toom33_mul, up to 3,000;
     44     r2gtc_44 against GMP's Toom-4, mpn_toom44_mul, 26 to 4,000;
     55     r2gtc_55 against GMP's Toom-4 and Toom-6.5 (mpn_toom6h_mul,
            GMP's balanced six-way split; GMP has no Toom-5), up to 5,000;
     r2gtc  r2gtc against GMP's Toom-4, Toom-6.5 and its full
            multiplication, mpn_mul_n, up to 10,000.

   One level of r2gtc_ii at the top, then its products below it by gtc's
   tables and thresholds within the cap i (r2gtc_jj, j <= i, GMP's
   Karatsuba and textbook product for the rational ones; gtc_jj, j <= i,
   and the textbook product for the Gaussian ones), as one level of GMP's
   mpn_toomii_mul then hands its products to GMP's algorithms up to
   itself by GMP's thresholds.  r2gtc and mpn_mul_n choose by size from
   the top.

   --methods names the methods instead, the first the reference of the
   ratios: gmp_toom22, gmp_toom33, gmp_toom44, gmp_toom6h, gmp_mul_n,
   r2gtc, r2gtc_22, r2gtc_33, r2gtc_44, r2gtc_55.  Without either, the
   methods are gmp_toom33, gmp_mul_n, r2gtc and r2gtc_22.  Run bench_gtc
   --help for the options.  */

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
{ M_TOOM22, M_TOOM33, M_TOOM44, M_TOOM6H, M_MUL_N, M_R2GTC, M_R2GTC_22,
  M_R2GTC_33, M_R2GTC_44, M_R2GTC_55, M_N };

/* By kind: the name, the smallest size it is measured at, whether it is
   gtc's.  GMP's Toom-3 from 17 limbs, its Toom-6.5 from 46 (the smallest
   its algorithm and tests allow), its Toom-4 from 26.  GMP calls
   mpn_toom44_mul from MUL_TOOM44_THRESHOLD on only, and where that is at
   least 4 MUL_TOOM22_THRESHOLD, its recursion leaves out the basecase:
   at 22 and 25 limbs its last part has 4 limbs, and mpn_toom33_mul on 4
   limbs can crash.  From 26 limbs the last part has at least 5.  */
static const struct
{
  const char *name;
  long min_n;
  int gtc;
} KIND[M_N] = {
  { "gmp_toom22", 4, 0 }, { "gmp_toom33", 17, 0 }, { "gmp_toom44", 26, 0 },
  { "gmp_toom6h", 46, 0 }, { "gmp_mul_n", 1, 0 }, { "r2gtc", 1, 1 },
  { "r2gtc_22", 2, 1 }, { "r2gtc_33", 5, 1 }, { "r2gtc_44", 13, 1 },
  { "r2gtc_55", 17, 1 } };

/* The presets of --compare: the methods (the first the reference) and the
   sizes, geometric from MIN to MAX.  */
static const struct
{
  const char *name, *methods;
  long min, max;
} COMPARE[] = {
  { "22", "gmp_toom22,r2gtc_22", 20, 2000 },
  { "33", "gmp_toom33,r2gtc_33", 20, 3000 },
  { "44", "gmp_toom44,r2gtc_44", 26, 4000 },
  { "55", "gmp_toom44,gmp_toom6h,r2gtc_55", 46, 5000 },
  { "r2gtc", "gmp_mul_n,gmp_toom44,gmp_toom6h,r2gtc", 46, 10000 } };

typedef struct
{
  int kind;
  char name[128];		/* the column's name */
  const char *config;		/* the --config it runs with, or NULL */
  gtc_ctx *ctx;
  double *trial;		/* per-call time of each trial, ns */
  long iters;
} method;

static mp_ptr A, B, SCRATCH;

/* The inputs of one size: a pool of POOL pairs of n limbs, cycled through
   by the timing loops.  Repeating one input would let the branch predictor
   learn every data-dependent branch of a small product, which hides the
   mispredictions of real use and favours the code with more such branches.
   The pool stays within 256 KB, since a product's inputs are usually hot
   in the cache.  */
static mp_ptr APOOL, BPOOL;
static long POOL, CUR;
static mp_size_t PN;

static void
use_input (long j)
{
  CUR = j;
  A = APOOL + j * PN;
  B = BPOOL + j * PN;
}

static long
pool_size (mp_size_t n)
{
  long p = 1;
  while (p < 256 && 2 * p * 2 * n * (long) sizeof (mp_limb_t) <= 256 * 1024)
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

static void
run (method *m, mp_ptr rp, mp_size_t n)
{
  switch (m->kind)
    {
    case M_TOOM22: mpn_toom22_mul (rp, A, n, B, n, SCRATCH); break;
    case M_TOOM33: mpn_toom33_mul (rp, A, n, B, n, SCRATCH); break;
    case M_TOOM44: mpn_toom44_mul (rp, A, n, B, n, SCRATCH); break;
    case M_TOOM6H: mpn_toom6h_mul (rp, A, n, B, n, SCRATCH); break;
    case M_MUL_N: mpn_mul_n (rp, A, B, n); break;
    case M_R2GTC: r2gtc_mul_n_ctx (m->ctx, rp, A, B, n); break;
    case M_R2GTC_22: r2gtc_22_ctx (m->ctx, rp, A, B, n); break;
    case M_R2GTC_33: r2gtc_33_ctx (m->ctx, rp, A, B, n); break;
    case M_R2GTC_44: r2gtc_44_ctx (m->ctx, rp, A, B, n); break;
    default: r2gtc_55_ctx (m->ctx, rp, A, B, n); break;
    }
}

static double
time_iters (method *m, mp_ptr rp, mp_size_t n, long iters)
{
  double t0 = now_ns ();
  for (long i = 0; i < iters; i++)
    {
      run (m, rp, n);
      use_input ((CUR + 1) & (POOL - 1));
    }
  return (now_ns () - t0) / iters;
}

/* Iterations such that one trial lasts about min_ns.  */
static long
calibrate (method *m, mp_ptr rp, mp_size_t n, double min_ns)
{
  long iters = 1;
  double t;
  while ((t = time_iters (m, rp, n, iters) * iters) < min_ns / 4
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

/* Width of a method's column in the results table.  */
static int
col_width (const method *m)
{
  int w = (int) strlen (m->name);
  return w > 20 ? w : 20;
}

/* One threshold: a number or "never".  */
static int
parse_thresh (const char *f, size_t len, mp_size_t *v)
{
  char *end;
  long x;
  if (len == 5 && strncmp (f, "never", 5) == 0)
    {
      *v = GTC_NEVER;
      return 1;
    }
  x = strtol (f, &end, 10);
  if (end != f + len || len == 0)
    return 0;
  *v = x;
  return 1;
}

/* --config KEY=VALUE,...: the defaults with the thresholds gtc22, gtc33,
   gtc44, gtc55, km, r2gtc22, r2gtc33, r2gtc44, r2gtc55, gmp22 (a number
   or never), and gtables=none, rtables=none (no tables: the thresholds
   at every size).  */
static int
parse_config (const char *s, gtc_config *c)
{
  static const struct
  {
    const char *key;
    size_t off;
  } keys[] = {
    { "gtc22", offsetof (gtc_config, gtc22_thresh) },
    { "gtc33", offsetof (gtc_config, gtc33_thresh) },
    { "gtc44", offsetof (gtc_config, gtc44_thresh) },
    { "gtc55", offsetof (gtc_config, gtc55_thresh) },
    { "km", offsetof (gtc_config, gtc_km_thresh) },
    { "r2gtc22", offsetof (gtc_config, r2gtc22_thresh) },
    { "r2gtc33", offsetof (gtc_config, r2gtc33_thresh) },
    { "r2gtc44", offsetof (gtc_config, r2gtc44_thresh) },
    { "r2gtc55", offsetof (gtc_config, r2gtc55_thresh) },
    { "gmp22", offsetof (gtc_config, gmp22_thresh) } };
  gtc_config_default (c);
  while (*s)
    {
      const char *eq = strchr (s, '='), *end = strchr (s, ',');
      size_t klen;
      if (end == NULL)
	end = s + strlen (s);
      if (eq == NULL || eq > end)
	return 0;
      klen = eq - s;
      if (klen == 7 && !strncmp (s, "gtables", 7))
	{
	  if (end - eq - 1 != 4 || strncmp (eq + 1, "none", 4))
	    return 0;
	  for (int j = 0; j < 4; j++)
	    c->gtc_table[j] = NULL;
	}
      else if (klen == 7 && !strncmp (s, "rtables", 7))
	{
	  if (end - eq - 1 != 4 || strncmp (eq + 1, "none", 4))
	    return 0;
	  for (int j = 0; j < 4; j++)
	    c->r2gtc_table[j] = NULL;
	}
      else
	{
	  unsigned k;
	  for (k = 0; k < sizeof keys / sizeof keys[0]; k++)
	    if (strlen (keys[k].key) == klen && !strncmp (s, keys[k].key, klen))
	      break;
	  if (k == sizeof keys / sizeof keys[0]
	      || !parse_thresh (eq + 1, end - eq - 1,
				(mp_size_t *) ((char *) c + keys[k].off)))
	    return 0;
	}
      s = *end ? end + 1 : end;
    }
  return 1;
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
  puts ("usage: bench_gtc [options]\n"
	"gtc's rational multiplication against GMP's: each method at the top,\n"
	"its own algorithms below it.  Times are per multiplication; ratio =\n"
	"time / time(first column).  The minimum over trials is the headline\n"
	"number, the median is also given.\n"
	"  --compare C           22 | 33 | 44 | 55 | r2gtc: r2gtc_22 against\n"
	"                        mpn_toom22_mul up to 2,000 limbs, r2gtc_33 against\n"
	"                        mpn_toom33_mul up to 3,000, r2gtc_44 against\n"
	"                        mpn_toom44_mul up to 4,000, r2gtc_55 against\n"
	"                        mpn_toom44_mul and mpn_toom6h_mul up to 5,000,\n"
	"                        r2gtc against those two and mpn_mul_n up to\n"
	"                        10,000 (sizes growing by 10%)\n"
	"  --methods M1,M2,...   the methods instead, the first the reference:\n"
	"                        gmp_toom22, gmp_toom33, gmp_toom44, gmp_toom6h,\n"
	"                        gmp_mul_n, r2gtc, r2gtc_22 ... r2gtc_55\n"
	"                        (default gmp_toom33,gmp_mul_n,r2gtc,r2gtc_22)\n"
	"  --range MIN:MAX:K     K sizes (limbs), geometric in [MIN, MAX]\n"
	"                        (default 100:20000:24, or the comparison's)\n"
	"  --sizes N1,N2,...     explicit sizes in limbs\n"
	"  --config KEY=V,...    gtc's methods once per --config (repeatable)\n"
	"                        instead of with the defaults: KEY gtc22, gtc33,\n"
	"                        gtc44, gtc55, km, r2gtc22, r2gtc33, r2gtc44,\n"
	"                        r2gtc55, gmp22 (a size or never), gtables=none,\n"
	"                        rtables=none (the thresholds alone)\n"
	"  --trials T            timing trials per method and size (default 9)\n"
	"  --min-time MS         length of one trial in ms (default 25)\n"
	"  --no-mul-n            leave gmp_mul_n out of the methods\n"
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

/* gtc's defaults, one line: the thresholds and which caps have tables.  */
static void
print_config (const gtc_config *c)
{
  char t[9][24];
  const mp_size_t v[9] = { c->gtc22_thresh, c->gtc33_thresh, c->gtc44_thresh,
    c->gtc55_thresh, c->r2gtc22_thresh, c->r2gtc33_thresh, c->r2gtc44_thresh,
    c->r2gtc55_thresh, c->gmp22_thresh };
  for (int i = 0; i < 9; i++)
    if (v[i] >= GTC_NEVER)
      strcpy (t[i], "never");
    else
      snprintf (t[i], sizeof t[i], "%ld", (long) v[i]);
  printf ("# gtc: gtc_22..55 from %s, %s, %s, %s; r2gtc_22..55 from %s, %s, "
	  "%s, %s; GMP's Karatsuba from %s;\n#      tables for the caps",
	  t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]);
  for (int i = 2; i <= 5; i++)
    printf (" %d:%s%s", i, c->gtc_table[GTC_CAP (i)] ? "G" : "",
	    c->r2gtc_table[GTC_CAP (i)] ? "R" : "");
  printf (" (G Gaussian, R rational; elsewhere the thresholds)\n");
}

static int
kind_of (const char *name, size_t len)
{
  for (int k = 0; k < M_N; k++)
    if (strlen (KIND[k].name) == len && !strncmp (name, KIND[k].name, len))
      return k;
  return -1;
}

int
main (int argc, char **argv)
{
  long sizes[4096];
  int nsizes = 0, trials = 9, quick = 0, with_mul_n = 1, pin = -1;
  double min_ms = 25;
  const char *methods = NULL, *compare = NULL;
  /* GTC_BENCH_SKEW bytes, allocated first and kept: every later heap
     block (the contexts, their arenas, the input pools) moves by them,
     which can shift single sizes by several percent; an A/B can vary it
     over its rounds.  */
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
  long rmin = 0, rmax = 0, rk = 0;
  const char *csv_name = NULL;
  const char *cfg_spec[64];
  gtc_config cfgs[64];
  int ncfgs = 0;

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
      else if (strcmp (a, "--no-mul-n") == 0)
	with_mul_n = 0;
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
      else if (strcmp (a, "--config") == 0 && i++)
	{
	  if (ncfgs == 64 || !parse_config (v, &cfgs[ncfgs]))
	    {
	      fprintf (stderr, "bad --config %s\n", v);
	      return 2;
	    }
	  cfg_spec[ncfgs++] = v;
	}
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

  /* The comparison's methods and sizes.  */
  if (compare)
    {
      unsigned c;
      for (c = 0; c < sizeof COMPARE / sizeof COMPARE[0]; c++)
	if (!strcmp (compare, COMPARE[c].name))
	  break;
      if (c == sizeof COMPARE / sizeof COMPARE[0])
	{
	  fprintf (stderr, "bad --compare %s (22, 33, 44, 55 or r2gtc)\n",
		   compare);
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
    methods = "gmp_toom33,gmp_mul_n,r2gtc,r2gtc_22";
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
      rmax = 20000;
      rk = 24;
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
  if (ncfgs == 0)
    {
      gtc_config_default (&cfgs[0]);
      cfg_spec[ncfgs++] = NULL;
    }

  /* The methods: GMP's once, gtc's once per configuration.  */
  int nm = 0;
  method *ms = calloc (64 * 16, sizeof *ms);
  long min_n = 1;
  for (const char *p = methods; *p;)
    {
      const char *e = strchr (p, ',');
      size_t len = e ? (size_t) (e - p) : strlen (p);
      int k = kind_of (p, len);
      if (k < 0)
	{
	  fprintf (stderr, "unknown method %.*s\n", (int) len, p);
	  return 2;
	}
      if (k != M_MUL_N || with_mul_n)
	for (int j = 0; j < (KIND[k].gtc ? ncfgs : 1); j++)
	  {
	    method *m = &ms[nm++];
	    m->kind = k;
	    if (KIND[k].gtc)
	      {
		m->ctx = gtc_ctx_new (&cfgs[j]);
		m->config = cfg_spec[j];
	      }
	    if (m->config)
	      snprintf (m->name, sizeof m->name, "%s[%s]", KIND[k].name,
			m->config);
	    else
	      snprintf (m->name, sizeof m->name, "%s", KIND[k].name);
	  }
      if (KIND[k].min_n > min_n)
	min_n = KIND[k].min_n;
      p += len;
      if (*p == ',')
	p++;
    }
  if (nm < 1)
    {
      fprintf (stderr, "no methods\n");
      return 2;
    }
  for (int j = 0; j < nsizes; j++)
    if (sizes[j] < min_n)
      {
	fprintf (stderr, "sizes must be at least %ld limbs for these "
		 "methods\n", min_n);
	return 2;
      }
  for (int j = 0; j < nm; j++)
    ms[j].trial = malloc (trials * sizeof (double));

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
  print_config (&cfgs[0]);
  printf ("# one level of each method at the top, its own algorithms below "
	  "it:\n# r2gtc_ii's products by r2gtc_jj (j <= i), GMP's Karatsuba "
	  "and textbook product,\n# its Gaussian ones by gtc_jj (j <= i) "
	  "and the textbook product; mpn_toomii_mul's\n# by GMP's "
	  "algorithms up to itself; r2gtc and mpn_mul_n by size from the "
	  "top\n");
  if (quick)
    printf ("# --quick: smoke test, the timings are not meaningful\n");
  printf ("# %d trial(s) of >= %.0f ms per method and size; min time per "
	  "multiplication, (ratio to %s)\n# each size cycles through up to 256 "
	  "different inputs (as many as fit in 256 KB)\n", trials, min_ms,
	  ms[0].name);
  printf ("%8s", "limbs");
  for (int j = 0; j < nm; j++)
    printf ("  %*s", col_width (&ms[j]), ms[j].name);
  printf ("\n");

  gmp_randstate_t rs;
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 20260918);
  mpz_t z;
  mpz_init (z);

  for (int si = 0; si < nsizes; si++)
    {
      mp_size_t n = sizes[si];
      POOL = pool_size (n);
      PN = n;
      APOOL = malloc (POOL * n * sizeof (mp_limb_t));
      BPOOL = malloc (POOL * n * sizeof (mp_limb_t));
      mp_size_t itch = mpn_toom22_mul_itch (n, n);
      if (itch < mpn_toom33_mul_itch (n, n))
	itch = mpn_toom33_mul_itch (n, n);
      if (itch < mpn_toom44_mul_itch (n, n))
	itch = mpn_toom44_mul_itch (n, n);
      if (itch < MPN_TOOM6H_MUL_ITCH_BOUND (n))
	itch = MPN_TOOM6H_MUL_ITCH_BOUND (n);
      SCRATCH = malloc ((itch + 1) * sizeof (mp_limb_t));
      mp_ptr ref = malloc (2 * n * sizeof (mp_limb_t));
      mp_ptr rp = malloc (2 * n * sizeof (mp_limb_t));
      for (long q = 0; q < POOL; q++)
	{
	  mpz_urandomb (z, rs, n * GMP_NUMB_BITS);
	  mpz_setbit (z, n * GMP_NUMB_BITS - 1);
	  mpn_copyi (APOOL + q * n, mpz_limbs_read (z), n);
	  mpz_urandomb (z, rs, n * GMP_NUMB_BITS);
	  mpz_setbit (z, n * GMP_NUMB_BITS - 1);
	  mpn_copyi (BPOOL + q * n, mpz_limbs_read (z), n);
	}

      /* Check every method on every input of the pool, then warm up and
	 calibrate.  The result area is overwritten first, so that a method
	 that writes nothing cannot inherit the previous method's correct
	 product.  */
      for (long q = 0; q < POOL; q++)
	{
	  use_input (q);
	  mpn_mul_n (ref, A, B, n);
	  for (int j = 0; j < nm; j++)
	    {
	      for (mp_size_t i = 0; i < 2 * n; i++)
		rp[i] = 0xdeadbeefcafef00dULL ^ i;
	      run (&ms[j], rp, n);
	      if (mpn_cmp (rp, ref, 2 * n) != 0)
		{
		  fprintf (stderr, "WRONG RESULT: %s at n=%ld\n", ms[j].name,
			   (long) n);
		  return 1;
		}
	    }
	}
      use_input (0);
      for (int j = 0; j < nm; j++)
	ms[j].iters = calibrate (&ms[j], rp, n, min_ms * 1e6);
      /* Interleave the methods, rotating the order, so that slow drifts in
	 the machine's state affect all of them alike.  */
      for (int t = 0; t < trials; t++)
	for (int jj = 0; jj < nm; jj++)
	  {
	    int j = (jj + t) % nm;
	    ms[j].trial[t] = time_iters (&ms[j], rp, n, ms[j].iters);
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
	    fprintf (csv, "%ld,%s,\"%s\",%.1f,%.1f,%.4f,%.4f\n", (long) n,
		     KIND[ms[j].kind].name,
		     ms[j].config ? ms[j].config : "", mn, md, mn / base_min,
		     md / base_med);
	  }

      free (APOOL);
      free (BPOOL);
      free (SCRATCH);
      free (ref);
      free (rp);
    }

  if (csv)
    fclose (csv);
  mpz_clear (z);
  gmp_randclear (rs);
  for (int j = 0; j < nm; j++)
    {
      gtc_ctx_free (ms[j].ctx);
      free (ms[j].trial);
    }
  free (ms);
  return 0;
}
