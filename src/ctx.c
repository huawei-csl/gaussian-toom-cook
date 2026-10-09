/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   Contexts: the configuration, the algorithm of each product size, and the
   scratch arena.  */

/* Sizes below GTC_TABLE_N take their algorithm from the context's tables.  */
#define GTC_TABLE_N 1024
/* The smallest sizes of a Toom-4 and a Toom-5 level: the last part, n -
   3k limbs with k = ceil(n/4) (n - 4k, k = ceil(n/5)), must not be
   empty.  */
#define GTC_44_MIN 13
#define GTC_55_MIN 17
/* mpn_toom22_mul's smallest size: at 3 limbs it reads a limb past the
   product.  */
#define GTC_GMP22_MIN 4

/* The algorithm of a Gaussian product; GTC_ALG_ii is gtc_ii, so that the
   algorithms allowed under the cap i are those up to i.  GTC_ALG_LEAF: a
   base case or a gtc_22 level in assembly, from the context's leaf table.  */
enum
{ GTC_ALG_TB, GTC_ALG_LEAF, GTC_ALG_22, GTC_ALG_33, GTC_ALG_44,
  GTC_ALG_55 };

/* A Gaussian product that needs no context (P, Q, ar, ai, br, bi as for
   gtc_q1), and the table of such functions through which a Karatsuba level
   in assembly reaches the products of its halves.  */
typedef void (*gtc_leaf_fn) (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,
			     mp_srcptr, const void *);
#define GTC_LEAF_N 65

/* The algorithm of a rational product; R2GTC_ALG_ii is r2gtc_ii.  */
enum
{ R2GTC_ALG_TB, R2GTC_ALG_GMP22, R2GTC_ALG_22, R2GTC_ALG_33, R2GTC_ALG_44,
  R2GTC_ALG_55 };

struct gtc_ctx
{
  gtc_config cfg;
  /* [GTC_CAP (i)][n]: the algorithm of an n-limb product of gtc_ii, and
     of an n-limb rational product of r2gtc_ii.  */
  unsigned char alg[4][GTC_TABLE_N];
  unsigned char ralg[4][GTC_TABLE_N];
  gtc_leaf_fn leaf[4][GTC_LEAF_N];	/* where alg is GTC_ALG_LEAF */
  /* The leaf functions of the cap 2 as the Karatsuba levels in assembly
     call them: the entries that save no callee-saved register.  */
  gtc_leaf_fn leaf_ns[GTC_LEAF_N];
  mp_ptr arena;			/* scratch, used as a stack */
  size_t cap, top;		/* in limbs */
  size_t vtop, peak;		/* usage as if every block fitted the arena */
  unsigned char tc4pad[512];	/* tc4_pad_table */
};

/* GTC_TC4_PAD: the Toom-4's and Toom-5's scratch buffers padded so that
   no two of them lie within 256 bytes of a multiple of 4096 bytes apart
   (4K aliasing between their loads and stores); per microarchitecture.  */
#if defined (GTC_ASM_X86) && !defined (GTC_ZEN)
#define GTC_TC4_PAD 1
#else
#define GTC_TC4_PAD 0
#endif

/* The pad, 0 to 31 limbs, by the buffer width modulo 512 limbs: the least
   that keeps ten buffers 256 bytes from aliasing, else the one that keeps
   them furthest from it.  */
static void
tc4_pad_table (unsigned char *t)
{
  for (unsigned r = 0; r < 512; r++)
    {
      unsigned best = 0, bestd = 0;
      for (unsigned p = 0; p < 32; p++)
	{
	  unsigned d = 2048, b = ((r + p) * 8) & 4095, o = 0;
	  for (int j = 1; j <= 9; j++)
	    {
	      unsigned e;
	      o = (o + b) & 4095;
	      e = o < 4096 - o ? o : 4096 - o;
	      d = e < d ? e : d;
	    }
	  if (d >= 256)
	    {
	      best = p;
	      break;
	    }
	  if (d > bestd)
	    bestd = d, best = p;
	}
      t[r] = (unsigned char) best;
    }
}

typedef struct
{
  size_t top, vtop;
  void *heap;
} tmp_mark;

static void *
xmalloc (size_t bytes)
{
  void *p = malloc (bytes);
  if (p == NULL)
    {
      fprintf (stderr, "gtc: out of memory\n");
      abort ();
    }
  return p;
}

/* Blocks of scratch start on a cache line.  */
#define GTC_TMP_ALIGN 8

/* Allocate n limbs of scratch, released in reverse order by tmp_free.
   When the arena is too small the block comes from malloc, and the arena
   grows after the current top-level call.  */
GTC_INLINE mp_ptr
tmp_alloc (gtc_ctx *c, tmp_mark *mk, size_t n)
{
  n = (n + GTC_TMP_ALIGN - 1) / GTC_TMP_ALIGN * GTC_TMP_ALIGN;
  mk->top = c->top;
  mk->vtop = c->vtop;
  mk->heap = NULL;
  c->vtop += n;
  if (c->vtop > c->peak)
    c->peak = c->vtop;
#ifndef GTC_NO_ARENA		/* debugging: every block from malloc */
  if (n <= c->cap - c->top)
#else
  if (0)
#endif
    {
      mp_ptr p = c->arena + c->top;
      c->top += n;
      return p;
    }
  mk->heap = xmalloc (n * sizeof (mp_limb_t));
  return mk->heap;
}

GTC_INLINE void
tmp_free (gtc_ctx *c, tmp_mark *mk)
{
  c->top = mk->top;
  c->vtop = mk->vtop;
  if (__builtin_expect (mk->heap != NULL, 0))
    free (mk->heap);
}

/* Between top-level calls: grow the arena to the peak.  */
static void
arena_fit (gtc_ctx *c)
{
  if (c->peak > c->cap)
    {
      free (c->arena);
      c->cap = c->peak + c->peak / 8;
      c->cap = (c->cap + GTC_TMP_ALIGN - 1) / GTC_TMP_ALIGN * GTC_TMP_ALIGN;
      c->arena = aligned_alloc (GTC_TMP_ALIGN * sizeof (mp_limb_t),
				c->cap * sizeof (mp_limb_t));
      if (c->arena == NULL)
	{
	  fprintf (stderr, "gtc: out of memory\n");
	  abort ();
	}
    }
}

void
gtc_config_default (gtc_config *cfg)
{
  cfg->gtc22_thresh = GTC_DEFAULT_GTC22;
  cfg->gtc33_thresh = GTC_DEFAULT_GTC33;
  cfg->gtc44_thresh = GTC_DEFAULT_GTC44;
  cfg->gtc55_thresh = GTC_DEFAULT_GTC55;
  cfg->gtc_km_thresh = GTC_DEFAULT_GTC_KM;
  cfg->gtc_table[GTC_CAP (2)] = NULL;
  cfg->gtc_table[GTC_CAP (3)] = GTC_DEFAULT_GTC_TABLE_33;
  cfg->gtc_table[GTC_CAP (4)] = GTC_DEFAULT_GTC_TABLE_44;
  cfg->gtc_table[GTC_CAP (5)] = GTC_DEFAULT_GTC_TABLE_55;
  cfg->r2gtc22_thresh = GTC_DEFAULT_R2GTC22;
  cfg->r2gtc33_thresh = GTC_DEFAULT_R2GTC33;
  cfg->r2gtc44_thresh = GTC_DEFAULT_R2GTC44;
  cfg->r2gtc55_thresh = GTC_DEFAULT_R2GTC55;
  cfg->gmp22_thresh = GTC_DEFAULT_GMP22;
  cfg->r2gtc_table[GTC_CAP (2)] = GTC_DEFAULT_R2GTC_TABLE_22;
  cfg->r2gtc_table[GTC_CAP (3)] = GTC_DEFAULT_R2GTC_TABLE_33;
  cfg->r2gtc_table[GTC_CAP (4)] = GTC_DEFAULT_R2GTC_TABLE_44;
  cfg->r2gtc_table[GTC_CAP (5)] = GTC_DEFAULT_R2GTC_TABLE_55;
}

/* The leaf kernels in assembly, of the form gtc_leaf_fn: the textbook
   products gtc_gmul_tbN, the Karatsuba over {1, i} gtc_gmul_kmN (aarch64)
   and the Karatsuba levels gtc_gkaraN.  The textbook and {1, i} kernels
   ignore the table argument.  Each also has an entry name_ns that saves no
   callee-saved register, which only the Karatsuba levels in assembly
   call.  */
#define GTC_LEAF_DECL(name)						\
  void name (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_srcptr,	\
	     const void *);						\
  void name##_ns (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr,	\
		  mp_srcptr, const void *);

#ifdef GTC_TEXTBOOK_ASM
GTC_LEAF_DECL (gtc_gmul_tb2) GTC_LEAF_DECL (gtc_gmul_tb3)
GTC_LEAF_DECL (gtc_gmul_tb4) GTC_LEAF_DECL (gtc_gmul_tb5)
GTC_LEAF_DECL (gtc_gmul_tb6) GTC_LEAF_DECL (gtc_gmul_tb7)
GTC_LEAF_DECL (gtc_gmul_tb8) GTC_LEAF_DECL (gtc_gmul_tb9)
GTC_LEAF_DECL (gtc_gmul_tb10) GTC_LEAF_DECL (gtc_gmul_tb11)
GTC_LEAF_DECL (gtc_gmul_tb12) GTC_LEAF_DECL (gtc_gmul_tb13)
GTC_LEAF_DECL (gtc_gmul_tb14) GTC_LEAF_DECL (gtc_gmul_tb15)
GTC_LEAF_DECL (gtc_gmul_tb16) GTC_LEAF_DECL (gtc_gmul_tb17)
GTC_LEAF_DECL (gtc_gmul_tb18) GTC_LEAF_DECL (gtc_gmul_tb19)
GTC_LEAF_DECL (gtc_gmul_tb20)
#endif
#ifdef GTC_KM_ASM
GTC_LEAF_DECL (gtc_gmul_km2) GTC_LEAF_DECL (gtc_gmul_km3)
GTC_LEAF_DECL (gtc_gmul_km4) GTC_LEAF_DECL (gtc_gmul_km5)
GTC_LEAF_DECL (gtc_gmul_km6) GTC_LEAF_DECL (gtc_gmul_km7)
GTC_LEAF_DECL (gtc_gmul_km8) GTC_LEAF_DECL (gtc_gmul_km9)
GTC_LEAF_DECL (gtc_gmul_km10)
#endif

#ifdef GTC_KARA_ASM
#define GTC_GKARA_MIN 4
#define GTC_GKARA_MAX 64
GTC_LEAF_DECL (gtc_gkara4) GTC_LEAF_DECL (gtc_gkara5)
GTC_LEAF_DECL (gtc_gkara6) GTC_LEAF_DECL (gtc_gkara7)
GTC_LEAF_DECL (gtc_gkara8) GTC_LEAF_DECL (gtc_gkara9)
GTC_LEAF_DECL (gtc_gkara10) GTC_LEAF_DECL (gtc_gkara11)
GTC_LEAF_DECL (gtc_gkara12) GTC_LEAF_DECL (gtc_gkara13)
GTC_LEAF_DECL (gtc_gkara14) GTC_LEAF_DECL (gtc_gkara15)
GTC_LEAF_DECL (gtc_gkara16) GTC_LEAF_DECL (gtc_gkara17)
GTC_LEAF_DECL (gtc_gkara18) GTC_LEAF_DECL (gtc_gkara19)
GTC_LEAF_DECL (gtc_gkara20) GTC_LEAF_DECL (gtc_gkara21)
GTC_LEAF_DECL (gtc_gkara22) GTC_LEAF_DECL (gtc_gkara23)
GTC_LEAF_DECL (gtc_gkara24) GTC_LEAF_DECL (gtc_gkara25)
GTC_LEAF_DECL (gtc_gkara26) GTC_LEAF_DECL (gtc_gkara27)
GTC_LEAF_DECL (gtc_gkara28) GTC_LEAF_DECL (gtc_gkara29)
GTC_LEAF_DECL (gtc_gkara30) GTC_LEAF_DECL (gtc_gkara31)
GTC_LEAF_DECL (gtc_gkara32) GTC_LEAF_DECL (gtc_gkara33)
GTC_LEAF_DECL (gtc_gkara34) GTC_LEAF_DECL (gtc_gkara35)
GTC_LEAF_DECL (gtc_gkara36) GTC_LEAF_DECL (gtc_gkara37)
GTC_LEAF_DECL (gtc_gkara38) GTC_LEAF_DECL (gtc_gkara39)
GTC_LEAF_DECL (gtc_gkara40) GTC_LEAF_DECL (gtc_gkara41)
GTC_LEAF_DECL (gtc_gkara42) GTC_LEAF_DECL (gtc_gkara43)
GTC_LEAF_DECL (gtc_gkara44) GTC_LEAF_DECL (gtc_gkara45)
GTC_LEAF_DECL (gtc_gkara46) GTC_LEAF_DECL (gtc_gkara47)
GTC_LEAF_DECL (gtc_gkara48) GTC_LEAF_DECL (gtc_gkara49)
GTC_LEAF_DECL (gtc_gkara50) GTC_LEAF_DECL (gtc_gkara51)
GTC_LEAF_DECL (gtc_gkara52) GTC_LEAF_DECL (gtc_gkara53)
GTC_LEAF_DECL (gtc_gkara54) GTC_LEAF_DECL (gtc_gkara55)
GTC_LEAF_DECL (gtc_gkara56) GTC_LEAF_DECL (gtc_gkara57)
GTC_LEAF_DECL (gtc_gkara58) GTC_LEAF_DECL (gtc_gkara59)
GTC_LEAF_DECL (gtc_gkara60) GTC_LEAF_DECL (gtc_gkara61)
GTC_LEAF_DECL (gtc_gkara62) GTC_LEAF_DECL (gtc_gkara63)
GTC_LEAF_DECL (gtc_gkara64)

static const gtc_leaf_fn gtc_gkara_fn[] = {
  gtc_gkara4, gtc_gkara5, gtc_gkara6, gtc_gkara7, gtc_gkara8, gtc_gkara9,
  gtc_gkara10, gtc_gkara11, gtc_gkara12, gtc_gkara13, gtc_gkara14,
  gtc_gkara15, gtc_gkara16, gtc_gkara17, gtc_gkara18, gtc_gkara19,
  gtc_gkara20, gtc_gkara21, gtc_gkara22, gtc_gkara23, gtc_gkara24,
  gtc_gkara25, gtc_gkara26, gtc_gkara27, gtc_gkara28, gtc_gkara29,
  gtc_gkara30, gtc_gkara31, gtc_gkara32, gtc_gkara33, gtc_gkara34,
  gtc_gkara35, gtc_gkara36, gtc_gkara37, gtc_gkara38, gtc_gkara39,
  gtc_gkara40, gtc_gkara41, gtc_gkara42, gtc_gkara43, gtc_gkara44,
  gtc_gkara45, gtc_gkara46, gtc_gkara47, gtc_gkara48, gtc_gkara49,
  gtc_gkara50, gtc_gkara51, gtc_gkara52, gtc_gkara53, gtc_gkara54,
  gtc_gkara55, gtc_gkara56, gtc_gkara57, gtc_gkara58, gtc_gkara59,
  gtc_gkara60, gtc_gkara61, gtc_gkara62, gtc_gkara63, gtc_gkara64
};
static const gtc_leaf_fn gtc_gkara_ns_fn[] = {
  gtc_gkara4_ns, gtc_gkara5_ns, gtc_gkara6_ns, gtc_gkara7_ns,
  gtc_gkara8_ns, gtc_gkara9_ns, gtc_gkara10_ns, gtc_gkara11_ns,
  gtc_gkara12_ns, gtc_gkara13_ns, gtc_gkara14_ns, gtc_gkara15_ns,
  gtc_gkara16_ns, gtc_gkara17_ns, gtc_gkara18_ns, gtc_gkara19_ns,
  gtc_gkara20_ns, gtc_gkara21_ns, gtc_gkara22_ns, gtc_gkara23_ns,
  gtc_gkara24_ns, gtc_gkara25_ns, gtc_gkara26_ns, gtc_gkara27_ns,
  gtc_gkara28_ns, gtc_gkara29_ns, gtc_gkara30_ns, gtc_gkara31_ns,
  gtc_gkara32_ns, gtc_gkara33_ns, gtc_gkara34_ns, gtc_gkara35_ns,
  gtc_gkara36_ns, gtc_gkara37_ns, gtc_gkara38_ns, gtc_gkara39_ns,
  gtc_gkara40_ns, gtc_gkara41_ns, gtc_gkara42_ns, gtc_gkara43_ns,
  gtc_gkara44_ns, gtc_gkara45_ns, gtc_gkara46_ns, gtc_gkara47_ns,
  gtc_gkara48_ns, gtc_gkara49_ns, gtc_gkara50_ns, gtc_gkara51_ns,
  gtc_gkara52_ns, gtc_gkara53_ns, gtc_gkara54_ns, gtc_gkara55_ns,
  gtc_gkara56_ns, gtc_gkara57_ns, gtc_gkara58_ns, gtc_gkara59_ns,
  gtc_gkara60_ns, gtc_gkara61_ns, gtc_gkara62_ns, gtc_gkara63_ns,
  gtc_gkara64_ns
};
#endif
#undef GTC_LEAF_DECL

#ifdef GTC_TEXTBOOK_ASM
static const gtc_leaf_fn gtc_tb_fn[] = {
  NULL, NULL, gtc_gmul_tb2, gtc_gmul_tb3, gtc_gmul_tb4, gtc_gmul_tb5,
  gtc_gmul_tb6, gtc_gmul_tb7, gtc_gmul_tb8, gtc_gmul_tb9, gtc_gmul_tb10,
  gtc_gmul_tb11, gtc_gmul_tb12, gtc_gmul_tb13, gtc_gmul_tb14,
  gtc_gmul_tb15, gtc_gmul_tb16, gtc_gmul_tb17, gtc_gmul_tb18,
  gtc_gmul_tb19, gtc_gmul_tb20
};
static const gtc_leaf_fn gtc_tb_ns_fn[] = {
  NULL, NULL, gtc_gmul_tb2_ns, gtc_gmul_tb3_ns, gtc_gmul_tb4_ns,
  gtc_gmul_tb5_ns, gtc_gmul_tb6_ns, gtc_gmul_tb7_ns, gtc_gmul_tb8_ns,
  gtc_gmul_tb9_ns, gtc_gmul_tb10_ns, gtc_gmul_tb11_ns, gtc_gmul_tb12_ns,
  gtc_gmul_tb13_ns, gtc_gmul_tb14_ns, gtc_gmul_tb15_ns, gtc_gmul_tb16_ns,
  gtc_gmul_tb17_ns, gtc_gmul_tb18_ns, gtc_gmul_tb19_ns, gtc_gmul_tb20_ns
};
#endif
#ifdef GTC_KM_ASM
static const gtc_leaf_fn gtc_km_fn[] = {
  NULL, NULL, gtc_gmul_km2, gtc_gmul_km3, gtc_gmul_km4, gtc_gmul_km5,
  gtc_gmul_km6, gtc_gmul_km7, gtc_gmul_km8, gtc_gmul_km9, gtc_gmul_km10
};
static const gtc_leaf_fn gtc_km_ns_fn[] = {
  NULL, NULL, gtc_gmul_km2_ns, gtc_gmul_km3_ns, gtc_gmul_km4_ns,
  gtc_gmul_km5_ns, gtc_gmul_km6_ns, gtc_gmul_km7_ns, gtc_gmul_km8_ns,
  gtc_gmul_km9_ns, gtc_gmul_km10_ns
};
#endif

/* Size n of gtc_I's products gets algorithm a.  With the assembly, the
   sizes whose product needs no context become leaf functions: the textbook
   product, or from gtc_km_thresh the Karatsuba over {1, i}, and a gtc_22
   level whose halves are both leaf functions of the cap 2.  The cap 2 must
   be set first, and the smaller sizes.  */
static void
set_alg (gtc_ctx *c, int I, mp_size_t n, int a)
{
  gtc_leaf_fn f = NULL, fns = NULL;

  c->alg[GTC_CAP (I)][n] = a;
  if (n >= GTC_LEAF_N)
    return;
#ifdef GTC_TEXTBOOK_ASM
#ifdef GTC_KM_ASM
  if (a == GTC_ALG_TB && n >= 2 && n <= GTC_KM_MAX
      && n >= c->cfg.gtc_km_thresh)
    {
      f = gtc_km_fn[n];
      fns = gtc_km_ns_fn[n];
    }
  else
#endif
  if (a == GTC_ALG_TB && n >= 2 && n <= GTC_TEXTBOOK_LIMIT)
    {
      f = gtc_tb_fn[n];
      fns = gtc_tb_ns_fn[n];
    }
#ifdef GTC_KARA_ASM
  else if (a == GTC_ALG_22 && n >= GTC_GKARA_MIN && n <= GTC_GKARA_MAX
	   && c->leaf_ns[n - (n >> 1)] != NULL && c->leaf_ns[n >> 1] != NULL)
    {
      f = gtc_gkara_fn[n - GTC_GKARA_MIN];
      fns = gtc_gkara_ns_fn[n - GTC_GKARA_MIN];
    }
#endif
  if (f != NULL)
    c->alg[GTC_CAP (I)][n] = GTC_ALG_LEAF;
#endif
  c->leaf[GTC_CAP (I)][n] = f;
  if (I == 2)
    c->leaf_ns[n] = fns;
}

/* The algorithm of a Gaussian product of n-limb components under the cap
   I by the thresholds.  */
static int
gtc_alg_thresh (const gtc_config *cfg, int I, mp_size_t n)
{
  if (I >= 5 && n >= cfg->gtc55_thresh)
    return GTC_ALG_55;
  if (I >= 4 && n >= cfg->gtc44_thresh)
    return GTC_ALG_44;
  if (I >= 3 && n >= cfg->gtc33_thresh)
    return GTC_ALG_33;
  if (n >= cfg->gtc22_thresh)
    return GTC_ALG_22;
  return GTC_ALG_TB;
}

/* A table's list form, "LO-HI:C" or "N:C" items with C one of the
   characters in ok, into rt[0 .. GTC_TABLE_N - 1] ('-' elsewhere).  */
static void
parse_table_list (char *rt, const char *t, const char *ok)
{
  memset (rt, '-', GTC_TABLE_N);
  if (t == NULL)
    return;
  for (;;)
    {
      char *e;
      long lo, hi;
      while (*t == ' ' || *t == ',')
	t++;
      if (*t < '0' || *t > '9')
	return;
      lo = hi = strtol (t, &e, 10);
      t = e;
      if (*t == '-')
	{
	  if (t[1] < '0' || t[1] > '9')
	    return;
	  hi = strtol (t + 1, &e, 10);
	  t = e;
	}
      if (t[0] != ':' || t[1] == 0 || strchr (ok, t[1]) == NULL)
	return;
      for (long n = lo < 0 ? 0 : lo; n <= hi && n < GTC_TABLE_N; n++)
	rt[n] = t[1];
      t += 2;
    }
}

/* The algorithm of gtc_I's products for each size below GTC_TABLE_N:
   gtc_table[GTC_CAP (I)] where it decides within the cap, the thresholds
   elsewhere, and what the size allows.  */
static void
resolve_gtc_table (gtc_ctx *c, int I)
{
  const char *t = c->cfg.gtc_table[GTC_CAP (I)];
  size_t len = t != NULL ? strlen (t) : 0;
  char rt[GTC_TABLE_N];

  if (t != NULL && strchr (t, ':') != NULL)
    {
      parse_table_list (rt, t, "tk2345");
      t = rt;
      len = GTC_TABLE_N;
    }

  for (mp_size_t n = 0; n < GTC_TABLE_N; n++)
    {
      char ch = (size_t) n < len ? t[n] : 0;
      int a = ch == 't' ? GTC_ALG_TB : ch == 'k' || ch == '2' ? GTC_ALG_22
	: ch == '3' ? GTC_ALG_33 : ch == '4' ? GTC_ALG_44
	: ch == '5' ? GTC_ALG_55 : -1;
      if (a < 0 || a > I)
	a = gtc_alg_thresh (&c->cfg, I, n);
      if (a == GTC_ALG_55 && n < GTC_55_MIN)
	a = GTC_ALG_44;
      if (a == GTC_ALG_44 && n < GTC_44_MIN)
	a = GTC_ALG_33;
      if (a == GTC_ALG_33 && n < 5 && n != 3)
	a = GTC_ALG_22;
      if (a == GTC_ALG_TB && n > GTC_TEXTBOOK_LIMIT)
	a = GTC_ALG_22;
      set_alg (c, I, n, a);
    }
}

/* The algorithm of a rational product of n limbs under the cap I by the
   thresholds.  */
static int
r2gtc_alg_thresh (const gtc_config *cfg, int I, mp_size_t n)
{
  if (I >= 5 && n >= cfg->r2gtc55_thresh)
    return R2GTC_ALG_55;
  if (I >= 4 && n >= cfg->r2gtc44_thresh)
    return R2GTC_ALG_44;
  if (I >= 3 && n >= cfg->r2gtc33_thresh)
    return R2GTC_ALG_33;
  if (n >= cfg->r2gtc22_thresh)
    return R2GTC_ALG_22;
  if (n >= cfg->gmp22_thresh)
    return R2GTC_ALG_GMP22;
  return R2GTC_ALG_TB;
}

/* The algorithm of r2gtc_I's rational products for each size below
   GTC_TABLE_N: the thresholds, then r2gtc_table[GTC_CAP (I)]'s items
   where the cap and the size allow them.  */
static void
resolve_r2gtc_table (gtc_ctx *c, int I)
{
  char rt[GTC_TABLE_N];

  parse_table_list (rt, c->cfg.r2gtc_table[GTC_CAP (I)], "tg2345");
  for (mp_size_t n = 0; n < GTC_TABLE_N; n++)
    {
      int a = r2gtc_alg_thresh (&c->cfg, I, n);
      switch (rt[n])
	{
	case 't': a = R2GTC_ALG_TB; break;
	case 'g': if (n >= GTC_GMP22_MIN) a = R2GTC_ALG_GMP22; break;
	case '2': if (n >= 2) a = R2GTC_ALG_22; break;
	case '3': if (I >= 3 && (n == 3 || n >= 5)) a = R2GTC_ALG_33; break;
	case '4': if (I >= 4 && n >= GTC_44_MIN) a = R2GTC_ALG_44; break;
	case '5': if (I >= 5 && n >= GTC_55_MIN) a = R2GTC_ALG_55; break;
	}
      c->ralg[GTC_CAP (I)][n] = a;
    }
}

GTC_INLINE int
r2gtc_alg (const gtc_ctx *c, int I, mp_size_t n)
{
  return n < GTC_TABLE_N ? c->ralg[GTC_CAP (I)][n]
    : r2gtc_alg_thresh (&c->cfg, I, n);
}

gtc_ctx *
gtc_ctx_new (const gtc_config *cfg)
{
  gtc_ctx *c = xmalloc (sizeof *c);
  if (cfg != NULL)
    c->cfg = *cfg;
  else
    gtc_config_default (&c->cfg);
  if (c->cfg.r2gtc33_thresh < 5)
    c->cfg.r2gtc33_thresh = 5;
  if (c->cfg.r2gtc22_thresh < 2)
    c->cfg.r2gtc22_thresh = 2;
  if (c->cfg.gtc33_thresh < 5)
    c->cfg.gtc33_thresh = 5;
  if (c->cfg.gtc22_thresh < 2)
    c->cfg.gtc22_thresh = 2;
  if (c->cfg.gtc44_thresh < GTC_44_MIN)
    c->cfg.gtc44_thresh = GTC_44_MIN;
  if (c->cfg.gtc55_thresh < GTC_55_MIN)
    c->cfg.gtc55_thresh = GTC_55_MIN;
  if (c->cfg.r2gtc44_thresh < GTC_44_MIN)
    c->cfg.r2gtc44_thresh = GTC_44_MIN;
  if (c->cfg.r2gtc55_thresh < GTC_55_MIN)
    c->cfg.r2gtc55_thresh = GTC_55_MIN;
  if (c->cfg.gtc_km_thresh < 2)
    c->cfg.gtc_km_thresh = 2;
  if (c->cfg.gmp22_thresh < GTC_GMP22_MIN)
    c->cfg.gmp22_thresh = GTC_GMP22_MIN;
  /* the textbook product must cover every size below gtc_22 */
  if (c->cfg.gtc22_thresh > GTC_TEXTBOOK_LIMIT + 1)
    c->cfg.gtc22_thresh = GTC_TEXTBOOK_LIMIT + 1;
  for (int I = 2; I <= 5; I++)
    {
      resolve_gtc_table (c, I);
      resolve_r2gtc_table (c, I);
    }
  c->arena = NULL;
  c->cap = c->top = c->vtop = c->peak = 0;
  tc4_pad_table (c->tc4pad);
  return c;
}

void
gtc_ctx_free (gtc_ctx *c)
{
  if (c != NULL)
    {
      free (c->arena);
      free (c);
    }
}

const gtc_config *
gtc_ctx_config (const gtc_ctx *c)
{
  return &c->cfg;
}
