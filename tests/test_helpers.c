/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   White-box tests of gtc.c's internal helpers against mpz, in all branches
   (flag combinations, widths, aliasing), including branches that the
   current interpolation sequences never take: two's complement arithmetic
   with lazy negation flags (reg_add, reg_add_half, reg_add_ext, acc,
   acc_limb, reg_aorsmul_u), abs_sub, cmp_lsh1, recompose, the
   evaluation of the Gaussian Toom-3 (gauss_eval3), the configuration
   clamping
   and the default-context wrappers.  gtc.c is included so that its static
   functions are visible.  */

#include "gtc.c"

#include <string.h>

static unsigned long long rs = 88172645463325252ULL;
static long checks, failures;

static mp_limb_t
rnd (void)
{
  rs ^= rs << 13;
  rs ^= rs >> 7;
  rs ^= rs << 17;
  return rs;
}

/* A limb that is often extreme.  */
static mp_limb_t
limb (void)
{
  switch (rnd () % 7)
    {
    case 0: return 0;
    case 1: return 1;
    case 2: return GMP_NUMB_MAX;
    case 3: return HIGHBIT;
    case 4: return HIGHBIT - 1;
    default: return rnd ();
    }
}

static void
fill (mp_ptr p, mp_size_t n)
{
  for (mp_size_t i = 0; i < n; i++)
    p[i] = limb ();
}

/* v = {p, n} as a natural number.  */
static void
nat_value (mpz_t v, mp_srcptr p, mp_size_t n)
{
  mpz_t ro;			/* read-only view, never cleared */
  mpz_set (v, mpz_roinit_n (ro, p, n));
}

/* v = tc({p, w}), the two's complement value.  */
static void
tc_value (mpz_t v, mp_srcptr p, mp_size_t w)
{
  nat_value (v, p, w);
  if (p[w - 1] & HIGHBIT)
    {
      mpz_t t;
      mpz_init (t);
      mpz_setbit (t, w * GMP_NUMB_BITS);
      mpz_sub (v, v, t);
      mpz_clear (t);
    }
}

/* v = (neg ? -1 : 1) * tc({p, w}).  */
static void
reg_value (mpz_t v, sreg r, mp_size_t w)
{
  tc_value (v, r.p, w);
  if (r.neg)
    mpz_neg (v, v);
}

/* a == b mod B^w ?  */
static int
eq_mod (mpz_t a, mpz_t b, mp_size_t w)
{
  mpz_t d;
  int r;
  mpz_init (d);
  mpz_sub (d, a, b);
  mpz_fdiv_r_2exp (d, d, w * GMP_NUMB_BITS);
  r = mpz_sgn (d) == 0;
  mpz_clear (d);
  return r;
}

static void
check (int ok, const char *what, long iter)
{
  checks++;
  if (!ok && failures++ < 20)
    printf ("FAIL %s (iteration %ld)\n", what, iter);
}

/* {p, n} with sign neg equals want, and its top limb is at most maxtop.  */
static void
check_sm (mpz_t want, mp_srcptr p, mp_size_t n, int neg, mp_limb_t maxtop,
	  const char *what, long iter)
{
  mpz_t v;
  mpz_init (v);
  nat_value (v, p, n);
  if (neg)
    mpz_neg (v, v);
  check (mpz_cmp (v, want) == 0 && p[n - 1] <= maxtop, what, iter);
  mpz_clear (v);
}

int
main (void)
{
  mp_limb_t xb[16], yb[16], db[16], pb[20];
  mpz_t X, Y, E, G, P;
  mpz_inits (X, Y, E, G, P, NULL);

  for (long it = 0; it < 200000; it++)
    {
      mp_size_t w = 1 + rnd () % 6;
      int s = (rnd () & 1) ? 1 : -1, alias = rnd () % 3;
      sreg x = { xb, rnd () & 1 }, y = { yb, rnd () & 1 }, d;
      mp_ptr dp;

      /* reg_add: d = x + s y (mod B^w), d may alias x or y.  */
      fill (xb, w);
      fill (yb, w);
      reg_value (X, x, w);
      reg_value (Y, y, w);
      mpz_set (E, X);
      if (s > 0) mpz_add (E, E, Y); else mpz_sub (E, E, Y);
      dp = alias == 0 ? db : alias == 1 ? xb : yb;
      d = reg_add (dp, x, y, s, w);
      reg_value (G, d, w);
      check (eq_mod (G, E, w), "reg_add", it);

      /* reg_add_half: d = (x + s y) / 2 exactly, for values with headroom
	 (top four bits equal) and x + s y even.  */
      fill (xb, w);
      fill (yb, w);
      xb[w - 1] = (mp_limb_t) ((mp_limb_signed_t) xb[w - 1] >> 3);
      yb[w - 1] = (mp_limb_t) ((mp_limb_signed_t) yb[w - 1] >> 3);
      yb[0] = (yb[0] & ~(mp_limb_t) 1) | (xb[0] & 1);
      reg_value (X, x, w);
      reg_value (Y, y, w);
      mpz_set (E, X);
      if (s > 0) mpz_add (E, E, Y); else mpz_sub (E, E, Y);
      mpz_tdiv_q_2exp (E, E, 1);	/* exact */
      dp = alias == 0 ? db : alias == 1 ? xb : yb;
      d = reg_add_half (dp, x, y, s, w);
      reg_value (G, d, w);
      check (mpz_cmp (G, E) == 0, "reg_add_half", it);

      /* reg_add_ext: y of width yw <= w, sign-extended; d may alias x.  */
      {
	mp_size_t W = 1 + rnd () % 7, yw = 1 + rnd () % W;
	fill (xb, W);
	fill (yb, yw);
	reg_value (X, x, W);
	reg_value (Y, y, yw);
	mpz_set (E, X);
	if (s > 0) mpz_add (E, E, Y); else mpz_sub (E, E, Y);
	d = reg_add_ext (alias == 1 ? xb : db, x, y, yw, s, W);
	reg_value (G, d, W);
	check (eq_mod (G, E, W), "reg_add_ext", it);
      }

      /* acc: {p, len} += q, q of width qw sign-extended, both qw < len and
	 qw >= len.  */
      {
	mp_size_t len = 1 + rnd () % 8, qw = 1 + rnd () % 9;
	sreg q = { yb, rnd () & 1 };
	fill (pb, len);
	fill (yb, qw);
	tc_value (P, pb, len);
	reg_value (Y, q, qw);
	mpz_add (E, P, Y);
	acc (pb, len, q, qw);
	tc_value (G, pb, len);
	check (eq_mod (G, E, len), "acc", it);
      }

      /* acc_limb: {p, len} += v, v a signed limb.  */
      {
	mp_size_t len = 1 + rnd () % 6;
	mp_limb_t v = (rnd () & 1) ? limb () : (mp_limb_t) (rnd () % 9) - 4;
	fill (pb, len);
	tc_value (P, pb, len);
	mpz_set_si (E, (mp_limb_signed_t) v);
	mpz_add (E, E, P);
	acc_limb (pb, len, v);
	tc_value (G, pb, len);
	check (eq_mod (G, E, len), "acc_limb", it);
      }

      /* abs_sub: {rp, an} = |a - b|, rp fresh or equal to a, with equal
	 values and zero high parts made likely.  */
      {
	mp_size_t an = 1 + rnd () % 5, bn = 1 + rnd () % an;
	int in_place = rnd () & 1, neg;
	mpz_t A, Bz, R;
	mpz_inits (A, Bz, R, NULL);
	fill (xb, an);
	fill (yb, bn);
	if (rnd () % 3 == 0)
	  mpn_zero (xb + bn, an - bn);
	if (rnd () % 4 == 0)
	  mpn_copyi (yb, xb, bn);
	nat_value (A, xb, an);
	nat_value (Bz, yb, bn);
	mpz_sub (E, A, Bz);
	dp = in_place ? xb : db;
	if (!in_place)
	  fill (db, an);	/* garbage: the high part must be zeroed */
	neg = abs_sub (dp, xb, an, yb, bn);
	nat_value (R, dp, an);
	if (neg)
	  mpz_neg (R, R);
	check (mpz_cmp (R, E) == 0 && neg == (mpz_sgn (E) < 0), "abs_sub", it);
	mpz_clears (A, Bz, R, NULL);
      }

      /* recompose: C = q0 + q1 X + q2 X^2 + q3 X^3 + q4 X^4 (mod B^(2n+1))
	 for arbitrary q0, q2, q4 in place — q2 in the gap between the other
	 two, where the interpolation writes it, over the two limbs it shares
	 with them — and q1, q3 of width 2k+1 with arbitrary flags.  */
      if (it % 4 == 0)
	{
	  mp_size_t n = 3 + rnd () % 10, k, h, wq, len;
	  mp_limb_t C[64], C2[64], q1b[32], q2b[32], q3b[32], c0, e4;
	  sreg q1 = { q1b, rnd () & 1 }, q3 = { q3b, rnd () & 1 };
	  mpz_t Q, T;
	  if (n == 4)
	    n = 5;
	  k = (n + 2) / 3;
	  h = n - 2 * k;
	  wq = 2 * k + 1;
	  len = 2 * n + 1;
	  mpz_inits (Q, T, NULL);
	  fill (C, len);		/* q0 in C[0..wq), q4 in C[4k..len) */
	  fill (C + wq, 4 * k - wq);	/* don't care */
	  fill (q1b, wq);
	  fill (q2b, wq);
	  fill (q3b, wq);
	  tc_value (E, C, wq);				/* q0 */
	  tc_value (T, C + 4 * k, 2 * h + 1);		/* q4 X^4 */
	  mpz_mul_2exp (T, T, 4 * k * GMP_NUMB_BITS);
	  mpz_add (E, E, T);
	  reg_value (Q, q1, wq);
	  mpz_mul_2exp (Q, Q, k * GMP_NUMB_BITS);
	  mpz_add (E, E, Q);
	  tc_value (Q, q2b, wq);	/* q2: its limbs are its value */
	  mpz_mul_2exp (Q, Q, 2 * k * GMP_NUMB_BITS);
	  mpz_add (E, E, Q);
	  reg_value (Q, q3, wq);
	  mpz_mul_2exp (Q, Q, 3 * k * GMP_NUMB_BITS);
	  mpz_add (E, E, Q);
	  c0 = C[2 * k];		/* the limbs q2's write covers */
	  e4 = C[4 * k];
	  mpn_copyi (C + 2 * k, q2b, wq);
	  mpn_copyi (C2, C, len);
	  recompose (C, n, k, q1, q3, c0, e4);
	  tc_value (G, C, len);
	  check (eq_mod (G, E, len), "recompose", it);
#ifdef GTC_ACC_ASM
	  /* gtc_33_recompose_k: the same even part in both components,
	     q1 and q3 with flags of their own in each, q1 a sum q1 + y as
	     the fused interpolation leaves it (y arbitrary: the sum is never
	     stored, so it may exceed the width).  */
	  {
	    mp_limb_t D[64], yb[32];
	    sreg q1i = { q1b, rnd () & 1 }, q3i = { q3b, rnd () & 1 };
	    sreg yre = { yb, rnd () & 1 }, yim = { yb, rnd () & 1 };
	    gauss_odd o = { .q1re = q1, .q1im = q1i, .q3re = q3, .q3im = q3i,
			    .c0re = c0, .c0im = c0, .e4re = e4, .e4im = e4,
			    .y1re = yre, .y1im = yim };
	    mpz_t B0, Ei, V;
	    mpz_inits (B0, Ei, V, NULL);
	    fill (yb, wq);
	    /* B0 = q0 + q2 X^2 + q4 X^4: E without q1 and q3 */
	    mpz_set (B0, E);
	    reg_value (V, q1, wq);
	    mpz_mul_2exp (V, V, k * GMP_NUMB_BITS);
	    mpz_sub (B0, B0, V);
	    reg_value (V, q3, wq);
	    mpz_mul_2exp (V, V, 3 * k * GMP_NUMB_BITS);
	    mpz_sub (B0, B0, V);
	    for (int im = 0; im < 2; im++)
	      {
		sreg a = im ? q1i : q1, b = im ? q3i : q3, y = im ? yim : yre;
		mpz_set (Ei, B0);
		reg_value (V, a, wq);
		{ mpz_t Y2; mpz_init (Y2); reg_value (Y2, y, wq);
		  mpz_add (V, V, Y2); mpz_clear (Y2); }
		mpz_mul_2exp (V, V, k * GMP_NUMB_BITS);
		mpz_add (Ei, Ei, V);
		reg_value (V, b, wq);
		mpz_mul_2exp (V, V, 3 * k * GMP_NUMB_BITS);
		mpz_add (Ei, Ei, V);
		if (im == 0)
		  mpz_set (E, Ei);
	      }
	    /* E: the real component's value, Ei the imaginary one's */
	    mpn_copyi (C, C2, len);
	    mpn_copyi (D, C2, len);
	    gtc_33_recompose_k (C, D, n, k, &o);
	    tc_value (G, C, len);
	    check (eq_mod (G, E, len), "recompose_k re", it);
	    tc_value (G, D, len);
	    check (eq_mod (G, Ei, len), "recompose_k im", it);
	    mpz_clears (B0, Ei, V, NULL);
	  }
#endif
	  mpz_clears (Q, T, NULL);
	}
    }

#ifdef GTC_ACC_ASM
  /* The kernels against GMP, every length to 40, every sign, in place.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = rnd () % 41;
      unsigned sg = rnd () & 3;
      mp_limb_t u[48], v[48], x[48], r[48], a[48], b[48], c, want, cc;
      fill (u, n); fill (v, n); fill (x, n);
      /* gtc_aors2_n */
      mpn_copyi (a, u, n);
      mpn_copyi (b, v, n);
      c = gtc_aors2_n (a, b, x, u, n, sg);
      mpn_copyi (r, u, n);
      want = n == 0 ? !!(sg & 1)
	: (sg & 1) ? !mpn_sub_n (r, r, x, n) : mpn_add_n (r, r, x, n);
      check (mpn_cmp (a, r, n) == 0 && (c & 1) == want, "aors2 P", it);
      mpn_copyi (r, v, n);
      want = n == 0 ? !!(sg & 2)
	: (sg & 2) ? !mpn_sub_n (r, r, u, n) : mpn_add_n (r, r, u, n);
      check (mpn_cmp (b, r, n) == 0 && (c >> 1) == want, "aors2 Q", it);
      /* gtc_add3_n, in place (rp = up) or not */
      {
	int inplace = rnd () & 1;
	mp_limb_signed_t sc = 0;
	mpn_copyi (a, u, n);
	c = gtc_add3_n (inplace ? a : b, a, v, x, n, sg);
	mpn_copyi (r, u, n);
	if (n)
	  {
	    sc += (sg & 1) ? -(mp_limb_signed_t) mpn_sub_n (r, r, v, n)
			   : (mp_limb_signed_t) mpn_add_n (r, r, v, n);
	    sc += (sg & 2) ? -(mp_limb_signed_t) mpn_sub_n (r, r, x, n)
			   : (mp_limb_signed_t) mpn_add_n (r, r, x, n);
	  }
	cc = c - (sg & 1) - (sg >> 1);
	check (mpn_cmp (inplace ? a : b, r, n) == 0
	       && (mp_limb_signed_t) cc == sc && c <= 2, "add3", it);
      }
#ifdef GTC_PAIR_ASM
      /* gtc_aors2x_n: dP = u +- v into the place of u, of v, or apart;
	 dQ = x +- u apart */
      {
	int mode = rnd () % 3;
	mp_limb_t d[48], e[48];
	mp_ptr dp = mode == 0 ? a : mode == 1 ? b : r;
	mpn_copyi (a, u, n);
	mpn_copyi (b, v, n);
	c = gtc_aors2x_n (dp, d, a, x, b, u, n, sg);
	mpn_copyi (e, u, n);
	want = n == 0 ? !!(sg & 1)
	  : (sg & 1) ? !mpn_sub_n (e, e, v, n) : mpn_add_n (e, e, v, n);
	check (mpn_cmp (dp, e, n) == 0 && (c & 1) == want, "aors2x P", it);
	mpn_copyi (e, x, n);
	want = n == 0 ? !!(sg & 2)
	  : (sg & 2) ? !mpn_sub_n (e, e, u, n) : mpn_add_n (e, e, u, n);
	check (mpn_cmp (d, e, n) == 0 && (c >> 1) == want, "aors2x Q", it);
      }
#endif
    }
#endif

#if GTC_X86_AVX512
  /* The AVX-512 pass gtc_add3v_n (gtc_avx512_x86.S) against GMP, at
     every length to 140 and then every 7th to 300 (heads of 0..7 limbs,
     odd and even block counts), every sign, with rp apart or equal to each source, operands
     whose lanes wrap as they take a carry in (runs of all-ones and zeros,
     which force the scalar redo of a block), and a canary above n.  */
  {
    enum { NM = 300, CAN = 8 };
    static mp_limb_t u[NM], v[NM], w[NM], r[NM + CAN], t[NM + CAN], s[NM];
    for (int pat = 0; pat < 7; pat++)
      for (mp_size_t n = 0; n <= NM; n += n < 140 ? 1 : 7)
	for (int rep = 0; rep < 6; rep++)
	  {
	    mp_limb_t *ops[3] = { u, v, w };
	    for (int o = 0; o < 3; o++)
	      for (mp_size_t i = 0; i < n; i++)
		{
		  mp_limb_t x = rnd ();
		  switch ((pat + o * (rep & 1)) % 7)
		    {
		    case 0: break;
		    case 1: x = GMP_NUMB_MAX; break;
		    case 2: x = 0; break;
		    case 3: x = (rnd () & 3) == 0 ? GMP_NUMB_MAX : x; break;
		    case 4: x = (rnd () & 1) ? ((rnd () & 1) ? GMP_NUMB_MAX : 0) : x;
		      break;
		    case 5: x = (rnd () & 1) ? GMP_NUMB_MAX - (rnd () & 1)
			      : 1 + (rnd () & 1); break;
		    default: x &= 3; break;
		    }
		  ops[o][i] = x;
		}
	    unsigned sg = rep & 3;
	    int al = rnd () % 4;	/* rp apart, = up, = vp, = wp */
	    mp_limb_t c;
	    mp_limb_signed_t sc = 0;
	    /* gtc_add3v_n against two GMP passes */
	    mpn_copyi (s, u, n);
	    if (n)
	      {
		sc += (sg & 1) ? -(mp_limb_signed_t) mpn_sub_n (s, s, v, n)
			       : (mp_limb_signed_t) mpn_add_n (s, s, v, n);
		sc += (sg & 2) ? -(mp_limb_signed_t) mpn_sub_n (s, s, w, n)
			       : (mp_limb_signed_t) mpn_add_n (s, s, w, n);
	      }
	    for (int i = 0; i < CAN; i++)
	      r[n + i] = t[n + i] = rnd ();
	    if (al)
	      mpn_copyi (r, ops[al - 1], n);
	    c = gtc_add3v_n (r, al == 1 ? r : u, al == 2 ? r : v,
			     al == 3 ? r : w, n, sg);
	    check (mpn_cmp (r, s, n) == 0
		   && (mp_limb_signed_t) (c - (sg & 1) - (sg >> 1)) == sc
		   && c <= 2 + (sg & 1) + (sg >> 1)
		   && mpn_cmp (r + n, t + n, CAN) == 0, "add3v", n);
	  }
  }
#endif

  /* The passes this build uses (GTC_ADD_N, GTC_SUB_N: our kernels or
     GMP's) and gtc_add, gtc_sub, against mpz: every length to 40 and some
     to 300 (every entry into the eight-limb turn, one turn and several),
     into a fresh area, in place over u or over v, the carry or borrow
     out included; operands often all ones or zero, so that carries run
     through whole turns.  */
  for (long it = 0; it < 60000; it++)
    {
      mp_size_t n = it % 8 == 0 ? 41 + rnd () % 260 : 1 + rnd () % 40;
      mp_size_t m = 1 + rnd () % n;
      int sub = rnd () & 1, alias = rnd () % 3, kind = rnd () % 4;
      mp_limb_t u[304], v[304], r[304], c;
      mp_ptr d = alias == 1 ? u : alias == 2 ? v : r;
      fill (u, n);
      fill (v, n);
      if (kind == 1)		/* all ones against a small or zero v */
	{
	  for (mp_size_t i = 0; i < n; i++)
	    u[i] = GMP_NUMB_MAX, v[i] = sub ? GMP_NUMB_MAX : 0;
	  v[0] = rnd () % 2;
	}
      else if (kind == 2)	/* zero minus something: borrow all along */
	{
	  for (mp_size_t i = 0; i < n; i++)
	    u[i] = sub ? 0 : GMP_NUMB_MAX, v[i] = 0;
	  v[0] = 1 + rnd () % 3;
	}
      nat_value (X, u, n);
      nat_value (Y, v, n);
      if (sub) mpz_sub (E, X, Y); else mpz_add (E, X, Y);
      c = sub ? GTC_SUB_N (d, u, v, n) : GTC_ADD_N (d, u, v, n);
      nat_value (G, d, n);
      if (c)			/* the carry adds B^n, the borrow takes it */
	{
	  mpz_set_ui (P, 1);
	  mpz_mul_2exp (P, P, GMP_NUMB_BITS * n);
	  if (sub) mpz_sub (G, G, P); else mpz_add (G, G, P);
	}
      check (mpz_cmp (E, G) == 0 && c <= 1, sub ? "sub_n" : "add_n", it);

      /* gtc_add, gtc_sub: {u, n} +- {v, m}, m <= n */
      fill (u, n);
      fill (v, m);
      if (kind == 1)
	for (mp_size_t i = 0; i < n; i++)
	  u[i] = sub ? 0 : GMP_NUMB_MAX;
      nat_value (X, u, n);
      nat_value (Y, v, m);
      if (sub) mpz_sub (E, X, Y); else mpz_add (E, X, Y);
      d = alias == 1 ? u : r;
      c = sub ? gtc_sub (d, u, n, v, m) : gtc_add (d, u, n, v, m);
      nat_value (G, d, n);
      if (c)
	{
	  mpz_set_ui (P, 1);
	  mpz_mul_2exp (P, P, GMP_NUMB_BITS * n);
	  if (sub) mpz_sub (G, G, P); else mpz_add (G, G, P);
	}
      check (mpz_cmp (E, G) == 0 && c <= 1, sub ? "gtc_sub" : "gtc_add", it);
    }

  /* The halvings this build uses (GTC_RSH1ADD_N, GTC_RSH1SUB_N: our
     kernel, GMP's or the fallback), every length to 40, in place or not,
     against mpz: {r, n} = (u +- v) / 2 with the carry or the borrow
     shifted into the top bit, as GMP's mpn_rsh1add_n and mpn_rsh1sub_n
     define it (reg_add_half overwrites that bit); or, with gtc's kernels
     (GTC_RSH1_SIGNED), the sum modulo B^n read as a signed number and
     halved by an arithmetic shift.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = 1 + rnd () % 40;
      int sub = rnd () & 1, inplace = rnd () & 1;
      mp_limb_t u[48], v[48], r[48];
      mp_ptr d = inplace ? u : r;
      fill (u, n);
      fill (v, n);
      nat_value (X, u, n);
      nat_value (Y, v, n);
      if (sub) mpz_sub (E, X, Y); else mpz_add (E, X, Y);
#if GTC_RSH1_SIGNED
      mpz_fdiv_r_2exp (E, E, GMP_NUMB_BITS * n);
      if (mpz_tstbit (E, GMP_NUMB_BITS * n - 1))
	{
	  mpz_set_ui (G, 1);
	  mpz_mul_2exp (G, G, GMP_NUMB_BITS * n);
	  mpz_sub (E, E, G);		/* the signed value */
	}
      mpz_fdiv_q_2exp (E, E, 1);	/* floor: the arithmetic shift */
      mpz_fdiv_r_2exp (E, E, GMP_NUMB_BITS * n);
#else
      mpz_fdiv_r_2exp (E, E, GMP_NUMB_BITS * n + 1);
      mpz_fdiv_q_2exp (E, E, 1);
#endif
      if (sub)
	GTC_RSH1SUB_N (d, u, v, n);
      else
	GTC_RSH1ADD_N (d, u, v, n);
      nat_value (G, d, n);
      check (mpz_cmp (E, G) == 0, sub ? "rsh1sub" : "rsh1add", it);
    }

#ifdef GTC_ACC_ASM
  /* The Toom-4's fused division (gtc_div15_lsh_n), every length to 40 and
     every sign, in place or not, with extreme limbs: x is made 15 q -+ 4 y
     -+ 8 z modulo B^n, and the kernel must give back q, and leave the
     limb after the result alone.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = rnd () % 41;
      unsigned sg = rnd () & 3;
      int inpl = rnd () & 1;
      mp_limb_t x[42], y[41], z[41], q[41], t[41], r[42];
      fill (y, 41);
      fill (z, 41);
      fill (q, 41);
      if (n > 0)
	{
	  mpn_mul_1 (x, q, n, 15);
	  mpn_lshift (t, y, n, 2);
	  if (sg & 1)
	    mpn_add_n (x, x, t, n);
	  else
	    mpn_sub_n (x, x, t, n);
	  mpn_lshift (t, z, n, 3);
	  if (sg & 2)
	    mpn_add_n (x, x, t, n);
	  else
	    mpn_sub_n (x, x, t, n);
	}
      x[n] = r[n] = 0x5a5a5a5a5a5a5a5aULL;
      gtc_div15_lsh_n (inpl ? x : r, x, y, z, n, sg);
      check ((n == 0 || mpn_cmp (inpl ? x : r, q, n) == 0)
	     && (inpl ? x : r)[n] == 0x5a5a5a5a5a5a5a5aULL,
	     "gtc_div15_lsh_n", it);
    }
#endif

#ifdef GTC_ACC_ASM
  /* The Toom-4 style reduction's fused division (gtc_div5_lsh_n), every
     length to 40, both signs, in place or not, with extreme limbs: x is
     made 5 q -+ 4 y modulo B^n, and the kernel must give back q, and leave
     the limb after the result alone.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = rnd () % 41;
      unsigned sub = rnd () & 1;
      int inpl = rnd () & 1;
      mp_limb_t x[42], y[41], q[41], t[41], r[42];
      fill (y, 41);
      fill (q, 41);
      if (n > 0)
	{
	  mpn_mul_1 (x, q, n, 5);
	  mpn_lshift (t, y, n, 2);
	  if (sub)
	    mpn_add_n (x, x, t, n);
	  else
	    mpn_sub_n (x, x, t, n);
	}
      x[n] = r[n] = 0x5a5a5a5a5a5a5a5aULL;
      gtc_div5_lsh_n (inpl ? x : r, x, y, n, sub);
      check ((n == 0 || mpn_cmp (inpl ? x : r, q, n) == 0)
	     && (inpl ? x : r)[n] == 0x5a5a5a5a5a5a5a5aULL,
	     "gtc_div5_lsh_n", it);
    }
#endif

#if GTC_RTC4_SHLD
  /* gtc_div5_rlsh_n: (x + 4 y)/5, or (4 y - x)/5 with rsb; x is made
     5 q - 4 y, or 4 y - 5 q.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = rnd () % 41;
      unsigned rsb = rnd () & 1;
      int inpl = rnd () & 1;
      mp_limb_t x[42], y[41], q[41], t[41], r[42];
      fill (y, 41);
      fill (q, 41);
      if (n > 0)
	{
	  mpn_mul_1 (x, q, n, 5);
	  mpn_lshift (t, y, n, 2);
	  if (rsb)
	    mpn_sub_n (x, t, x, n);
	  else
	    mpn_sub_n (x, x, t, n);
	}
      x[n] = r[n] = 0x5a5a5a5a5a5a5a5aULL;
      gtc_div5_rlsh_n (inpl ? x : r, x, y, n, rsb);
      check ((n == 0 || mpn_cmp (inpl ? x : r, q, n) == 0)
	     && (inpl ? x : r)[n] == 0x5a5a5a5a5a5a5a5aULL,
	     "gtc_div5_rlsh_n", it);
    }
#endif

  /* tc_abs_sub, |X - Y| of two two's complement values and whether X < Y,
     with values of a few bits around 0 on top and equal top limbs (so
     that the comparison reaches the loop and its end), against mpz.  */
  for (long it = 0; it < 100000; it++)
    {
      mp_size_t w = 3 + rnd () % 8, j;
      mp_limb_t x[12], y[12], d[12];
      fill (x, w);
      fill (y, w);
      x[w - 1] = (mp_limb_t) ((long) (rnd () % 9) - 4);
      y[w - 1] = rnd () % 2 ? x[w - 1] : (mp_limb_t) ((long) (rnd () % 9) - 4);
      j = rnd () % (w + 1);		/* equal from limb j up */
      for (mp_size_t i = j; i < w - 1; i++)
	y[i] = x[i];
      tc_value (X, x, w);
      tc_value (Y, y, w);
      mpz_sub (E, X, Y);
      int lt = tc_abs_sub (d, x, y, w);
      mpz_abs (E, E);
      nat_value (G, d, w);
      check (mpz_cmp (E, G) == 0 && lt == (mpz_cmp (X, Y) < 0),
	     "tc_abs_sub", it);
    }

#ifdef GTC_ACC_ASM
  /* The Toom-5's division by 5 of a sum already divided by 4
     (gtc_div5_add_n), every length to 40, all three modes, in place on x
     or y or not, with extreme limbs: x made 5 q - y, 5 q + y or y - 5 q
     modulo B^n, and the kernel must give back q, and leave the limb after
     the result alone.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = rnd () % 41;
      unsigned mode = (unsigned) (rnd () % 3);
      int where = (int) (rnd () % 3);	/* 0: apart; 1, 2: on x, y */
      mp_limb_t x[42], y[42], q[41], t[41], r[42];
      fill (y, 41);
      fill (q, 41);
      if (n > 0)
	{
	  mpn_mul_1 (t, q, n, 5);
	  if (mode == 0)
	    mpn_sub_n (x, t, y, n);
	  else if (mode == 1)
	    mpn_add_n (x, t, y, n);
	  else
	    mpn_sub_n (x, y, t, n);
	}
      x[n] = y[n] = r[n] = 0x5a5a5a5a5a5a5a5aULL;
      mp_ptr d = where == 1 ? x : where == 2 ? y : r;
      gtc_div5_add_n (d, x, y, n, mode);
      check ((n == 0 || mpn_cmp (d, q, n) == 0)
	     && d[n] == 0x5a5a5a5a5a5a5a5aULL, "gtc_div5_add_n", it);
    }
  /* The shifted sums of the Toom-5 (gtc_rsh2add_n, gtc_rsh2sub_n,
     gtc_rsh2add3_n, gtc_rsh3add3_n), every length to 40, in place on any
     operand or not, with extreme limbs, against mpz: the sum modulo B^n
     read as a two's complement number and shifted arithmetically, and
     for the first two the bits shifted out.  */
  for (long it = 0; it < 80000; it++)
    {
      mp_size_t n = 1 + rnd () % 40;
      int kind = (int) (rnd () % 4);	/* rsh2add, rsh2sub, rsh2add3, rsh3add3 */
      unsigned comp = rnd () & 1, sh = kind == 3 ? 3 : 2;
      int where = (int) (rnd () % 4);	/* 0: r apart; 1..3: on a, b, c */
      mp_limb_t a[48], b[48], c[48], r[48], ret, low;
      for (mp_size_t i = 0; i < n; i++)
	{
	  a[i] = limb ();
	  b[i] = limb ();
	  c[i] = limb ();
	}
      nat_value (X, a, n);
      nat_value (Y, b, n);
      nat_value (G, c, n);
      if (kind < 2)
	{
	  if (kind) mpz_sub (E, X, Y); else mpz_add (E, X, Y);
	  if (where == 3)
	    where = 0;
	}
      else
	{
	  mpz_add (E, Y, G);
	  if (comp) mpz_sub (E, E, X); else mpz_add (E, E, X);
	}
      mpz_fdiv_r_2exp (E, E, GMP_NUMB_BITS * n);
      low = mpz_getlimbn (E, 0) & ((1u << sh) - 1);
      if (mpz_tstbit (E, GMP_NUMB_BITS * n - 1))
	{
	  mpz_set_ui (X, 1);
	  mpz_mul_2exp (X, X, GMP_NUMB_BITS * n);
	  mpz_sub (E, E, X);
	}
      mpz_fdiv_q_2exp (E, E, sh);
      mpz_fdiv_r_2exp (E, E, GMP_NUMB_BITS * n);
      mp_ptr d = where == 1 ? a : where == 2 ? b : where == 3 ? c : r;
      if (kind == 0)
	ret = gtc_rsh2add_n (d, a, b, n);
      else if (kind == 1)
	ret = gtc_rsh2sub_n (d, a, b, n);
      else
	ret = (kind == 2 ? gtc_rsh2add3_n : gtc_rsh3add3_n) (d, a, b, c, n,
							      comp);
      nat_value (G, d, n);
      check (mpz_cmp (E, G) == 0 && (kind >= 2 || ret == low),
	     "rsh2add/rsh2add3/rsh3add3", it);
    }
#endif

#if defined (GTC_ASM_X86) || (defined (__aarch64__) && defined (GTC_ACC_ASM))
  /* The Toom-5s' second kernels.  gtc_div5_add_sub_n and
     gtc_div5_add_acc_n: gtc_div5_add_n's q (x as in its test above) and,
     in the same pass, z - q into sp (z = y with mode bit 2, else x), or
     ap +- q with the carry or borrow; every length to 40, all modes, the
     acc kernel also in place on x or y.  gtc_rsh3add_n, gtc_rsh3sub_n:
     (a +- b mod B^n) >> 3, arithmetic, and the bits shifted out.
     gtc_t5r_n: d = x0 - 4 x4 (or 4 x4 - x0) and dm = 4 y - d over m
     limbs (a multiple of 4), the shifted limbs taking the top bits of
     x4[-1] and y[-1], the borrows in and out; against GMP's shifts and
     subtractions.  The limb after each result left alone.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = rnd () % 41;
      unsigned mode = (unsigned) (rnd () % 3) | (rnd () & 4);
      int acc = rnd () & 1, where = acc ? (int) (rnd () % 3) : 0;
      mp_limb_t x[42], y[42], q[41], t[41], r[42], z[42], ref[42], cr = 0, c;
      fill (y, 41);
      fill (q, 41);
      fill (z, 41);
      if (n > 0)
	{
	  mpn_mul_1 (t, q, n, 5);
	  if ((mode & 3) == 0)
	    mpn_sub_n (x, t, y, n);
	  else if ((mode & 3) == 1)
	    mpn_add_n (x, t, y, n);
	  else
	    mpn_sub_n (x, y, t, n);
	}
      mpn_copyi (ref, z, n);
      if (n > 0)
	{
	  if (!acc)
	    mpn_sub_n (ref, mode & 4 ? y : x, q, n);
	  else
	    cr = mode & 4 ? mpn_sub_n (ref, ref, q, n)
	      : mpn_add_n (ref, ref, q, n);
	}
      x[n] = y[n] = r[n] = z[n] = 0x5a5a5a5a5a5a5a5aULL;
      mp_ptr d = where == 1 ? x : where == 2 ? y : r;
      if (acc)
	c = gtc_div5_add_acc_n (d, x, y, n, mode, z);
      else
	{
	  gtc_div5_add_sub_n (d, x, y, n, mode, z);
	  c = cr;
	}
      check ((n == 0 || (mpn_cmp (d, q, n) == 0 && mpn_cmp (z, ref, n) == 0))
	     && c == cr && d[n] == 0x5a5a5a5a5a5a5a5aULL
	     && z[n] == 0x5a5a5a5a5a5a5a5aULL,
	     acc ? "gtc_div5_add_acc_n" : "gtc_div5_add_sub_n", it);
    }
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = 1 + rnd () % 40;
      int sub = rnd () & 1, where = (int) (rnd () % 3);
      mp_limb_t a[42], b[42], r[42], ret, low;
      fill (a, n);
      fill (b, n);
      nat_value (X, a, n);
      nat_value (Y, b, n);
      if (sub) mpz_sub (E, X, Y); else mpz_add (E, X, Y);
      mpz_fdiv_r_2exp (E, E, GMP_NUMB_BITS * n);
      low = mpz_getlimbn (E, 0) & 7;
      if (mpz_tstbit (E, GMP_NUMB_BITS * n - 1))
	{
	  mpz_set_ui (X, 1);
	  mpz_mul_2exp (X, X, GMP_NUMB_BITS * n);
	  mpz_sub (E, E, X);
	}
      mpz_fdiv_q_2exp (E, E, 3);
      mpz_fdiv_r_2exp (E, E, GMP_NUMB_BITS * n);
      mp_ptr d = where == 1 ? a : where == 2 ? b : r;
      d[n] = 0x5a5a5a5a5a5a5a5aULL;
      ret = (sub ? gtc_rsh3sub_n : gtc_rsh3add_n) (d, a, b, n);
      nat_value (G, d, n);
      check (mpz_cmp (E, G) == 0 && ret == low
	     && d[n] == 0x5a5a5a5a5a5a5a5aULL, "gtc_rsh3add_n/sub_n", it);
    }
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t m = 4 * (1 + rnd () % 10);
      unsigned b = rnd () & 1, cb = rnd () & 1, neg = rnd () & 1, bo, co;
      mp_limb_t x0[42], x4[42], y[42], d[42], dm[42], f[42], g[42], rd[42],
	rdm[42], ret;
      fill (x0, m);
      fill (x4, m + 1);
      fill (y, m + 1);
      mpn_lshift (f, x4, m + 1, 2);	/* 4 x4 from x4[-1], limbs 1..m */
      mpn_lshift (g, y, m + 1, 2);
      bo = neg ? mpn_sub_n (rd, f + 1, x0, m) : mpn_sub_n (rd, x0, f + 1, m);
      bo |= mpn_sub_1 (rd, rd, m, b);
      co = mpn_sub_n (rdm, g + 1, rd, m);
      co |= mpn_sub_1 (rdm, rdm, m, cb);
      d[m] = dm[m] = 0x5a5a5a5a5a5a5a5aULL;
      ret = gtc_t5r_n (d, dm, x0, x4 + 1, y + 1,
		       (mp_limb_t) m << 3 | neg << 2 | cb << 1 | b);
      check (mpn_cmp (d, rd, m) == 0 && mpn_cmp (dm, rdm, m) == 0
	     && ret == (bo | co << 1) && d[m] == 0x5a5a5a5a5a5a5a5aULL
	     && dm[m] == 0x5a5a5a5a5a5a5a5aULL, "gtc_t5r_n", it);
    }
#endif

#ifdef GTC_ASM_X86
  /* gtc_aorslsh_n: x +- 2^s y and 2^s y - x, every s from 1 to 63, every
     length to 40, in place or not, against mpn_addmul_1 and mpn_submul_1
     by 2^s, the limb above or the borrow included, and for 2^s y - x
     against the shift and mpn_sub_n, the bits shifted out minus the
     borrow.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = rnd () % 41;
      unsigned s = rnd () & 1 ? 1 + (unsigned) (rnd () % 3)
	: 1 + (unsigned) (rnd () % 63), sub = (unsigned) (rnd () % 3);
      int inpl = rnd () & 1;
      mp_limb_t x[42], y[41], ref[41], r[42], c, cr = 0;
      fill (x, 41);
      fill (y, 41);
      mpn_copyi (ref, x, n);
      if (n > 0 && sub == 2)
	{
	  mp_limb_t hi = mpn_lshift (ref, y, n, s);
	  cr = hi - mpn_sub_n (ref, ref, x, n);
	}
      else if (n > 0)
	cr = sub ? mpn_submul_1 (ref, y, n, (mp_limb_t) 1 << s)
	  : mpn_addmul_1 (ref, y, n, (mp_limb_t) 1 << s);
      if (inpl)
	mpn_copyi (r, x, n);
      r[n] = 0x5a5a5a5a5a5a5a5aULL;
      c = gtc_aorslsh_n (r, inpl ? r : x, y, n, s, sub);
      check ((n == 0 || mpn_cmp (r, ref, n) == 0) && c == cr
	     && r[n] == 0x5a5a5a5a5a5a5a5aULL, "gtc_aorslsh_n", it);
    }
#endif

#ifdef GTC_ACC_ASM
  /* The Toom-4's evaluation at 2 (gtc_addlsh123_n), every length to 40,
     in place or not, with extreme limbs, against GMP's shifts and sums,
     the limb above included; the limb after the result left alone.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = 1 + rnd () % 40;
      int inpl = rnd () & 1;
      mp_limb_t a[41], b[41], c[41], d[41], t[41], ref[41], r[41], top;
      fill (a, n);
      fill (b, n);
      fill (c, n);
      fill (d, n);
      mpn_copyi (ref, a, n);
      ref[n] = 0;
      t[n] = mpn_lshift (t, b, n, 1);
      ref[n] += t[n] + mpn_add_n (ref, ref, t, n);
      t[n] = mpn_lshift (t, c, n, 2);
      ref[n] += t[n] + mpn_add_n (ref, ref, t, n);
      t[n] = mpn_lshift (t, d, n, 3);
      ref[n] += t[n] + mpn_add_n (ref, ref, t, n);
      if (inpl)
	mpn_copyi (r, a, n);
      r[n] = 0x5a5a5a5a5a5a5a5aULL;
      top = gtc_addlsh123_n (r, inpl ? r : a, b, c, d, n);
      check (mpn_cmp (r, ref, n) == 0 && top == ref[n]
	     && r[n] == 0x5a5a5a5a5a5a5a5aULL, "gtc_addlsh123_n", it);
    }
#endif

#ifdef GTC_ASM_X86
  /* The Toom-4 style reduction's x +- c u (gtc_aorsmul_1), every length
     to 40, both signs, small and full multipliers, in place, with extreme
     limbs, against GMP's mpn_addmul_1 and mpn_submul_1, the high limb or
     borrow included; the limb after the result left alone.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = rnd () % 41;
      unsigned sub = rnd () & 1;
      mp_limb_t x[42], u[41], ref[41], v, c = 0, cr = 0;
      fill (x, 41);
      fill (u, 41);
      fill (&v, 1);
      if (rnd () & 1)
	v = v % 16;
      mpn_copyi (ref, x, n);
      if (n > 0)
	cr = sub ? mpn_submul_1 (ref, u, n, v) : mpn_addmul_1 (ref, u, n, v);
      x[n] = 0x5a5a5a5a5a5a5a5aULL;
      c = gtc_aorsmul_1 (x, u, n, v, sub);
      check ((n == 0 || mpn_cmp (x, ref, n) == 0) && c == cr
	     && x[n] == 0x5a5a5a5a5a5a5a5aULL, "gtc_aorsmul_1", it);
    }
#endif

#if GTC_TC4_SLK
  /* The Toom-4's x +- 2 y (gtc_aorslsh1_n), every length to 40, both
     signs, in place or not, with extreme limbs, against GMP's
     mpn_addlsh1_n and mpn_sublsh1_n, the carry or borrow included; the
     limb after the result left alone.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = rnd () % 41;
      unsigned sub = rnd () & 1;
      int inpl = rnd () & 1;
      mp_limb_t x[42], y[41], ref[41], r[42], c = 0, cr = 0;
      fill (x, 41);
      fill (y, 41);
      if (n > 0)
	cr = sub ? mpn_sublsh1_n (ref, x, y, n) : mpn_addlsh1_n (ref, x, y, n);
      if (inpl)
	mpn_copyi (r, x, n);
      r[n] = 0x5a5a5a5a5a5a5a5aULL;
      c = gtc_aorslsh1_n (r, inpl ? r : x, y, n, sub);
      check ((n == 0 || mpn_cmp (r, ref, n) == 0) && c == cr
	     && r[n] == 0x5a5a5a5a5a5a5a5aULL, "gtc_aorslsh1_n", it);
    }
#endif

  /* The Toom-4 style reduction's helpers.  reg_aorsmul_u: x += s c u at
     width w, u >= 0 of un < w limbs, every flag and sign, c 1 (the plain
     addition) or small.  */
  for (long it = 0; it < 100000; it++)
    {
      mp_size_t w = 2 + rnd () % 7, un = 1 + rnd () % (w - 1);
      int s = (rnd () & 1) ? 1 : -1;
      static const mp_limb_t cs[] = { 1, 1, 3, 5, 8, 10 };
      mp_limb_t c = cs[rnd () % 6], u[8];
      sreg x = { xb, rnd () & 1 };
      fill (xb, w);
      fill (u, un);
      reg_value (E, x, w);
      nat_value (G, u, un);
      mpz_mul_ui (G, G, c);
      if (s > 0) mpz_add (E, E, G); else mpz_sub (E, E, G);
      x = reg_aorsmul_u (x, u, un, c, s, w);
      reg_value (G, x, w);
      check (x.p == xb && eq_mod (E, G, w), "reg_aorsmul_u", it);
    }
  /* cmp_lsh1, the sign of 2t - a in the evaluation at (-1-i,1): random
     limbs, and a = 2t with t's top bit clear (equal), then off in one
     limb, so that the comparison runs down to any limb.  */
  for (long it = 0; it < 100000; it++)
    {
      mp_size_t k = 1 + rnd () % 8, j;
      mp_limb_t t[8], a[8];
      fill (t, k);
      fill (a, k);
      if (rnd () % 4 != 0)
	{
	  t[k - 1] &= rnd () % 2 ? ~(mp_limb_t) 0 >> 1 : t[k - 1];
	  if (!(t[k - 1] >> (GMP_NUMB_BITS - 1)))
	    {
	      mpn_lshift (a, t, k, 1);
	      j = rnd () % (k + 1);
	      if (j < k)
		a[j] += rnd () % 2 ? 1 : -1;
	    }
	}
      nat_value (E, t, k);
      mpz_mul_2exp (E, E, 1);
      nat_value (G, a, k);
      check (cmp_lsh1 (t, a, k) == (mpz_cmp (E, G) > 0), "cmp_lsh1", it);
    }

#ifdef GTC_ACC_ASM
  /* The fused halving (gtc_rsh1add3_n), every length to 40, in place on
     any operand or not, with extreme limbs, against mpz: {r, n} = (a + b +
     c mod B^n) >> 1, or with comp (b + c - a mod B^n) >> 1, the sum read
     as an n-limb two's complement number and shifted arithmetically.  The
     library's calls are long; this reaches every entry and n = 1.  */
  for (long it = 0; it < 40000; it++)
    {
      mp_size_t n = 1 + rnd () % 40;
      unsigned comp = rnd () & 1;
      int where = (int) (rnd () % 4);	/* 0: r apart; 1..3: on a, b, c */
      mp_limb_t a[48], b[48], c[48], r[48];
      for (mp_size_t i = 0; i < n; i++)
	{
	  a[i] = limb ();
	  b[i] = limb ();
	  c[i] = limb ();
	}
      nat_value (X, a, n);
      nat_value (Y, b, n);
      nat_value (G, c, n);
      mpz_add (E, Y, G);
      if (comp) mpz_sub (E, E, X); else mpz_add (E, E, X);
      mpz_fdiv_r_2exp (E, E, GMP_NUMB_BITS * n);
      if (mpz_tstbit (E, GMP_NUMB_BITS * n - 1))
	{
	  mpz_set_ui (X, 1);
	  mpz_mul_2exp (X, X, GMP_NUMB_BITS * n);
	  mpz_sub (E, E, X);
	}
      mpz_fdiv_q_2exp (E, E, 1);
      mpz_fdiv_r_2exp (E, E, GMP_NUMB_BITS * n);
      mp_ptr d = where == 1 ? a : where == 2 ? b : where == 3 ? c : r;
      gtc_rsh1add3_n (d, a, b, c, n, comp);
      nat_value (G, d, n);
      check (mpz_cmp (E, G) == 0, "rsh1add3", it);
    }
#endif

  /* The evaluation of the Gaussian Toom-3 at every point, against mpz,
     with the bounds of the extra top limbs: at most 2 at paper I's point 1,
     at most 1 at -1 and i.  */
  {
    mp_limb_t ar[24], ai[24], buf[6 * 9];
    mpz_t a0r, a1r, a2r, a0i, a1i, a2i, W;
    mpz_inits (a0r, a1r, a2r, a0i, a1i, a2i, W, NULL);
    for (long it = 0; it < 100000; it++)
      {
	mp_size_t k = 1 + rnd () % 8, h = 1 + rnd () % k;
	geval e;
	fill (ar, 2 * k + h);
	fill (ai, 2 * k + h);
	nat_value (a0r, ar, k);
	nat_value (a1r, ar + k, k);
	nat_value (a2r, ar + 2 * k, h);
	nat_value (a0i, ai, k);
	nat_value (a1i, ai + k, k);
	nat_value (a2i, ai + 2 * k, h);
	for (int j = 0; j < 6 * 9; j++)
	  buf[j] = limb ();		/* stale values must not leak */
	e.p1r = buf;             e.p1i = e.p1r + (k + 1);
	e.m1r = e.p1i + (k + 1); e.m1i = e.m1r + (k + 1);
	e.pir = e.m1i + (k + 1); e.pii = e.pir + (k + 1);

	/* Paper I: 1, -1, i.  */
	gauss_eval3 (&e, ar, ai, k, h);
	mpz_add (W, a0r, a1r); mpz_add (W, W, a2r);
	check_sm (W, e.p1r, k + 1, 0, 2, "eval3 Re P(1)", it);
	mpz_add (W, a0i, a1i); mpz_add (W, W, a2i);
	check_sm (W, e.p1i, k + 1, 0, 2, "eval3 Im P(1)", it);
	mpz_sub (W, a0r, a1r); mpz_add (W, W, a2r);
	check_sm (W, e.m1r, k + 1, e.m1r_neg, 1, "eval3 Re P(-1)", it);
	mpz_sub (W, a0i, a1i); mpz_add (W, W, a2i);
	check_sm (W, e.m1i, k + 1, e.m1i_neg, 1, "eval3 Im P(-1)", it);
	mpz_sub (W, a0r, a2r); mpz_sub (W, W, a1i);
	check_sm (W, e.pir, k + 1, e.pir_neg, 1, "eval3 Re P(i)", it);
	mpz_sub (W, a0i, a2i); mpz_add (W, W, a1r);
	check_sm (W, e.pii, k + 1, e.pii_neg, 1, "eval3 Im P(i)", it);
      }
    mpz_clears (a0r, a1r, a2r, a0i, a1i, a2i, W, NULL);
  }

#ifdef GTC_KM_ASM
  /* The Gaussian Karatsuba over {1, i} in assembly (gtc_km_arm.S), its
     ABI entries, against mpz: P = ar br - ai bi in two's complement and Q =
     ar bi + ai br, 2n+1 limbs each, nothing written beyond them.  Extreme
     limbs, so that the sums ar + ai and br + bi carry (or not) in every
     combination.  */
  {
    mp_limb_t in[4][GTC_KM_MAX], Pb[2 * GTC_KM_MAX + 3],
      Qb[2 * GTC_KM_MAX + 3];
    mpz_t zr, zi, wr, wi, Re, Im, t;
    mpz_inits (zr, zi, wr, wi, Re, Im, t, NULL);
    for (long it = 0; it < 40000; it++)
      for (mp_size_t n = 2; n <= GTC_KM_MAX; n++)
	{
	  for (int j = 0; j < 4; j++)
	    {
	      int mode = (int) (rnd () % 4);	/* all ones, zero, extreme, random */
	      for (mp_size_t i = 0; i < n; i++)
		in[j][i] = mode == 0 ? GMP_NUMB_MAX : mode == 1 ? 0
		  : mode == 2 ? limb () : rnd ();
	    }
	  for (int i = 0; i < 2 * GTC_KM_MAX + 3; i++)
	    Pb[i] = Qb[i] = 0x5a5a5a5a5a5a5a5aULL;
	  /* The ABI entry; the _ns one, which keeps no callee-saved
	     register, is the Karatsuba levels' (test_gtc).  */
	  gtc_km_fn[n] (Pb + 1, Qb + 1, in[0], in[1], in[2], in[3], NULL);
	  nat_value (zr, in[0], n);
	  nat_value (zi, in[1], n);
	  nat_value (wr, in[2], n);
	  nat_value (wi, in[3], n);
	  mpz_mul (Re, zr, wr);
	  mpz_mul (t, zi, wi);
	  mpz_sub (Re, Re, t);
	  mpz_mul (Im, zr, wi);
	  mpz_mul (t, zi, wr);
	  mpz_add (Im, Im, t);
	  tc_value (t, Pb + 1, 2 * n + 1);
	  check (mpz_cmp (t, Re) == 0, "km real part", it);
	  nat_value (t, Qb + 1, 2 * n + 1);
	  check (mpz_cmp (t, Im) == 0, "km imaginary part", it);
	  check (Pb[0] == 0x5a5a5a5a5a5a5a5aULL
		 && Qb[0] == 0x5a5a5a5a5a5a5a5aULL
		 && Pb[2 * n + 2] == 0x5a5a5a5a5a5a5a5aULL
		 && Qb[2 * n + 2] == 0x5a5a5a5a5a5a5a5aULL,
		 "km wrote outside its result", it);
	}
    mpz_clears (zr, zi, wr, wi, Re, Im, t, NULL);
  }
#endif

  /* r2gtc_table: the algorithm per cap and size it resolves to, over
     thresholds T = 12, R = 60, R4 = 300 and R5 = 600: items with and
     without a range, both separators, later items over earlier ones,
     sizes a choice does not allow (r2gtc_55 below 17, r2gtc_44 below 13,
     r2gtc_33 at 4, GMP's Karatsuba below 4) left to the thresholds, the
     parser stopping at a malformed item, sizes from 1024 on, and choices
     above the cap left to the thresholds, which skip the algorithms
     above it.  The cap I is 5 where the field is 0.  */
  {
    static const struct { const char *t; int I; mp_size_t n; int want; }
    rt[] = {
      { NULL, 0, 100, R2GTC_ALG_33 }, { NULL, 0, 300, R2GTC_ALG_44 },
      { "100-120:4", 0, 99, R2GTC_ALG_33 },
      { "100-120:4", 0, 100, R2GTC_ALG_44 },
      { "100-120:4", 0, 120, R2GTC_ALG_44 },
      { "100-120:4", 0, 121, R2GTC_ALG_33 },
      { "100-120:4 110:3", 0, 110, R2GTC_ALG_33 },
      { "100-120:4,110:3", 0, 111, R2GTC_ALG_44 },
      { "310-320:3", 0, 315, R2GTC_ALG_33 },
      { "310-320:3", 0, 321, R2GTC_ALG_44 },
      { "5-20:4", 0, 12, R2GTC_ALG_22 },
      { "5-20:4", 0, 13, R2GTC_ALG_44 },
      { "4:3", 0, 4, R2GTC_ALG_TB },
      { "40:3", 0, 40, R2GTC_ALG_33 },
      { "100:4 x 110:4", 0, 100, R2GTC_ALG_44 },
      { "100:4 x 110:4", 0, 110, R2GTC_ALG_33 },
      { "100:4 110:6 120:4", 0, 120, R2GTC_ALG_33 },
      { "100:4 110:5 120:4", 0, 110, R2GTC_ALG_55 },
      { "100:4 110:5 120:4", 0, 120, R2GTC_ALG_44 },
      { "15-20:5", 0, 16, R2GTC_ALG_22 },
      { "15-20:5", 0, 17, R2GTC_ALG_55 },
      { "500-700:4", 0, 599, R2GTC_ALG_44 },
      { "500-700:4", 0, 701, R2GTC_ALG_55 },
      { NULL, 0, 599, R2GTC_ALG_44 }, { NULL, 0, 600, R2GTC_ALG_55 },
      { "100-120:5", 2, 110, R2GTC_ALG_22 },
      { "1000-1100:3", 0, 1023, R2GTC_ALG_33 },
      { "1000-1100:3", 0, 1024, R2GTC_ALG_55 },
      { "100-120:4", 2, 110, R2GTC_ALG_22 },
      { "100-120:4", 3, 110, R2GTC_ALG_33 },
      { "100-120:5", 4, 110, R2GTC_ALG_33 },
      { "100-120:5", 3, 110, R2GTC_ALG_33 },
      { NULL, 3, 700, R2GTC_ALG_33 }, { NULL, 4, 700, R2GTC_ALG_44 },
      { NULL, 3, 2000, R2GTC_ALG_33 }, { NULL, 2, 2000, R2GTC_ALG_22 },
      { NULL, 2, 11, R2GTC_ALG_TB }, { NULL, 2, 12, R2GTC_ALG_22 },
      { "50:t", 0, 50, R2GTC_ALG_TB }, { "50:g", 0, 50, R2GTC_ALG_GMP22 },
      { "3:g", 0, 3, R2GTC_ALG_TB }, { "50:2", 0, 50, R2GTC_ALG_22 },
      { "50:g", 2, 50, R2GTC_ALG_GMP22 }, { "50:3 51:2", 2, 51,
					     R2GTC_ALG_22 },
      { "50:3 51:2", 2, 50, R2GTC_ALG_22 },
      { "650:3", 4, 650, R2GTC_ALG_33 },
    };
    for (size_t j = 0; j < sizeof rt / sizeof rt[0]; j++)
      {
	gtc_config cfg;
	gtc_ctx *c;
	gtc_config_default (&cfg);
	cfg.r2gtc33_thresh = 60;
	cfg.r2gtc44_thresh = 300;
	cfg.r2gtc55_thresh = 600;
	cfg.r2gtc22_thresh = 12;
	cfg.gmp22_thresh = GTC_NEVER;
	int I = rt[j].I ? rt[j].I : 5;
	for (int k = 0; k < 4; k++)
	  cfg.r2gtc_table[k] = NULL;
	cfg.r2gtc_table[GTC_CAP (I)] = rt[j].t;
	c = gtc_ctx_new (&cfg);
	check (r2gtc_alg (c, I, rt[j].n) == rt[j].want, "r2gtc_table",
	       (long) j);
	gtc_ctx_free (c);
      }
  }

  /* gtc_table's list form: items over the thresholds, later ones over
     earlier ones, choices a size does not allow falling back (gtc_55 at
     15 limbs to gtc_44, gtc_44 at 12 to gtc_33), nothing from where the
     parser stops (the 'x' item and what follows); the same table under
     the caps 4, 3 and 2, where the choices above the cap are left to the
     thresholds, which skip the algorithms above it.  */
  {
    gtc_config cfg;
    gtc_ctx *c;
    gtc_config_default (&cfg);
    cfg.gtc22_thresh = 12;
    cfg.gtc33_thresh = 70;
    cfg.gtc44_thresh = 100;
    cfg.gtc55_thresh = 300;
    for (int k = 0; k < 4; k++)
      cfg.gtc_table[k]
	= "150-160:5 155:4,90:4 15:5 12:4 200-210:3 220:x 230:5";
    c = gtc_ctx_new (&cfg);
    static const struct { int I; mp_size_t n; int want; } gt[] = {
      { 5, 149, GTC_ALG_44 }, { 5, 150, GTC_ALG_55 }, { 5, 155, GTC_ALG_44 },
      { 5, 160, GTC_ALG_55 }, { 5, 161, GTC_ALG_44 }, { 5, 90, GTC_ALG_44 },
      { 5, 89, GTC_ALG_33 }, { 5, 15, GTC_ALG_44 }, { 5, 12, GTC_ALG_33 },
      { 5, 205, GTC_ALG_33 }, { 5, 211, GTC_ALG_44 }, { 5, 230, GTC_ALG_44 },
      { 5, 300, GTC_ALG_55 },
      { 4, 150, GTC_ALG_44 }, { 4, 155, GTC_ALG_44 }, { 4, 300, GTC_ALG_44 },
      { 4, 205, GTC_ALG_33 }, { 3, 150, GTC_ALG_33 }, { 3, 90, GTC_ALG_33 },
      { 3, 300, GTC_ALG_33 }, { 3, 89, GTC_ALG_33 }, { 3, 69, GTC_ALG_22 },
      { 3, 205, GTC_ALG_33 }, { 2, 205, GTC_ALG_22 }, { 2, 300, GTC_ALG_22 },
      { 2, 150, GTC_ALG_22 }, { 4, 2000, GTC_ALG_44 },
      { 3, 2000, GTC_ALG_33 } };
    for (unsigned j = 0; j < sizeof gt / sizeof gt[0]; j++)
      {
	int a = gt[j].n < GTC_TABLE_N ? c->alg[GTC_CAP (gt[j].I)][gt[j].n]
	  : gtc_alg_thresh (&c->cfg, gt[j].I, gt[j].n);
	check (a == gt[j].want, "gtc_table list", (long) j);
      }
    gtc_ctx_free (c);
  }

  /* The caps: no algorithm above the cap anywhere in the resolved tables,
     with the defaults and with random tables (the leaf functions are
     the base cases and gtc_22), and a leaf function wherever one is set
     (the textbook product and the {1, i} Karatsuba up to their sizes,
     gtc_22 levels in assembly over leaf halves of the cap 2).  */
  for (int v = 0; v < 200; v++)
    {
      static char tab[8][300];
      gtc_config cfg;
      gtc_ctx *c;
      gtc_config_default (&cfg);
      if (v)
	for (int k = 0; k < 4; k++)
	  {
	    int len = 1 + (int) (rnd () % 299);
	    for (int j = 0; j < len; j++)
	      tab[k][j] = "tk2345-"[rnd () % 7];
	    tab[k][len] = 0;
	    snprintf (tab[4 + k], sizeof tab[0], "%d-%d:%c %d-%d:%c",
		      (int) (rnd () % 900), (int) (rnd () % 1100),
		      "tg2345"[rnd () % 6], (int) (rnd () % 900),
		      (int) (rnd () % 1100), "tg2345"[rnd () % 6]);
	    cfg.gtc_table[k] = rnd () & 1 ? tab[k] : NULL;
	    cfg.r2gtc_table[k] = rnd () & 1 ? tab[4 + k] : NULL;
	    cfg.gtc33_thresh = 5 + rnd () % 100;
	    cfg.gtc44_thresh = 13 + rnd () % 200;
	    cfg.gtc55_thresh = 17 + rnd () % 300;
	    cfg.r2gtc33_thresh = 5 + rnd () % 100;
	    cfg.r2gtc44_thresh = 13 + rnd () % 200;
	    cfg.r2gtc55_thresh = 17 + rnd () % 300;
	  }
      c = gtc_ctx_new (&cfg);
      for (int I = 2; I <= 5; I++)
	for (mp_size_t n = 2; n < GTC_TABLE_N; n++)
	  {
	    int a = c->alg[GTC_CAP (I)][n], r = c->ralg[GTC_CAP (I)][n];
	    check (a <= I && r <= I, "an algorithm above the cap",
		   (long) (v * 10000 + I * 1000 + n));
	    check ((a == GTC_ALG_LEAF) == (n < GTC_LEAF_N
					   && c->leaf[GTC_CAP (I)][n] != NULL),
		   "leaf function", (long) (v * 10000 + I * 1000 + n));
	  }
      gtc_ctx_free (c);
    }

  /* Configuration clamping: thresholds below 5 become 5, and products are
     still right.  */
  {
    gtc_config cfg;
    gtc_ctx *c;
    mp_limb_t a[40], b[40], r[80], ref[80];
    gtc_config_default (&cfg);
    cfg.r2gtc33_thresh = 1;
    cfg.gtc33_thresh = 0;
    cfg.gtc22_thresh = -1;
    cfg.r2gtc22_thresh = -3;
    cfg.gtc_km_thresh = -4;
    cfg.gmp22_thresh = -5;
    cfg.gtc44_thresh = 3;
    cfg.gtc55_thresh = 4;
    cfg.r2gtc44_thresh = 2;
    cfg.r2gtc55_thresh = 6;
    c = gtc_ctx_new (&cfg);
    check (c->cfg.r2gtc33_thresh == 5 && c->cfg.gtc33_thresh == 5
	   && c->cfg.gtc22_thresh == 2
	   && c->cfg.r2gtc22_thresh == 2 && c->cfg.gtc_km_thresh == 2
	   && c->cfg.gmp22_thresh == GTC_GMP22_MIN
	   && c->cfg.gtc44_thresh == GTC_44_MIN
	   && c->cfg.gtc55_thresh == GTC_55_MIN
	   && c->cfg.r2gtc44_thresh == GTC_44_MIN
	   && c->cfg.r2gtc55_thresh == GTC_55_MIN,
	   "config clamping", 0);
    for (mp_size_t n = 1; n <= 40; n++)
      {
	fill (a, n);
	fill (b, n);
	mpn_mul_n (ref, a, b, n);
	r2gtc_mul_n_ctx (c, r, a, b, n);
	check (mpn_cmp (r, ref, 2 * n) == 0, "clamped config product", n);
      }
    gtc_ctx_free (c);
  }

  /* The default-context wrappers.  */
  {
    mp_limb_t a[300], b[200], r[500], ref[500];
    mpz_t za, zb, zr;
    fill (a, 300);
    fill (b, 200);
    mpn_mul (ref, a, 300, b, 200);
    r2gtc_mul (r, a, 300, b, 200);
    check (mpn_cmp (r, ref, 500) == 0, "r2gtc_mul", 0);
    mpn_mul_n (ref, a, b, 200);
    r2gtc_mul_n (r, a, b, 200);
    check (mpn_cmp (r, ref, 400) == 0, "r2gtc_mul_n", 0);
    mpz_inits (za, zb, zr, NULL);
    nat_value (za, a, 300);
    mpz_neg (za, za);
    nat_value (zb, b, 200);
    mpz_mul (E, za, zb);
    r2gtc_mpz_mul (zr, za, zb);
    check (mpz_cmp (zr, E) == 0, "r2gtc_mpz_mul", 0);
    mpz_clears (za, zb, zr, NULL);
  }

  mpz_clears (X, Y, E, G, P, NULL);
  printf ("%s: %ld helper checks, %ld failures\n",
	  failures ? "FAILED" : "PASSED", checks, failures);
  return failures != 0;
}
