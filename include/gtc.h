/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   libgtc: Gaussian Toom-Cook, and rational long integer multiplication
   through Gaussian integers.

   The algorithms follow two papers, referred to as paper I and paper II
   (references: tbd).

   Gaussian multiplication, (ar + i ai)(br + i bi):
     gtc_22  Gaussian Karatsuba (GMP's toom22 over Z[i]);
     gtc_33  Gaussian Toom-Cook-3, paper I, Sections 4.2 and 4.3:
             points (0,1), (-1,1), (1,1), (i,1), (1,0);
     gtc_44  Gaussian Toom-Cook-4, paper I (4.9) and paper II, Section
             5.1: points 0, 1, -1, i, -i, 2, infinity;
     gtc_55  Gaussian Toom-Cook-5, paper II, Sections 5.2 and 5.3:
             points 0, 1, -1, i, -i, 1+i, 1-i, -1+i, infinity;
     the Gaussian textbook product (four real products per limb pair),
     and on aarch64 below gtc_km_thresh the Gaussian Karatsuba over
     {1, i} (three real products), the base cases.
   gtc_ii's own products are gtc_jj (j <= i) or a base case.  gtc picks
   any of them by size.

   Rational multiplication, a * b:
     r2gtc_22  Rational-to-Gaussian Karatsuba reduction, paper I,
               Algorithm 1.1: one rational and one Gaussian product of
               n/2 limbs per level;
     r2gtc_33  Rational-to-Gaussian Toom-Cook-3 reduction, paper II,
               Section 4.4: points (0,1), (-1,1), (1,0) (rational) and
               (i,1) (Gaussian), three rational and one Gaussian product
               of n/3 limbs per level;
     r2gtc_44  Rational-to-Gaussian Toom-Cook-4 reduction, paper II,
               Section 4.5: points (0,1), (-1,1), (1,0) and (i,1),
               (-1-i,1), three rational and two Gaussian products of n/4
               limbs per level;
     r2gtc_55  Rational-to-Gaussian Toom-Cook-5 reduction, paper II,
               Section 4.6: points (0,1), (-1,1), (1,0) and (i,1),
               (1+i,1), (-1+i,1), three rational and three Gaussian
               products of n/5 limbs per level;
     GMP's rational Karatsuba (mpn_toom22_mul) and GMP's rational textbook
     product (mpn_mul_basecase).
   (The papers' (1,1) is mirrored to (-1,1) by x -> -x: the value there
   needs an extra top limb far less often.)
   r2gtc_ii's rational products are r2gtc_jj (j <= i), GMP's Karatsuba or
   the textbook product; its Gaussian products are gtc_jj (j <= i) or a
   Gaussian base case.  r2gtc picks any of r2gtc_ii, GMP's Karatsuba and
   the textbook product by size.

   So a product of gtc_ii or r2gtc_ii never recurses into an algorithm of
   a higher degree, as GMP's mpn_toomii_mul never does: within that cap
   the algorithm of each product is chosen by size, from a table for the
   cap (gtc_table, r2gtc_table) or from the thresholds.

   The linear-time work is done with GMP's mpn layer and, on x86-64 with
   BMI2 and ADX and on aarch64, gtc's own assembly: the Gaussian textbook
   product up to 20 limbs, the Gaussian Karatsuba levels over it up to 64
   limbs, and passes of the levels above.  Elsewhere the textbook product
   is C, up to 20 limbs.  */

#ifndef GTC_H
#define GTC_H

#include <gmp.h>

#define GTC_VERSION_MAJOR 0
#define GTC_VERSION_MINOR 1
#define GTC_VERSION_PATCH 0
#define GTC_VERSION_STRING "0.1.0"

#ifdef __cplusplus
extern "C" {
#endif

/* A threshold that no size reaches: the algorithm is not used.  */
#define GTC_NEVER ((mp_size_t) 1 << 40)

/* The index of the cap i = 2, 3, 4, 5 in gtc_table and r2gtc_table.  */
#define GTC_CAP(i) ((i) - 2)

typedef struct
{
  /* Gaussian products with n-limb components, by the thresholds: gtc_55
     when n >= gtc55_thresh (minimum 17), else gtc_44 when n >=
     gtc44_thresh (minimum 13), else gtc_33 when n >= gtc33_thresh
     (minimum 5), else gtc_22 when n >= gtc22_thresh (minimum 2), else the
     base case; an algorithm above the cap is skipped (under gtc_33, n >=
     gtc55_thresh goes on to the gtc_33 test).  GTC_NEVER leaves one out.
     The base case is the textbook product up to 20 limbs and the
     Gaussian Karatsuba over {1, i} from gtc_km_thresh (minimum 2) where
     it is in assembly (aarch64: 2 to 10 limbs); gtc22_thresh is lowered
     to 21 if it lies above.  Defaults per microarchitecture, see
     gtc_config_default.  */
  mp_size_t gtc22_thresh;
  mp_size_t gtc33_thresh;
  mp_size_t gtc44_thresh;
  mp_size_t gtc55_thresh;
  mp_size_t gtc_km_thresh;
  /* Optional, per cap: gtc_table[GTC_CAP (i)] gives the algorithm of the
     products of gtc_ii by size, over the thresholds (gtc_table[GTC_CAP
     (5)] also gtc's own choice).  For n < strlen (table) (and n < 1024),
     table[n] is 't' (the base case), '2' or 'k' (gtc_22), '3', '4' or
     '5' (gtc_33, gtc_44, gtc_55); other characters, the sizes beyond the
     string and algorithms above the cap follow the thresholds.  A string
     with a ':' is a list instead, of "LO-HI:C" or "N:C" items with C one
     of those, separated by spaces or commas, later items over earlier
     ones (the parser stops at what it does not understand).  Choices a
     size does not allow (gtc_55 needs n >= 17, gtc_44 n >= 13, gtc_33
     n = 3 or n >= 5, the textbook product at most 20 limbs) fall back to
     gtc_44, gtc_33 or gtc_22.  NULL: the thresholds.  tune_gauss --table
     --cap i finds the fastest choice per size.  */
  const char *gtc_table[4];
  /* Rational n x n products, by the thresholds: r2gtc_55 when n >=
     r2gtc55_thresh (minimum 17), else r2gtc_44 when n >= r2gtc44_thresh
     (minimum 13), else r2gtc_33 when n >= r2gtc33_thresh (minimum 5), else
     r2gtc_22 when n >= r2gtc22_thresh (minimum 2), else GMP's Karatsuba
     when n >= gmp22_thresh (minimum 4), else GMP's textbook product,
     mpn_mul_basecase; an algorithm above the cap is skipped.  GTC_NEVER
     leaves one out.  Never GMP's Toom.  */
  mp_size_t r2gtc22_thresh;
  mp_size_t r2gtc33_thresh;
  mp_size_t r2gtc44_thresh;
  mp_size_t r2gtc55_thresh;
  mp_size_t gmp22_thresh;
  /* Optional, per cap, as gtc_table: r2gtc_table[GTC_CAP (i)] gives the
     algorithm of the rational products of r2gtc_ii below 1024 limbs
     (r2gtc_table[GTC_CAP (5)] also r2gtc's own choice), over the
     thresholds.  A list of "LO-HI:C" or "N:C" items separated by spaces
     or commas, C 't' (the textbook product), 'g' (GMP's Karatsuba, from
     4 limbs), '2' (r2gtc_22), '3' (r2gtc_33, n = 3 or n >= 5), '4'
     (r2gtc_44, from 13 limbs) or '5' (r2gtc_55, from 17 limbs), later
     items over earlier ones; other sizes follow the thresholds, and so
     do items above the cap or that a size does not allow, and anything
     the parser does not understand from there on.  tune_gauss
     --rat-table --cap i finds the fastest per size.  */
  const char *r2gtc_table[4];
} gtc_config;

/* A context holds a configuration and a scratch arena that grows to the
   peak requirement, so that repeated calls do not allocate.  A context
   must not be used by two threads at the same time.  */
typedef struct gtc_ctx gtc_ctx;

void gtc_config_default (gtc_config *cfg);
gtc_ctx *gtc_ctx_new (const gtc_config *cfg);	/* cfg == NULL: defaults */
void gtc_ctx_free (gtc_ctx *ctx);
const gtc_config *gtc_ctx_config (const gtc_ctx *ctx);

/* Rational multiplication.  */

/* {rp, 2n} = {ap, n} * {bp, n}, n >= 1, by r2gtc.  rp must not overlap
   the inputs.  */
void r2gtc_mul_n_ctx (gtc_ctx *ctx, mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
		      mp_size_t n);
/* The same with one level of r2gtc_ii at the top (r2gtc_22 for n >= 2,
   r2gtc_33 for n = 3 and n >= 5, r2gtc_44 for n >= 13, r2gtc_55 for n >=
   17; smaller n as its products), and its products below.  */
void r2gtc_22_ctx (gtc_ctx *ctx, mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
		   mp_size_t n);
void r2gtc_33_ctx (gtc_ctx *ctx, mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
		   mp_size_t n);
void r2gtc_44_ctx (gtc_ctx *ctx, mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
		   mp_size_t n);
void r2gtc_55_ctx (gtc_ctx *ctx, mp_ptr rp, mp_srcptr ap, mp_srcptr bp,
		   mp_size_t n);
/* {rp, an+bn} = {ap, an} * {bp, bn}, an >= bn >= 1, no overlap.  Unbalanced
   operands are cut into bn-limb chunks; where a bn x bn product would be
   the textbook product, the whole is mpn_mul_basecase.  */
void r2gtc_mul_ctx (gtc_ctx *ctx, mp_ptr rp, mp_srcptr ap, mp_size_t an,
		    mp_srcptr bp, mp_size_t bn);
/* r = a * b; r may alias a or b.  */
void r2gtc_mpz_mul_ctx (gtc_ctx *ctx, mpz_ptr r, mpz_srcptr a, mpz_srcptr b);

/* The same with the default configuration and a per-thread context (which
   is never freed; long-running threaded programs should use contexts).  */
void r2gtc_mul_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n);
void r2gtc_mul (mp_ptr rp, mp_srcptr ap, mp_size_t an, mp_srcptr bp,
		mp_size_t bn);
void r2gtc_mpz_mul (mpz_ptr r, mpz_srcptr a, mpz_srcptr b);

/* Gaussian multiplication.  */

/* The Gaussian product (xr + i xi)(yr + i yi) by gtc.  Each input
   component is given in sign-magnitude form: n >= 1 limbs (leading zeros
   allowed) and a sign flag (nonzero = negative).  The result components
   are written in the same form to zr and zi, 2n+1 limbs each.  No overlap
   between outputs and inputs.  */
void gtc_mul_n_ctx (gtc_ctx *ctx,
		    mp_ptr zr, int *zr_neg, mp_ptr zi, int *zi_neg,
		    mp_srcptr xr, int xr_neg, mp_srcptr xi, int xi_neg,
		    mp_srcptr yr, int yr_neg, mp_srcptr yi, int yi_neg,
		    mp_size_t n);
/* The same with one level of gtc_ii at the top (gtc_22 for n >= 2, gtc_33
   for n = 3 and n >= 5, gtc_44 for n >= 13, gtc_55 for n >= 17; smaller
   n as its products), and its products below.  */
#define GTC_GAUSS_ARGS							\
  gtc_ctx *ctx, mp_ptr zr, int *zr_neg, mp_ptr zi, int *zi_neg,		\
  mp_srcptr xr, int xr_neg, mp_srcptr xi, int xi_neg,			\
  mp_srcptr yr, int yr_neg, mp_srcptr yi, int yi_neg, mp_size_t n
void gtc_22_ctx (GTC_GAUSS_ARGS);
void gtc_33_ctx (GTC_GAUSS_ARGS);
void gtc_44_ctx (GTC_GAUSS_ARGS);
void gtc_55_ctx (GTC_GAUSS_ARGS);
#undef GTC_GAUSS_ARGS

#ifdef __cplusplus
}
#endif

#endif
