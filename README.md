# libgtc

Long integer multiplication through Gaussian integers, on top of
[GMP](https://gmplib.org). Version 0.1.0.

libgtc multiplies rational integers by reducing a product to smaller
rational products and *Gaussian* products (of numbers a + bi), and
multiplies the Gaussian ones by Gaussian Toom-Cook, recursively. One level
of the Toom-Cook-3 style reduction, for instance, makes three rational
products and one Gaussian product of a third of the size, where GMP's
Toom-3 makes five rational ones; a Gaussian product of that size costs
less than two rational ones, so the reduction pays. The algorithms are
those of two papers, referred to here as paper I and paper II
(references: tbd, see [Citation](#citation)).

libgtc works on GMP's `mpn` and `mpz` types and calls GMP's `mpn` layer
for most of its linear-time work. On x86-64 (with BMI2 and ADX) and on
aarch64 it adds its own assembly: the Gaussian base case, the Gaussian
Karatsuba levels up to 64 limbs, and the linear passes of the levels
above. Elsewhere it is portable C.

## Results

Time of libgtc's product over GMP's, median over the sizes from 100 limbs
(each size the minimum of 7 trials, `bench_gtc --compare ... --trials 7`). Each side runs
one level of its algorithm at the top and its own algorithms below it:
`r2gtc_ii` its products within the cap i (see below), GMP's
`mpn_toomii_mul` its products by GMP's thresholds. GMP 6.3.0 with the
kernels it selects itself.

| time ratio | Xeon W-1270 (Comet Lake) | EPYC 7763 (Zen 3) | Xeon Gold 6238T (Cascade Lake) | Kunpeng 920 |
|---|---|---|---|---|
| `r2gtc_22` ÷ `mpn_toom22_mul` | 0.77 | 1.00 | 0.93 | 0.93 |
| `r2gtc_33` ÷ `mpn_toom33_mul` | 0.72 | 0.92 | 0.87 | 0.78 |
| `r2gtc_44` ÷ `mpn_toom44_mul` | 0.68 | 0.89 | 0.83 | 0.82 |
| `r2gtc_55` ÷ `mpn_toom44_mul` | 0.63 | 0.82 | 0.77 | 0.74 |
| `r2gtc_55` ÷ `mpn_toom6h_mul` | 0.66 | 0.87 | 0.81 | 0.81 |
| `r2gtc` ÷ `mpn_mul_n` | 0.69 | 0.91 | 0.84 | 0.85 |

GMP has no balanced Toom-Cook-5; its neighbours are Toom-4 and Toom-6.5
(`mpn_toom6h_mul`). `mpn_mul_n` is GMP's full multiplication, FFT
included. `r2gtc` takes less time than `mpn_mul_n` up to about 3,000
limbs on the Zen 3, 3,800 on the Kunpeng 920, 5,100 on the Cascade Lake
and beyond 10,000 on the Comet Lake. Above that GMP's Toom-8.5 and FFT
win; libgtc has no FFT.

## The algorithms

Gaussian products, (ar + i ai)(br + i bi) with n-limb components:

| name | algorithm | one level makes |
|---|---|---|
| `gtc_22` | Gaussian Karatsuba | 3 Gaussian products of n/2 limbs |
| `gtc_33` | Gaussian Toom-Cook-3 (paper I): points (0,1), (−1,1), (1,1), (i,1), (1,0) | 5 of n/3 |
| `gtc_44` | Gaussian Toom-Cook-4 (papers I and II): points 0, ±1, ±i, 2, ∞ | 7 of n/4 |
| `gtc_55` | Gaussian Toom-Cook-5 (paper II): points 0, ±1, ±i, 1+i, 1−i, −1+i, ∞ | 9 of n/5 |
| base case | the textbook product (four real products), in assembly up to 20 limbs; on aarch64 also the Karatsuba over {1, i} (three) | |

Rational products, a · b with n-limb factors:

| name | algorithm | one level makes |
|---|---|---|
| `r2gtc_22` | Rational-to-Gaussian Karatsuba reduction (paper I) | 1 rational and 1 Gaussian product of n/2 limbs |
| `r2gtc_33` | Rational-to-Gaussian Toom-Cook-3 reduction (paper II): points (0,1), (−1,1), (1,0), (i,1) | 3 rational and 1 Gaussian of n/3 |
| `r2gtc_44` | Rational-to-Gaussian Toom-Cook-4 reduction (paper II): points (0,1), (−1,1), (1,0), (i,1), (−1−i,1) | 3 rational and 2 Gaussian of n/4 |
| `r2gtc_55` | Rational-to-Gaussian Toom-Cook-5 reduction (paper II): points (0,1), (−1,1), (1,0), (i,1), (1+i,1), (−1+i,1) | 3 rational and 3 Gaussian of n/5 |
| base cases | GMP's Karatsuba (`mpn_toom22_mul`) and textbook product (`mpn_mul_basecase`) | |

The papers' point (1,1) is mirrored to (−1,1): the value there needs an
extra limb far less often. `gtc` and `r2gtc` choose among all of these by
size. `gtc_ii` and `r2gtc_ii` recurse only into algorithms of degree j ≤
i, as GMP's `mpn_toomii_mul` does: the cap i. See
[doc/DESIGN.md](doc/DESIGN.md) for how it all fits together.

## Building

Requirements: Linux (or another ELF system), GCC, GNU make, GMP
6.x with its development files (`libgmp-dev`, `gmp-devel`), and curl or
wget for the first build. Python 3 only to regenerate the assembly.

```sh
make                 # build/libgtc.a, build/libgtc.so, the benchmark and the tuner
make check           # the tests (a few minutes)
make install         # PREFIX=/usr/local; DESTDIR for staging
```

libgtc calls a few GMP functions that libgmp exports but only GMP's
private `gmp-impl.h` declares (`mpn_mul_basecase`, `mpn_toom22_mul`,
`mpn_bdiv_dbm1c`, ...). Some of them (`mpn_addlsh1_n`, ...) exist only
in a libgmp built with its assembly, as distributions build it on x86-64
and aarch64; one configured with `--disable-assembly` cannot be used.
The first build runs `tools/fetch_gmp.sh`, which
downloads the GMP 6.3.0 source once, checks its SHA-256 and takes those
declarations from it into `build/gmp_decls.h`. Set
`GMP_TARBALL=/path/to/gmp-6.3.0.tar.xz` to use a local copy.

The build is for the machine it runs on (`-march=native`, `-mcpu=native`
on aarch64), as GMP's own build is: the assembly, some code paths and the
default thresholds are chosen at compile time for the microarchitecture.
There are tuned defaults for AMD Zen, Intel cores with and without
AVX-512, and the Kunpeng 920; other machines get those of the nearest
kind. Other options:

```sh
make NO_ASM=1                # portable C, no assembly
make B=build-x               # another build directory
make GMP_PREFIX=$HOME/gmp    # GMP installed elsewhere
make CFLAGS='-O2 -march=znver3'
```

## Using it

```c
#include <gtc.h>

/* {rp, 2n} = {ap, n} * {bp, n} */
r2gtc_mul_n (rp, ap, bp, n);
/* {rp, an+bn} = {ap, an} * {bp, bn}, an >= bn >= 1 */
r2gtc_mul (rp, ap, an, bp, bn);
/* r = a * b */
r2gtc_mpz_mul (r, a, b);
```

Link with `-lgtc -lgmp` (or `pkg-config --cflags --libs libgtc`). These use a
per-thread context with the default configuration. A context
(`gtc_ctx_new`) holds a configuration, the thresholds and per-size tables
of `gtc_config`, and a scratch arena that grows to the peak need, so that
repeated products do not allocate; the `_ctx` forms take one. A context
must not be used by two threads at once. `r2gtc_22_ctx` .. `r2gtc_55_ctx`
run one level of `r2gtc_ii` at the top; `gtc_mul_n_ctx` and `gtc_22_ctx`
.. `gtc_55_ctx` are the Gaussian products, with components in
sign-magnitude form. See [include/gtc.h](include/gtc.h).

## Tuning, benchmarks, tests

* `build/bench_gtc --compare 22|33|44|55|r2gtc` runs the comparisons
  above (`--help` for the options; `--csv FILE` writes the results).
  Pin it to a core of an otherwise idle machine (`--pin CPU`).
* `tools/tune.sh [CPU]` finds the per-size tables for the machine (a few
  hours); `build/tune_gauss` the thresholds (`--help`). Both print what
  `gtc_config` takes and `src/config.h` holds as defaults.
* `make check` runs `test_gtc` (every interface against GMP over many
  sizes, configurations and adversarial inputs) and `test_helpers`
  (white-box tests of the internal helpers and the assembly kernels), in
  several builds. `make check-asan` and `make check-valgrind` run
  `test_gtc` under the sanitizers and under valgrind.

## Citation

tbd

## License

BSD 3-Clause Clear; see [LICENSE](LICENSE). Copyright (c) 2026 Huawei
Technologies Co., Ltd.
