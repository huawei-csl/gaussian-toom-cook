/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
   SPDX-License-Identifier: BSD-3-Clause-Clear

   The platform: which assembly kernels exist, and the default thresholds
   and tables, tuned per microarchitecture as GMP tunes gmp-mparam.h.  */

/* GTC_TEXTBOOK_ASM: the Gaussian textbook kernels (2 to 20 limbs).
   GTC_ASM_X86, GTC_ASM_ARM: the linear kernels of gtc_linear_*.S.
   GTC_KARA_ASM: the Gaussian Karatsuba levels of gtc_kara_*.S (4 to 64).
   GTC_KM_ASM: the Gaussian Karatsuba over {1, i}, gtc_km_arm.S.
   GTC_ACC_ASM: the accumulation kernels (gtc_add3_n, gtc_aors2_n, ...).  */
#if defined (__x86_64__) && defined (__ELF__) && defined (__BMI2__) \
  && defined (__ADX__) && !defined (GTC_NO_ASM)
#define GTC_TEXTBOOK_ASM 1
#define GTC_ASM_X86 1
#elif defined (__aarch64__) && defined (__ELF__) && !defined (GTC_NO_ASM)
#define GTC_TEXTBOOK_ASM 1
#define GTC_ASM_ARM 1
#define GTC_KM_ASM 1
#define GTC_KM_MAX 10
#endif
#define GTC_TEXTBOOK_LIMIT 20
#if defined (GTC_ASM_X86) \
  || (defined (__aarch64__) && defined (GTC_TEXTBOOK_ASM))
#define GTC_KARA_ASM 1
#define GTC_ACC_ASM 1
#endif
/* gtc_aors2x_n, aarch64's: two additions or subtractions in one pass.  */
#if defined (GTC_ACC_ASM) && defined (__aarch64__)
#define GTC_PAIR_ASM 1
#endif

#if defined (__znver1__) || defined (__znver2__) || defined (__znver3__) \
  || defined (__znver4__) || defined (__znver5__)
#define GTC_ZEN 1
#endif

/* The thresholds (gtc_config): K = gtc22, G = gtc33, G4 = gtc44, G5 =
   gtc55, KM = gtc_km; T = r2gtc22, R = r2gtc33, R4 = r2gtc44, R5 =
   r2gtc55, X = gmp22.  Measured on an EPYC 7763 (Zen 3), a Kunpeng 920
   (aarch64), a Xeon Gold 6238T (Cascade Lake, AVX-512) and a Xeon W-1270
   (Comet Lake, the other x86-64): tools/tune.sh, then whole products.  */
#if !defined (GTC_TEXTBOOK_ASM)
#define GTC_DEFAULT_GTC22 18
#define GTC_DEFAULT_GTC33 57
#define GTC_DEFAULT_GTC55 250
#define GTC_DEFAULT_R2GTC22 20
#define GTC_DEFAULT_R2GTC33 45
#elif defined (GTC_ZEN)
#define GTC_DEFAULT_GTC22 12
#define GTC_DEFAULT_GTC33 68
#define GTC_DEFAULT_GTC55 150
#define GTC_DEFAULT_R2GTC22 100
#define GTC_DEFAULT_R2GTC33 33
#define GTC_DEFAULT_R2GTC44 240
#define GTC_DEFAULT_GMP22 21
#elif defined (__aarch64__)
#define GTC_DEFAULT_GTC22 11
#define GTC_DEFAULT_GTC33 68
#define GTC_DEFAULT_GTC55 150
#define GTC_DEFAULT_GTC_KM 6
#define GTC_DEFAULT_R2GTC22 26
#define GTC_DEFAULT_R2GTC33 35
#define GTC_DEFAULT_R2GTC44 240
#elif defined (__AVX512F__)
#define GTC_DEFAULT_GTC22 11
#define GTC_DEFAULT_GTC33 70
#define GTC_DEFAULT_GTC55 250
#define GTC_DEFAULT_R2GTC22 18
#define GTC_DEFAULT_R2GTC33 35
#else
#define GTC_DEFAULT_GTC22 11
#define GTC_DEFAULT_GTC33 74
#define GTC_DEFAULT_GTC55 250
#define GTC_DEFAULT_R2GTC22 12
#define GTC_DEFAULT_R2GTC33 60
#endif
#define GTC_DEFAULT_GTC44 100
#define GTC_DEFAULT_R2GTC55 150
#ifndef GTC_DEFAULT_R2GTC44
#define GTC_DEFAULT_R2GTC44 300
#endif
#ifndef GTC_DEFAULT_GTC_KM
#define GTC_DEFAULT_GTC_KM GTC_NEVER
#endif
#ifndef GTC_DEFAULT_GMP22
#define GTC_DEFAULT_GMP22 GTC_NEVER
#endif

/* The tables (gtc_config's gtc_table and r2gtc_table), per cap: the sizes
   below 1024 limbs where another algorithm than the thresholds' wins
   (tools/tune.sh).  Kept per cap and machine only where they beat the
   thresholds on whole products.  */
#if !defined (GTC_TEXTBOOK_ASM)
#elif defined (GTC_ZEN)
#define GTC_DEFAULT_GTC_TABLE_33 \
  "59-60:3 65-67:3"
#define GTC_DEFAULT_GTC_TABLE_44 \
  "66:3 67-68:4 76:4 88:4 91-92:4 99:4 165-168:3"
#define GTC_DEFAULT_GTC_TABLE_55 \
  "59-60:3 66:3 67-68:4 76:4 85:5 88:4 91-92:4 96:4 99:4 " \
  "113-115:5 117-120:5 125:5 129-140:5 143:5 145-149:5 329-332:4"
#define GTC_DEFAULT_R2GTC_TABLE_22 \
  "18:2 20:2 27-28:2 31-33:2 36:2 39-40:2 54-56:2 63-74:2 78-80:2 " \
  "95-96:2 100-104:g 119:g 152:g 163-184:g 196-197:g 201-205:g " \
  "207-208:g 303:g 323-354:g 356-377:g 379-380:g 383:g 385-391:g " \
  "393-409:g 412-416:g 418-419:g 586:g 591:g 608:g 647:g " \
  "649-654:g 656-712:g 714-734:g 736-741:g 743:g 745-747:g " \
  "749-754:g 756:g 772-778:g 780:g 782-799:g 801-848:g 921:g " \
  "925-928:g 935:g 941-942:g 944-945:g 947:g 949:g 955:g 957:g " \
  "959-960:g 970-971:g 973:g 976:g"
#define GTC_DEFAULT_R2GTC_TABLE_44 \
  "28:2 32:2 43:g 49:g 61-62:4 64:4 85:4 88:4 91-96:4 100-104:4 " \
  "109-112:4 115:4 121-160:4 163-165:4 167-179:4 181-232:4 " \
  "235-239:4 247-248:3 250-252:3"
#define GTC_DEFAULT_R2GTC_TABLE_55 \
  "61:4 63-64:4 85:4 88:4 91-95:4 97:4 100:4 102-104:4 109-112:4 " \
  "113-125:5 126-127:4 128-140:5 141-144:4 145-149:5 287-288:4 " \
  "321:4 326:4 331-332:4 335:4"
#elif defined (__aarch64__)
#define GTC_DEFAULT_GTC_TABLE_33 \
  "42:3 44-45:3 50-54:3 57-60:3 73:2 76:2 79-80:2 127-128:2"
#define GTC_DEFAULT_GTC_TABLE_44 \
  "42:3 44-45:3 50-51:3 53-54:3 57-60:3 65-80:4 88-92:4 94:4 96:4 " \
  "99:4 161-162:3 165-171:3 175:3 177-180:3"
#define GTC_DEFAULT_GTC_TABLE_55 \
  "41-42:3 44-45:3 47:3 50-54:3 57-60:3 61-62:4 64-80:4 85-86:5 " \
  "88-90:5 92-95:5 99:4 101:3 113-115:5 129-135:5 145:5 148-149:5 " \
  "156:4 206-216:4 219-220:4 222-224:4 236:4 251-252:4 283:4 " \
  "296-297:4 300:4 426-428:4 486-489:4 506-508:4"
#define GTC_DEFAULT_R2GTC_TABLE_44 \
  "17:2 21:3 23:g 24:2 27:3 29-30:3 33-34:3 55-56:4 61-64:4 67:4 " \
  "74-75:4 79:4 109-112:4 121:4 127-129:4 131:4 133-134:4 " \
  "136-137:4 139-140:4 145-156:4 163-165:4 181-192:4 197-212:4 " \
  "214-239:4 242-244:3 247:3 250-252:3 258:3 269-271:3 274:3 " \
  "285-288:3 305-306:3 317-324:3"
#define GTC_DEFAULT_R2GTC_TABLE_55 \
  "62-64:4 68:4 69:5 73-80:5 85-95:5 97-100:5 109-110:5 111-112:4 " \
  "113-116:5 118-120:5 121:4 122:5 123:4 124-140:5 145-149:5 " \
  "211-216:4 287-288:3 321-324:3 331-332:4"
#elif defined (__AVX512F__)
#define GTC_DEFAULT_GTC_TABLE_33 \
  "66-69:3 112:2 128:2"
#define GTC_DEFAULT_GTC_TABLE_44 \
  "66:3 68-69:3 92:4 99:4 161-168:3"
#define GTC_DEFAULT_GTC_TABLE_55 \
  "66-69:3 92:4 99:4 114:5 129-130:5 132-135:5 137-140:5 " \
  "145-185:5 189-190:5 193-200:5 204-205:5 209-210:5 212-214:5 " \
  "225-249:5 326-332:4 334-335:4 377-378:4 380:4 432:4 441:4 " \
  "443:4"
#define GTC_DEFAULT_R2GTC_TABLE_33 \
  "16:2 37:2 40:2 55-56:2 64:2"
#define GTC_DEFAULT_R2GTC_TABLE_44 \
  "16:2 37:2 40:2 55-56:2 64:2 103:4 109-113:4 115-116:4 118:4 " \
  "121-131:4 133-136:4 145:4 147:4 149-160:4 173-176:4 181-215:4 " \
  "217-233:4 235-240:4 253-257:4 259-264:4 271-299:4"
#define GTC_DEFAULT_R2GTC_TABLE_55 \
  "16:2 37:2 40:2 55-56:2 61:2 64:2 99-100:5 103-104:4 109-112:4 " \
  "113-120:5 122-125:5 126-127:4 128-149:5 331-335:4"
#else
#define GTC_DEFAULT_GTC_TABLE_33 \
  "68-72:3 111-112:2 128:2"
#define GTC_DEFAULT_GTC_TABLE_44 \
  "68:4 69-72:3 88:4 91-92:4 96-99:4 161-162:3 164-168:3"
#define GTC_DEFAULT_GTC_TABLE_55 \
  "11:t 67-68:4 69-72:3 75-76:4 80:4 88-92:4 96-99:4 113-115:5 " \
  "129-140:5 145-155:5 157-185:5 189-191:5 193-200:5 202-205:5 " \
  "209-210:5 212-215:5 225-249:5 326-332:4 335-336:4 361-364:4 " \
  "366-368:4 376-378:4 380:4 384:4 421-428:4 431-432:4 437:4 " \
  "441-444:4 481-483:4"
#define GTC_DEFAULT_R2GTC_TABLE_44 \
  "9-10:2 41-42:3 45-48:3 54:3 58-59:3 61-64:2 103-104:4 " \
  "109-118:4 121-143:4 145:4 147-160:4 173-176:4 181-215:4 " \
  "217-240:4 242:4 244:4 253-256:4 259:4 262-263:4 271-291:4 " \
  "293-299:4"
#define GTC_DEFAULT_R2GTC_TABLE_55 \
  "9-10:2 41-42:3 45-48:3 51:3 53-54:3 58-59:3 61-64:2 103-107:4 " \
  "109-112:4 113-120:5 121-123:4 124:5 125-128:4 129-140:5 141:4 " \
  "142-149:5 221-224:4 326-329:4 331-336:4"
#endif
#ifndef GTC_DEFAULT_GTC_TABLE_33
#define GTC_DEFAULT_GTC_TABLE_33 NULL
#endif
#ifndef GTC_DEFAULT_GTC_TABLE_44
#define GTC_DEFAULT_GTC_TABLE_44 NULL
#endif
#ifndef GTC_DEFAULT_GTC_TABLE_55
#define GTC_DEFAULT_GTC_TABLE_55 NULL
#endif
#ifndef GTC_DEFAULT_R2GTC_TABLE_22
#define GTC_DEFAULT_R2GTC_TABLE_22 NULL
#endif
#ifndef GTC_DEFAULT_R2GTC_TABLE_33
#define GTC_DEFAULT_R2GTC_TABLE_33 NULL
#endif
#ifndef GTC_DEFAULT_R2GTC_TABLE_44
#define GTC_DEFAULT_R2GTC_TABLE_44 NULL
#endif
#ifndef GTC_DEFAULT_R2GTC_TABLE_55
#define GTC_DEFAULT_R2GTC_TABLE_55 NULL
#endif
