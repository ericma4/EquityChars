import polars as pl
import numpy as np
from tqdm import tqdm
import re

INPUT_PATH = "../data/raw/"
OUTPUT_PATH = "../data/processed/"

def ffi49():
    """
    Returns a Polars expression that classifies SIC codes into 49 Fama-French industries.
    Usage: df.with_columns(ffi49_polars().alias('ffi49'))
    """
    sic = pl.col('sic')
    
    return (
        pl.when(sic.is_between(100, 199) | sic.is_between(200, 299) | sic.is_between(700, 799) | 
                sic.is_between(910, 919) | (sic == 2048)).then(1)
        .when(sic.is_between(2000, 2009) | sic.is_between(2010, 2019) | sic.is_between(2020, 2029) | 
              sic.is_between(2030, 2039) | sic.is_between(2040, 2046) | sic.is_between(2050, 2059) | 
              sic.is_between(2060, 2063) | sic.is_between(2070, 2079) | sic.is_between(2090, 2092) | 
              (sic == 2095) | sic.is_between(2098, 2099)).then(2)
        .when(sic.is_between(2064, 2068) | (sic == 2086) | (sic == 2087) | (sic == 2096) | (sic == 2097)).then(3)
        .when(sic.is_between(2080, 2080) | (sic == 2082) | (sic == 2083) | (sic == 2084) | (sic == 2085)).then(4)
        .when(sic.is_between(2100, 2199)).then(5)
        .when(sic.is_between(920, 999) | sic.is_between(3650, 3652) | (sic == 3732) | 
              sic.is_between(3930, 3931) | sic.is_between(3940, 3949)).then(6)
        .when(sic.is_between(7800, 7829) | sic.is_between(7830, 7833) | sic.is_between(7840, 7841) | 
              (sic == 7900) | sic.is_between(7910, 7911) | sic.is_between(7920, 7929) | 
              sic.is_between(7930, 7933) | sic.is_between(7940, 7949) | (sic == 7980) | sic.is_between(7990, 7999)).then(7)
        .when(sic.is_between(2700, 2709) | sic.is_between(2710, 2719) | sic.is_between(2720, 2729) | 
              sic.is_between(2730, 2739) | sic.is_between(2740, 2749) | sic.is_between(2770, 2771) | 
              sic.is_between(2780, 2789) | sic.is_between(2790, 2799)).then(8)
        .when((sic == 2047) | sic.is_between(2391, 2392) | sic.is_between(2510, 2519) | sic.is_between(2590, 2599) | 
              sic.is_between(2840, 2844) | sic.is_between(3160, 3161) | sic.is_between(3170, 3172) | 
              sic.is_between(3190, 3199) | (sic == 3229) | (sic == 3260) | sic.is_between(3262, 3263) | (sic == 3269) | 
              sic.is_between(3230, 3231) | sic.is_between(3630, 3639) | sic.is_between(3750, 3751) | (sic == 3800) | 
              sic.is_between(3860, 3861) | sic.is_between(3870, 3873) | sic.is_between(3910, 3911) | (sic == 3914) | 
              (sic == 3915) | sic.is_between(3960, 3962) | (sic == 3991) | (sic == 3995)).then(9)
        .when(sic.is_between(2300, 2390) | sic.is_between(3020, 3021) | sic.is_between(3100, 3111) | 
              (sic == 3130) | (sic == 3131) | sic.is_between(3140, 3151) | sic.is_between(3963, 3965)).then(10)
        .when(sic.is_between(8000, 8099)).then(11)
        .when((sic == 3693) | sic.is_between(3840, 3851)).then(12)
        .when(sic.is_between(2830, 2836)).then(13)
        .when(sic.is_between(2800, 2809) | sic.is_between(2810, 2819) | sic.is_between(2820, 2829) | 
              sic.is_between(2850, 2859) | sic.is_between(2860, 2869) | sic.is_between(2870, 2879) | 
              sic.is_between(2890, 2899)).then(14)
        .when((sic == 3031) | (sic == 3041) | sic.is_between(3050, 3053) | sic.is_between(3060, 3069) | 
              sic.is_between(3070, 3079) | sic.is_between(3080, 3089) | sic.is_between(3090, 3099)).then(15)
        .when(sic.is_between(2200, 2269) | sic.is_between(2270, 2279) | sic.is_between(2280, 2284) | 
              sic.is_between(2290, 2295) | (sic == 2297) | (sic == 2298) | (sic == 2299) | 
              sic.is_between(2393, 2395) | sic.is_between(2397, 2399)).then(16)
        .when(sic.is_between(800, 899) | sic.is_between(2400, 2439) | sic.is_between(2450, 2459) | 
              sic.is_between(2490, 2499) | sic.is_between(2660, 2661) | sic.is_between(2950, 2952) | 
              (sic == 3200) | sic.is_between(3210, 3211) | sic.is_between(3240, 3241) | sic.is_between(3250, 3259) | 
              (sic == 3261) | (sic == 3264) | sic.is_between(3270, 3275) | sic.is_between(3280, 3281) | 
              sic.is_between(3290, 3293) | sic.is_between(3295, 3299) | sic.is_between(3420, 3429) | 
              sic.is_between(3430, 3433) | sic.is_between(3440, 3442) | (sic == 3446) | (sic == 3448) | 
              (sic == 3449) | sic.is_between(3450, 3452) | sic.is_between(3490, 3499) | (sic == 3996)).then(17)
        .when(sic.is_between(1500, 1511) | sic.is_between(1520, 1529) | sic.is_between(1530, 1539) | 
              sic.is_between(1540, 1549) | sic.is_between(1600, 1699) | sic.is_between(1700, 1799)).then(18)
        .when((sic == 3300) | sic.is_between(3310, 3317) | sic.is_between(3320, 3325) | sic.is_between(3330, 3339) | 
              sic.is_between(3340, 3341) | sic.is_between(3350, 3357) | sic.is_between(3360, 3369) | 
              sic.is_between(3370, 3379) | sic.is_between(3390, 3399)).then(19)
        .when((sic == 3400) | sic.is_between(3443, 3444) | sic.is_between(3460, 3469) | sic.is_between(3470, 3479)).then(20)
        .when(sic.is_between(3510, 3519) | sic.is_between(3520, 3529) | (sic == 3530) | (sic == 3531) | 
              (sic == 3532) | (sic == 3533) | (sic == 3534) | (sic == 3535) | (sic == 3536) | (sic == 3538) | 
              sic.is_between(3540, 3549) | sic.is_between(3550, 3559) | sic.is_between(3560, 3569) | 
              (sic == 3580) | (sic == 3581) | (sic == 3582) | (sic == 3585) | (sic == 3586) | 
              (sic == 3589) | sic.is_between(3590, 3599)).then(21)
        .when((sic == 3600) | sic.is_between(3610, 3613) | sic.is_between(3620, 3621) | sic.is_between(3623, 3629) | 
              sic.is_between(3640, 3646) | sic.is_between(3648, 3649) | (sic == 3660) | (sic == 3690) | 
              sic.is_between(3691, 3692) | sic.is_between(3699, 3699)).then(22)
        .when((sic == 2296) | (sic == 2396) | sic.is_between(3010, 3011) | (sic == 3537) | (sic == 3647) | 
              (sic == 3694) | (sic == 3700) | (sic == 3710) | (sic == 3711) | (sic == 3713) | 
              (sic == 3714) | (sic == 3715) | (sic == 3716) | (sic == 3792) | sic.is_between(3790, 3791) | 
              sic.is_between(3799, 3799)).then(23)
        .when((sic == 3720) | (sic == 3721) | sic.is_between(3723, 3725) | sic.is_between(3728, 3729)).then(24)
        .when(sic.is_between(3730, 3731) | sic.is_between(3740, 3743)).then(25)
        .when(sic.is_between(3760, 3769) | (sic == 3795) | sic.is_between(3480, 3489)).then(26)
        .when(sic.is_between(1040, 1049)).then(27)
        .when(sic.is_between(1000, 1009) | sic.is_between(1010, 1019) | sic.is_between(1020, 1029) | 
              sic.is_between(1030, 1039) | sic.is_between(1050, 1059) | sic.is_between(1060, 1069) | 
              sic.is_between(1070, 1079) | sic.is_between(1080, 1089) | sic.is_between(1090, 1099) | 
              sic.is_between(1100, 1119) | sic.is_between(1400, 1499)).then(28)
        .when(sic.is_between(1200, 1299)).then(29)
        .when((sic == 1300) | sic.is_between(1310, 1319) | sic.is_between(1320, 1329) | sic.is_between(1330, 1339) | 
              sic.is_between(1370, 1379) | (sic == 1380) | (sic == 1381) | (sic == 1382) | (sic == 1389) | 
              sic.is_between(2900, 2912) | sic.is_between(2990, 2999)).then(30)
        .when((sic == 4900) | sic.is_between(4910, 4911) | sic.is_between(4920, 4925) | (sic == 4930) | 
              (sic == 4931) | (sic == 4932) | (sic == 4939) | sic.is_between(4940, 4942)).then(31)
        .when((sic == 4800) | sic.is_between(4810, 4813) | sic.is_between(4820, 4822) | sic.is_between(4830, 4839) | 
              sic.is_between(4840, 4841) | sic.is_between(4880, 4889) | (sic == 4890) | (sic == 4891) | 
              (sic == 4892) | sic.is_between(4899, 4899)).then(32)
        .when(sic.is_between(7020, 7021) | sic.is_between(7030, 7033) | (sic == 7200) | sic.is_between(7210, 7212) | 
              (sic == 7214) | sic.is_between(7215, 7217) | sic.is_between(7219, 7221) | sic.is_between(7230, 7231) | 
              sic.is_between(7240, 7241) | sic.is_between(7250, 7251) | sic.is_between(7260, 7269) | 
              sic.is_between(7270, 7291) | sic.is_between(7292, 7299) | (sic == 7395) | (sic == 7500) | 
              sic.is_between(7520, 7529) | sic.is_between(7530, 7539) | sic.is_between(7540, 7549) | 
              (sic == 7600) | (sic == 7620) | (sic == 7622) | (sic == 7623) | (sic == 7629) | 
              sic.is_between(7630, 7631) | sic.is_between(7640, 7641) | sic.is_between(7690, 7699) | 
              sic.is_between(8100, 8199) | sic.is_between(8200, 8299) | sic.is_between(8300, 8399) | 
              sic.is_between(8400, 8499) | sic.is_between(8600, 8699) | sic.is_between(8800, 8899) | 
              sic.is_between(7510, 7515)).then(33)
        .when(sic.is_between(2750, 2759) | (sic == 3993) | (sic == 7218) | (sic == 7300) | 
              sic.is_between(7310, 7319) | sic.is_between(7320, 7329) | sic.is_between(7330, 7339) | 
              sic.is_between(7340, 7342) | (sic == 7349) | sic.is_between(7350, 7353) | (sic == 7359) | 
              sic.is_between(7360, 7369) | (sic == 7374) | (sic == 7376) | (sic == 7377) | (sic == 7378) | 
              (sic == 7379) | (sic == 7380) | sic.is_between(7381, 7385) | sic.is_between(7389, 7394) | 
              (sic == 7396) | (sic == 7397) | sic.is_between(7399, 7399) | (sic == 7519) | (sic == 8700) | 
              sic.is_between(8710, 8713) | sic.is_between(8720, 8721) | sic.is_between(8730, 8734) | 
              sic.is_between(8740, 8748) | sic.is_between(8900, 8911) | sic.is_between(8920, 8999) | 
              sic.is_between(4220, 4229)).then(34)
        .when(sic.is_between(3570, 3579) | (sic == 3680) | sic.is_between(3681, 3689) | (sic == 3695)).then(35)
        .when(sic.is_between(7370, 7372) | (sic == 7375) | (sic == 7373)).then(36)
        .when((sic == 3622) | sic.is_between(3661, 3666) | sic.is_between(3669, 3679) | (sic == 3810) | (sic == 3812)).then(37)
        .when((sic == 3811) | (sic == 3820) | sic.is_between(3821, 3827) | sic.is_between(3829, 3839)).then(38)
        .when(sic.is_between(2520, 2549) | sic.is_between(2600, 2639) | sic.is_between(2670, 2699) | 
              sic.is_between(2760, 2761) | sic.is_between(3950, 3955)).then(39)
        .when(sic.is_between(2440, 2449) | sic.is_between(2640, 2659) | sic.is_between(3220, 3221) | 
              sic.is_between(3410, 3412)).then(40)
        .when(sic.is_between(4000, 4013) | sic.is_between(4040, 4049) | (sic == 4100) | sic.is_between(4110, 4121) | 
              sic.is_between(4130, 4131) | sic.is_between(4140, 4142) | sic.is_between(4150, 4151) | 
              sic.is_between(4170, 4173) | sic.is_between(4190, 4199) | (sic == 4200) | sic.is_between(4210, 4219) | 
              sic.is_between(4230, 4231) | sic.is_between(4240, 4249) | sic.is_between(4400, 4499) | 
              sic.is_between(4500, 4599) | sic.is_between(4600, 4699) | (sic == 4700) | sic.is_between(4710, 4712) | 
              sic.is_between(4720, 4729) | sic.is_between(4730, 4739) | sic.is_between(4740, 4749) | 
              (sic == 4780) | (sic == 4782) | (sic == 4783) | (sic == 4784) | (sic == 4785) | sic.is_between(4789, 4789)).then(41)
        .when((sic == 5000) | sic.is_between(5010, 5015) | sic.is_between(5020, 5023) | sic.is_between(5030, 5039) | 
              sic.is_between(5040, 5049) | sic.is_between(5050, 5059) | (sic == 5060) | (sic == 5063) | 
              (sic == 5064) | (sic == 5065) | sic.is_between(5070, 5078) | (sic == 5080) | sic.is_between(5081, 5088) | 
              (sic == 5090) | sic.is_between(5091, 5094) | (sic == 5099) | (sic == 5100) | sic.is_between(5110, 5113) | 
              sic.is_between(5120, 5122) | sic.is_between(5130, 5139) | sic.is_between(5140, 5149) | 
              sic.is_between(5150, 5159) | sic.is_between(5160, 5169) | sic.is_between(5170, 5172) | 
              sic.is_between(5180, 5182) | sic.is_between(5190, 5199)).then(42)
        .when((sic == 5200) | sic.is_between(5210, 5219) | sic.is_between(5220, 5229) | sic.is_between(5230, 5231) | 
              sic.is_between(5250, 5251) | sic.is_between(5260, 5261) | sic.is_between(5270, 5271) | 
              (sic == 5300) | sic.is_between(5310, 5311) | (sic == 5320) | sic.is_between(5330, 5331) | 
              (sic == 5334) | sic.is_between(5340, 5349) | sic.is_between(5390, 5400) | sic.is_between(5410, 5412) | 
              sic.is_between(5420, 5429) | sic.is_between(5430, 5439) | sic.is_between(5440, 5449) | 
              sic.is_between(5450, 5459) | sic.is_between(5460, 5469) | sic.is_between(5490, 5500) | 
              sic.is_between(5510, 5529) | sic.is_between(5530, 5539) | sic.is_between(5540, 5549) | 
              sic.is_between(5550, 5559) | sic.is_between(5560, 5569) | sic.is_between(5570, 5579) | 
              sic.is_between(5590, 5599) | sic.is_between(5600, 5700) | sic.is_between(5710, 5722) | 
              sic.is_between(5730, 5736) | sic.is_between(5750, 5799) | (sic == 5900) | sic.is_between(5910, 5912) | 
              sic.is_between(5920, 5929) | sic.is_between(5930, 5932) | (sic == 5940) | sic.is_between(5941, 5949) | 
              sic.is_between(5950, 5959) | sic.is_between(5960, 5969) | sic.is_between(5970, 5979) | 
              sic.is_between(5980, 5990) | (sic == 5992) | (sic == 5993) | (sic == 5994) | (sic == 5995) | 
              sic.is_between(5999, 5999)).then(43)
        .when(sic.is_between(5800, 5819) | sic.is_between(5820, 5829) | sic.is_between(5890, 5899) | 
              (sic == 7000) | sic.is_between(7010, 7019) | sic.is_between(7040, 7049) | (sic == 7213)).then(44)
        .when((sic == 6000) | sic.is_between(6010, 6036) | sic.is_between(6040, 6062) | sic.is_between(6080, 6082) | 
              sic.is_between(6090, 6100) | sic.is_between(6110, 6113) | sic.is_between(6120, 6179) | 
              sic.is_between(6190, 6199)).then(45)
        .when((sic == 6300) | sic.is_between(6310, 6331) | sic.is_between(6350, 6351) | sic.is_between(6360, 6361) | 
              sic.is_between(6370, 6379) | sic.is_between(6390, 6411)).then(46)
        .when((sic == 6500) | (sic == 6510) | sic.is_between(6512, 6515) | sic.is_between(6517, 6519) | 
              sic.is_between(6520, 6532) | sic.is_between(6540, 6541) | sic.is_between(6550, 6553) | 
              sic.is_between(6590, 6599) | sic.is_between(6610, 6611)).then(47)
        .when(sic.is_between(6200, 6299) | (sic == 6700) | sic.is_between(6710, 6726) | sic.is_between(6730, 6733) | 
              sic.is_between(6740, 6779) | sic.is_between(6790, 6795) | (sic == 6798) | sic.is_between(6799, 6799)).then(48)
        .when(sic.is_between(4950, 4959) | sic.is_between(4960, 4961) | sic.is_between(4970, 4971) | 
              sic.is_between(4990, 4991)).then(49)
        .otherwise(None)
    )


def ffi30():
    """
    Returns a Polars expression that classifies SIC codes into 30 Fama-French industries.
    Usage: df.with_columns(ffi30_polars().alias('ffi30'))
    """
    sic = pl.col('sic')
    
    return (
        pl.when(sic.is_between(100, 199) | sic.is_between(200, 299) | sic.is_between(700, 799) | 
                sic.is_between(910, 919) | sic.is_between(2000, 2099)).then(1)
        .when(sic.is_between(2080, 2085)).then(2)
        .when(sic.is_between(2100, 2199)).then(3)
        .when(sic.is_between(920, 999) | sic.is_between(3650, 3652) | (sic == 3732) | 
              sic.is_between(3930, 3949) | sic.is_between(7800, 7999)).then(4)
        .when(sic.is_between(2700, 2799) | (sic == 3993)).then(5)
        .when((sic == 2047) | sic.is_between(2391, 2392) | sic.is_between(2510, 2519) | 
              sic.is_between(2590, 2599) | sic.is_between(2840, 2844) | sic.is_between(3160, 3172) | 
              sic.is_between(3190, 3199) | (sic == 3229) | (sic == 3260) | sic.is_between(3262, 3263) | 
              (sic == 3269) | sic.is_between(3230, 3231) | sic.is_between(3630, 3639) | 
              sic.is_between(3750, 3751) | (sic == 3800) | sic.is_between(3860, 3873) | 
              sic.is_between(3910, 3911) | (sic == 3914) | (sic == 3915) | sic.is_between(3960, 3962) | 
              (sic == 3991) | (sic == 3995)).then(6)
        .when(sic.is_between(2300, 2390) | sic.is_between(3020, 3021) | sic.is_between(3100, 3111) | 
              (sic == 3130) | (sic == 3131) | sic.is_between(3140, 3151) | sic.is_between(3963, 3965)).then(7)
        .when(sic.is_between(2830, 2836) | (sic == 3693) | sic.is_between(3840, 3851) | sic.is_between(8000, 8099)).then(8)
        .when(sic.is_between(2800, 2829) | sic.is_between(2850, 2899)).then(9)
        .when(sic.is_between(2200, 2284) | sic.is_between(2290, 2295) | (sic == 2297) | (sic == 2298) | 
              (sic == 2299) | sic.is_between(2393, 2399)).then(10)
        .when(sic.is_between(800, 899) | sic.is_between(1500, 1799) | sic.is_between(2400, 2439) | 
              sic.is_between(2450, 2459) | sic.is_between(2490, 2499) | sic.is_between(2660, 2661) | 
              sic.is_between(2950, 2952) | (sic == 3200) | sic.is_between(3210, 3211) | 
              sic.is_between(3240, 3241) | sic.is_between(3250, 3259) | (sic == 3261) | (sic == 3264) | 
              sic.is_between(3270, 3275) | sic.is_between(3280, 3281) | sic.is_between(3290, 3299) | 
              sic.is_between(3420, 3442) | (sic == 3446) | (sic == 3448) | (sic == 3449) | 
              sic.is_between(3450, 3452) | sic.is_between(3490, 3499) | (sic == 3996)).then(11)
        .when((sic == 3300) | sic.is_between(3310, 3317) | sic.is_between(3320, 3325) | 
              sic.is_between(3330, 3341) | sic.is_between(3350, 3357) | sic.is_between(3360, 3369) | 
              sic.is_between(3370, 3379) | sic.is_between(3390, 3399)).then(12)
        .when((sic == 3400) | sic.is_between(3443, 3444) | sic.is_between(3460, 3479) | 
              sic.is_between(3510, 3599)).then(13)
        .when((sic == 3600) | sic.is_between(3610, 3613) | sic.is_between(3620, 3621) | 
              sic.is_between(3623, 3629) | sic.is_between(3640, 3660) | (sic == 3690) | 
              sic.is_between(3691, 3692) | sic.is_between(3699, 3699)).then(14)
        .when((sic == 2296) | (sic == 2396) | sic.is_between(3010, 3011) | (sic == 3537) | 
              (sic == 3647) | (sic == 3694) | (sic == 3700) | sic.is_between(3710, 3716) | 
              (sic == 3792) | sic.is_between(3790, 3791) | sic.is_between(3799, 3799)).then(15)
        .when(sic.is_between(3720, 3721) | sic.is_between(3723, 3725) | sic.is_between(3728, 3731) | 
              sic.is_between(3740, 3743)).then(16)
        .when(sic.is_between(1000, 1119) | sic.is_between(1400, 1499)).then(17)
        .when(sic.is_between(1200, 1299)).then(18)
        .when((sic == 1300) | sic.is_between(1310, 1389) | sic.is_between(2900, 2912) | sic.is_between(2990, 2999)).then(19)
        .when((sic == 4900) | sic.is_between(4910, 4942)).then(20)
        .when((sic == 4800) | sic.is_between(4810, 4899)).then(21)
        .when(sic.is_between(7020, 7021) | sic.is_between(7030, 7033) | (sic == 7200) | 
              sic.is_between(7210, 7299) | (sic == 7395) | (sic == 7500) | sic.is_between(7510, 7549) | 
              (sic == 7600) | sic.is_between(7620, 7641) | sic.is_between(7690, 7699) | 
              sic.is_between(8100, 8199) | sic.is_between(8200, 8299) | sic.is_between(8300, 8399) | 
              sic.is_between(8400, 8499) | sic.is_between(8600, 8748) | sic.is_between(8800, 8999)).then(22)
        .when(sic.is_between(3570, 3579) | (sic == 3622) | sic.is_between(3661, 3679) | 
              sic.is_between(3680, 3689) | (sic == 3695) | sic.is_between(3810, 3812) | 
              sic.is_between(3820, 3839) | (sic == 7373)).then(23)
        .when(sic.is_between(2440, 2449) | sic.is_between(2520, 2549) | sic.is_between(2600, 2639) | 
              sic.is_between(2640, 2659) | sic.is_between(2670, 2699) | sic.is_between(2760, 2761) | 
              sic.is_between(3220, 3221) | sic.is_between(3410, 3412) | sic.is_between(3950, 3955)).then(24)
        .when(sic.is_between(4000, 4013) | sic.is_between(4040, 4049) | (sic == 4100) | 
              sic.is_between(4110, 4173) | sic.is_between(4190, 4231) | sic.is_between(4240, 4249) | 
              sic.is_between(4400, 4499) | sic.is_between(4500, 4599) | sic.is_between(4600, 4699) | 
              (sic == 4700) | sic.is_between(4710, 4749) | (sic == 4780) | sic.is_between(4782, 4789)).then(25)
        .when((sic == 5000) | sic.is_between(5010, 5199)).then(26)
        .when((sic == 5200) | sic.is_between(5210, 5999)).then(27)
        .when(sic.is_between(5800, 5829) | sic.is_between(5890, 5899) | (sic == 7000) | 
              sic.is_between(7010, 7019) | sic.is_between(7040, 7049) | (sic == 7213)).then(28)
        .when((sic == 6000) | sic.is_between(6010, 6799)).then(29)
        .when(sic.is_between(4950, 4961) | sic.is_between(4970, 4971) | sic.is_between(4990, 4991)).then(30)
        .otherwise(None)
    )


def ffi12():
    """
    Returns a Polars expression that classifies SIC codes into 12 Fama-French industries.
    Usage: df.with_columns(ffi12_polars().alias('ffi12'))
    """
    sic = pl.col('sic')
    
    return (
        pl.when(sic.is_between(100, 999) | sic.is_between(2000, 2399) | sic.is_between(2700, 2749) | 
                sic.is_between(2770, 2799) | sic.is_between(3100, 3199) | sic.is_between(3940, 3989)).then(1)
        .when(sic.is_between(2500, 2519) | sic.is_between(2590, 2599) | sic.is_between(3630, 3659) | 
              sic.is_between(3710, 3711) | (sic == 3714) | (sic == 3716) | sic.is_between(3750, 3751) | 
              (sic == 3792) | sic.is_between(3900, 3939) | sic.is_between(3990, 3999)).then(2)
        .when(sic.is_between(2520, 2589) | sic.is_between(2600, 2699) | sic.is_between(2750, 2769) | 
              sic.is_between(3000, 3099) | sic.is_between(3200, 3569) | sic.is_between(3580, 3629) | 
              sic.is_between(3700, 3709) | sic.is_between(3712, 3713) | (sic == 3715) | 
              sic.is_between(3717, 3749) | sic.is_between(3752, 3791) | sic.is_between(3793, 3799) | 
              sic.is_between(3830, 3839) | sic.is_between(3860, 3899)).then(3)
        .when(sic.is_between(1200, 1399) | sic.is_between(2900, 2999)).then(4)
        .when(sic.is_between(2800, 2829) | sic.is_between(2840, 2899)).then(5)
        .when(sic.is_between(3570, 3579) | sic.is_between(3660, 3692) | sic.is_between(3694, 3699) | 
              sic.is_between(3810, 3829) | sic.is_between(7370, 7379)).then(6)
        .when(sic.is_between(4800, 4899)).then(7)
        .when(sic.is_between(4900, 4949)).then(8)
        .when(sic.is_between(5000, 5999) | sic.is_between(7200, 7299) | sic.is_between(7600, 7699)).then(9)
        .when(sic.is_between(2830, 2839) | (sic == 3693) | sic.is_between(3840, 3859) | sic.is_between(8000, 8099)).then(10)
        .when(sic.is_between(6000, 6999)).then(11)
        .otherwise(12)
    )

#######################################################################################################################
#                                                    TTM functions                                                    #
#######################################################################################################################


def ttm4(series, df):
    """
    Calculate trailing 4-period sum (TTM4) using Polars.
    
    :param series: variables' name (string)
    :param df: polars dataframe (used to compute the result as a Series)
    :return: polars Expression that can be used in with_columns()
    
    Note: This function returns a Polars Expression for use in with_columns().
    Example: data_rawq.with_columns([ttm4('ibq', data_rawq).alias('ibq4')])
    """
    # Build expression for sum of current + 3 lags
    return (
        pl.col(series) + 
        pl.col(series).shift(1).over('permno') + 
        pl.col(series).shift(2).over('permno') + 
        pl.col(series).shift(3).over('permno')
    )


def ttm12(series, df):
    """
    Calculate trailing 12-period sum (TTM12) using Polars.
    
    :param series: variables' name (string)
    :param df: polars dataframe (used to compute the result as a Series)
    :return: polars Expression that can be used in with_columns()
    
    Note: This function returns a Polars Expression for use in with_columns().
    Example: crsp_mom.with_columns([(ttm12('mdivpay', crsp_mom) / pl.col('me')).alias('dy')])
    """
    # Build expression for sum of current + 11 lags
    return (
        pl.col(series) + 
        pl.col(series).shift(1).over('permno') + 
        pl.col(series).shift(2).over('permno') + 
        pl.col(series).shift(3).over('permno') +
        pl.col(series).shift(4).over('permno') + 
        pl.col(series).shift(5).over('permno') + 
        pl.col(series).shift(6).over('permno') + 
        pl.col(series).shift(7).over('permno') +
        pl.col(series).shift(8).over('permno') + 
        pl.col(series).shift(9).over('permno') + 
        pl.col(series).shift(10).over('permno') + 
        pl.col(series).shift(11).over('permno')
    )

def fillna_atq(df_q: pl.DataFrame, df_a: pl.DataFrame):
    """
    Use annual chars to fill null values in quarterly chars.
    Skips columns matching 'mom*' pattern.
    """
    # find columns that are null in df_q AND exist in df_a
    q_null_cols = [c for c in df_q.columns if df_q[c].is_null().any()]
    a_cols = df_a.columns
    candidates = list(set(q_null_cols) & set(a_cols))

    # exclude mom* columns
    na_columns_list = [c for c in candidates if not re.match(r'mom.', c)]

    if not na_columns_list:
        return df_q

    # extract annual cols + keys, rename to '*_a'
    df_temp = (
        df_a.select(['permno', 'date'] + na_columns_list)
        .rename({c: f'{c}_a' for c in na_columns_list})
    )

    # left join and coalesce
    df_q = df_q.join(df_temp, on=['permno', 'date'], how='left')
    df_q = df_q.with_columns([
        pl.coalesce([pl.col(c), pl.col(f'{c}_a')]).alias(c)
        for c in na_columns_list
    ]).drop([f'{c}_a' for c in na_columns_list])

    return df_q


def fillna_ind(
    df: pl.DataFrame,
    method: str,
    ffi: int,
    not_fill_col: list
):
    """
    Fill null values using industry-level mean or median grouped by date + ffi code.
    """
    ffi_col = f'ffi{ffi}'
    na_columns_list = [
        c for c in df.columns
        if df[c].is_null().any() and c not in not_fill_col
    ]

    if not na_columns_list:
        return df

    if method == 'mean':
        agg_exprs = [pl.col(c).mean().alias(f'{c}_fill') for c in na_columns_list]
    elif method == 'median':
        agg_exprs = [pl.col(c).median().alias(f'{c}_fill') for c in na_columns_list]
    else:
        raise ValueError(f"method must be 'mean' or 'median', got '{method}'")

    df_fill = (
        df.group_by(['date', ffi_col])
        .agg(agg_exprs)
    )

    df = df.join(df_fill, on=['date', ffi_col], how='left')
    df = df.with_columns([
        pl.coalesce([pl.col(c), pl.col(f'{c}_fill')]).alias(c)
        for c in na_columns_list
    ]).drop([f'{c}_fill' for c in na_columns_list])

    return df


def fillna_all(
    df: pl.DataFrame,
    method: str,
    not_fill_col: list
):
    """
    Fill null values using cross-sectional mean or median grouped by date.
    """
    na_columns_list = [
        c for c in df.columns
        if df[c].is_null().any() and c not in not_fill_col
    ]

    if not na_columns_list:
        return df

    if method == 'mean':
        agg_exprs = [pl.col(c).mean().alias(f'{c}_fill') for c in na_columns_list]
    elif method == 'median':
        agg_exprs = [pl.col(c).median().alias(f'{c}_fill') for c in na_columns_list]
    else:
        raise ValueError(f"method must be 'mean' or 'median', got '{method}'")

    df_fill = (
        df.group_by('date')
        .agg(agg_exprs)
    )

    df = df.join(df_fill, on='date', how='left')
    df = df.with_columns([
        pl.coalesce([pl.col(c), pl.col(f'{c}_fill')]).alias(c)
        for c in na_columns_list
    ]).drop([f'{c}_fill' for c in na_columns_list])

    return df


def standardize(df: pl.DataFrame):
    """
    Cross-sectionally rank and standardize all char columns to [-1, 1].
    Excludes info columns. Null ranks are filled with 0.
    """
    INFO_COLS = {
        'permno', 'date', 'datadate', 'gvkey', 'sic', 'count',
        'exchcd', 'shrcd', 'ffi49', 'ret', 'retadj', 'retx',
        'lag_me', 'ticker', 'conm', 'comnam', 'prc', 'shrout'
    }
    col_names = [c for c in df.columns if c not in INFO_COLS]

    for col_name in tqdm(col_names):
        # dense rank within each date, then scale to [-1, 1]
        df = df.with_columns([
            pl.col(col_name)
              .rank(method='dense')
              .over('date')
              .alias(f'_rank_{col_name}')
        ])
        # count non-null unique values per date
        df = df.with_columns([
            pl.col(f'_rank_{col_name}')
              .max()
              .over('date')
              .alias('_max_rank')
        ])
        df = df.with_columns([
            pl.when(pl.col('_max_rank') > 1)
              .then(
                (pl.col(f'_rank_{col_name}') - 1) /
                (pl.col('_max_rank') - 1) * 2 - 1
              )
              .otherwise(None)
              .fill_null(0)
              .alias(f'rank_{col_name}')
        ]).drop([col_name, f'_rank_{col_name}', '_max_rank'])

    return df
