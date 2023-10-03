# author: leyla noroozbabaee
# author: leyla noroozbabaee
# created on fri April  5 14:02:47 2023

ps = 9.00
vs = 0.00
rte = 2.57  # rte-gas const. times temp  (Joule/mmol)
rt = 1.93e+4 # rt-gas const. times temp  (mmhg.ml/mmol)
f = 0.965e+5  # f-faraday (coulombs/mol)
rt_f = 0.0266362647

# z-   valence of i'th solute
z_na = 1
z_k = 1
z_cl = -1
z_hco3 = -1
z_gluc = 0

lis_nak = 0.1000e-08
lmi_clhco3 = 0.2000e-08
lmi_nagluc = 0.7500e-8

lis_kcl = 0.50e-06
lis_nahco3 = 0.050e-07  # 0.050e-07 #
lis_na_clhco3 = 0.50e-08  # 0.0700e-06 #
A_ATP = - 30

ame = 0.001
ae = 0.2000e-01  # aes0
aie = 0.3600e+02
ami = 0.3600e+02
ais = 0.1000e+01

vm = -60

cs_na = 0.140
cs_k = 0.0049
cs_cl = 0.1132
cs_hco3 = 0.024
cs_gluc = 0.005
print("that's the PCT_GLOB_new")

cm_na = 0.140
cm_k = 0.0049
cm_cl = 0.1132
cm_hco3 = 0.02400
cm_gluc = 0.005

m =5
sme_na = 0.750
ses_na = 0.000
hme_na = 0.1300e+01/m
hes_na = 0.5000e-01

sme_k = 0.600
ses_k = 0.000
hme_k = 0.1450e+01/m
hes_k = 0.7000e-01

sme_cl = 0.300
ses_cl = 0.000
hme_cl = 0.1000e+01/m
hes_cl = 0.6000e-01

sme_hco3 = 0.900
ses_hco3 = 0.000
hme_hco3 = 0.1000e+00/m
hes_hco3 = 0.5000e-01

# sme_gluc = 1.000
# ses_gluc = 0.000
# hme_gluc = 0.8000e-01/m
# hes_gluc = 0.3000e-01

sme_gluc = 1.0
ses_gluc = 0.00
hme_gluc = 0.0071
hes_gluc = 0.0016
his_gluc = 0.0

# luminal model  lpmi = 0.4000e-03
# epithelial model lpmi = 0.2000e-03
# luminal model  lpis = 0.4000e-03
# epithelial model lpis = 0.2000e-03
lpmi = 0.2000e-03
lpis = 0.2000e-03
lpie = 0.2000e-03
lpme = 0.2000e+02/m
lpes = 0.6000e+01
# luminal model  his_na = 0.7800e-08
# epithelial model his_na = 0.3900e-08

smi_na = 1.000
sis_na = 1.000
hmi_na = 0.0000e+00
his_na = 0.3900e-08
# luminal model  hmi_k = 0.500e-06  * his_k = 0.4000e-05
# epithelial model hmi_k = 0.2500e-06  * his_k = 0.2000e-05
smi_k = 1.000
sis_k = 1.000
hmi_k = 0.2500e-06
his_k = 0.2000e-05

smi_cl = 1.000
sis_cl = 1.000
hmi_cl = 0.0000e+00
his_cl = 0.0000e+00
# luminal model  hmi_hco3 = 0.2000e-07
# epithelial model hmi_hco3 = 0.1000e-07
## epithelial model hmi_hco3 = 0.1000e-07
smi_hco3 = 1.000
sis_hco3 = 1.000
hmi_hco3 = 0.1000e-07
his_hco3 = 0.0000e+00

smi_gluc = 1.000
sis_gluc = 1
hmi_gluc = 0.0000e+00
pm = 15.000



R = 8.314
T = 310
F = 96485
imp_init = 0.6000e-01
clvl_init = 0.001 #0.1000e-02
chvl0_init = 0.7000e-04
ci_na_init = 0.20443615264909884011e-01
ci_k_init = 0.103742335290954643678e+00
ci_cl_init = 0.16901004032978079322e-01
ci_hco3_init = 0.150905e-02
ci_gluc_init = 0.2487617424e-01
vi_init = -80
ve_init = -0
vm_init = 0
vs_init = -0
pe_init = -0.230937e+0
ps_init = 9
pm_init = 15
ce_na_init = 0.1404e+00
ce_k_init = 0.46537e-02
ce_cl_init = 0.1120e+00
ce_hco3_init = 0.20560787e-01
ce_gluc_init = 0.772369e-02
p_i_init = 0.0e-01
tf = 60000
T_F = 600000
# initial for the salt-sensitivity
# imp_init = 0.6000e-01
# clvl_init = 0.001 #0.1000e-02
# chvl0_init = 0.7000e-04
# ci_na_init = 0.020443615264909884011e-01
# ci_k_init = 0.103742335290954643678e+00
# ci_cl_init = 0.000016901004032978079322e-01
# ci_hco3_init = 0.00150905e-02
# ci_gluc_init = 0.002487617424e-01
# vi_init = -80
# ve_init = -0
# vm_init = 0
# vs_init = -0
# pe_init = -0.230937e+0
# ps_init = 9
# pm_init = 15
# ce_na_init = 0.0001404e+00
# ce_k_init = 0.0046537e-02
# ce_cl_init = 0.001120e+00
# ce_hco3_init = 0.0020560787e-01
# ce_gluc_init = 0.772369e-02
# p_i_init = 0.0e-01

c_elec = 10000000
scale_factor = 1
# param_clhco3_mi = Param[0]
# param_sglt_mi = Param[1]
# param_na_cl_hco3 = Param[2]
# param_nak = Param[3]
# param_kcl = Param[4]
# param_na_hco3 = Param[5]
