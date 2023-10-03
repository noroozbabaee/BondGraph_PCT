import numpy as np
import scipy
import math
import pandas as pd
import matplotlib.cm as cm
import decimal
import pprint
import pickle
import math
import PCT_GLOB_new
from PCT_GLOB_new import *
from scipy.integrate import odeint
import altair as alt
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure


def diff_volume_flow(lp, pa, pb, ca, cb, reflect, a):
    ca =[0 if x is None else x for x in ca]
    cb =[0 if x is None else x for x in cb]
    osmotic_a = sum([a * b for a, b in zip(reflect, ca)])
    flux_va = lp * (pa - rt * osmotic_a) / rt
    osmotic_b = sum([a * b for a, b in zip(reflect, cb)])
    flux_vb = lp * (pb - rt * osmotic_b) / rt
    diff_flux_v = a*(flux_va - flux_vb)
    return diff_flux_v


def osmotic_pressure(ca, rt):
    osmotic =[i1 * rt for i1 in ca]
    return osmotic


def diff_osmotic_pressure(ca, cb, rt):
    tot_osmotic = sum([(i1 - i2) * rt for i1, i2 in zip(ca, cb)])
    return tot_osmotic


def solute_concen_mean(ca, cb):
    ff =[]
    for i, j in zip(ca, cb):
        if i is not None and j is not None and np.any(i > 0) and np.any(j > 0):
            cab_bar = (i - j) / (np.log10(i) - np.log10(j))
            ff.append(cab_bar)
        else:
            cab_bar = i
            ff.append(cab_bar)
    return ff


def convective_solute_flux(diff_flux_v, reflect, cab_bar,  param_csf):
    if param_csf == 0:
        return 0
    r1 =[1 - i for i in reflect]
    r2 =[a * b if b is not None else 0 for a, b in zip(r1, cab_bar) ]
    flux_conv =[i1 * diff_flux_v if i1 is not None else 0 for i1 in r2]
    return flux_conv


def f_eps(c, z, v):
    ff =[]
    for i in range(5):
        if c[i] is not None and np.any(c[i] > 0):
            f_eps = rte * np.log10(c[i]) + z[i] * f * v * 1.e-6
            ff.append(f_eps)
        elif c[i] is not None:
            f_eps = rte * np.log10(abs(c[i])) + z[i] * f * v * 1.e-6
            ff.append(f_eps)

        else:
            ff.append(0)
    return ff


def k_cl(xi_k, xs_k, xi_cl, xs_cl, l_kcl, param_kcl):
    if param_kcl == 0:
        return[0, 0]
    else:
        fec_k_is = xi_k - xs_k
        fec_cl_is = xi_cl - xs_cl
        k_kcl = l_kcl * (fec_k_is + fec_cl_is)
        cl_kcl = l_kcl * (fec_k_is + fec_cl_is)
    return[k_kcl, cl_kcl]


def na_k_is(xi_na, xs_na, xi_k, xs_k,  l_nak, A_ATP, param_nak):
    if param_nak == 0:
        return[0,0]
    else:
        fec_na_is = (xi_na - xs_na)
        fec_k_is = (xi_k - xs_k)
        J = 1*l_nak * (A_ATP - 3 * fec_na_is + 2 * fec_k_is)
    return[-3*J, 2*J]


def solute_flux(flux_conv, f_electro_chemical_a,f_electro_chemical_b,  cab_bar, hab, a, active_transport):
    diff_mu =[a - b for a, b in zip(f_electro_chemical_a, f_electro_chemical_b)]
    diffusion1 =[a * b if b is not None else 0 for a, b in zip(diff_mu, cab_bar)]
    diffusion2 =[a * b/(rt) if a is not None else 0 for a, b in zip(diffusion1, hab)]
    flux1 =[i1 + a * i2 if i2 is not None else 0 for i1, i2 in zip(flux_conv, diffusion2)]
    flux0 =[(i1 + a * (i2 if i2 is not None else 0)) for i1, i2 in zip(flux1, active_transport)]
    return flux0


def pressure(lp, slt_cnst, p):
    tp = lp * (p - rt * slt_cnst) / rt
    return tp


def Slt_Cnst_FODE(diff_volume_flow, solute_flux,  clvl, ci):
    rv =[i * (diff_volume_flow / clvl) for i in ci]
    ri =[i / clvl for i in solute_flux]
    EQ_slt_cnst =[a - b for a, b in zip(rv, ri)]
    return EQ_slt_cnst


def Voltage_FODE(solute_flux, z, c_elec):
    result_a =[a * b for a, b in zip(solute_flux, z)]
    tot_charge = sum(result_a)
    EQ_Voltage = - f * tot_charge / c_elec
    return EQ_Voltage


def Hydrolic_Pressure_FODE(diff_volume_flow, k_comp):
    EQ_Hydrolic_Pressure = ((diff_volume_flow)) / k_comp
    return EQ_Hydrolic_Pressure


def cvEQni(ni, clvl):
     ci =[i / clvl for i in ni]
     return ci


def clhco3_mi(xm_cl, xi_cl, xm_hco3, xi_hco3,  lmi_clhco3, param_clhco3_mi):
    if param_clhco3_mi == 0:
        return[0, 0]
    else:
        cl_mi_clhco3 = lmi_clhco3 * (xm_cl - xi_cl - xm_hco3 + xi_hco3)
        hco3_mi_clhco3 = -lmi_clhco3 * (xm_cl - xi_cl - xm_hco3 + xi_hco3)
    return[cl_mi_clhco3, hco3_mi_clhco3]


def sglt_mi(xi_na, xi_gluc, xm_na, xm_gluc, lmi_nagluc, param_sglt_mi):
    # See Eq: ()
    if param_sglt_mi == 0:
        return[0, 0]
    else:
        na_mi_nagluc = lmi_nagluc * (xm_na - xi_na + xm_gluc - xi_gluc)
        gluc_mi_nagluc = lmi_nagluc * (xm_na - xi_na + xm_gluc - xi_gluc)
    return[na_mi_nagluc, gluc_mi_nagluc]


def na_hco3_is(xi_na, xi_hco3, xs_na, xs_hco3, lis_nahco3, param_na_hco3):
    # See Eq: ()
    if param_na_hco3 == 0:
        return[0, 0]
    else:
        na_is_nahco3 = lis_nahco3 * (xi_na - xs_na + 3 * (xi_hco3 - xs_hco3))
        hco3_is_nahco3 = 3 * lis_nahco3 * (xi_na - xs_na + 3 * (xi_hco3 - xs_hco3))
        return[na_is_nahco3, hco3_is_nahco3]


def na_cl_hco3_is(xi_na,  xi_cl,  xi_hco3, xs_na , xs_cl, xs_hco3, lis_na_clhco3, param_na_cl_hco3):
    # See Eq: ()
    if param_na_cl_hco3 == 0:
        return[0, 0, 0]
    else:
        na_na_clhco3 = lis_na_clhco3 * (xi_na - xs_na - xi_cl + xs_cl + 2 * (xi_hco3 - xs_hco3))
        cl_na_clhco3 = -  lis_na_clhco3 * (xi_na - xs_na - xi_cl + xs_cl + 2 * (xi_hco3 - xs_hco3))
        hco3_na_clhco3 = 2 * lis_na_clhco3 * (xi_na - xs_na - xi_cl + xs_cl + 2 * (xi_hco3 - xs_hco3))
        return[na_na_clhco3, cl_na_clhco3, hco3_na_clhco3]


solver = 1


def dSdt(tt, S, solver, Param):

    decimal.getcontext().prec = 123

    ci_na, ci_k, ci_cl, ci_hco3, ci_gluc, ce_na, ce_k, ce_cl, ce_hco3, ce_gluc, clvl, chvl,  vi,  ve,  p_i,  pe = S
    param_csf = 1
    # Solute Concentrations
    ci =[ci_na, ci_k, ci_cl, ci_hco3, ci_gluc]
    ce =[ce_na, ce_k, ce_cl, ce_hco3, ce_gluc]
    cm =[cm_na, cm_k, cm_cl, cm_hco3, cm_gluc]
    cs =[cs_na, cs_k, cs_cl, cs_hco3, cs_gluc]
    # Reflection coefficients
    S_me =[sme_na, sme_k, sme_cl, sme_hco3,  sme_gluc]
    S_es =[ses_na/10, ses_k/10, ses_cl/10, ses_hco3,  ses_gluc]
    S_mi =[smi_na, smi_k, smi_cl, smi_hco3, smi_gluc]
    S_is =[sis_na, sis_k, sis_cl, sis_hco3, sis_gluc]

    # Permeability coefficients
    h_em =[hme_na, hme_k, hme_cl, hme_hco3,  hme_gluc]
    h_es =[hes_na, hes_k, hes_cl, hes_hco3, hes_gluc]
    h_mi =[hmi_na, hmi_k, hmi_cl, hmi_hco3, hmi_gluc]
    h_is =[his_na, his_k, his_cl, his_hco3, his_gluc]

    # z-valence of i'th solute
    z =[z_na, z_k, z_cl, z_hco3, z_gluc]

    S_ie = S_is
    h_ie = h_is
    fivm = diff_volume_flow(lpmi, pm, p_i, cm, ci, S_mi, ami)
    fevm = diff_volume_flow(lpme, pm, pe, cm, ce, S_me, ame)
    fivs = diff_volume_flow(lpis, p_i, ps, ci, cs, S_is,ais)
    five = diff_volume_flow(lpie, p_i, pe, ci, ce, S_ie, aie)
    fevs = diff_volume_flow(lpes, pe, ps, ce, cs, S_is, ae)

    mu_s = f_eps(cs, z, vs)
    mu_e = f_eps(ce, z, ve)
    mu_m = f_eps(cm, z, vm)
    mu_i = f_eps(ci, z, vi)

    cmi_bar = solute_concen_mean(cm, ci)
    param_clhco3_mi = Param[0]
    param_sglt_mi = Param[1]
    param_na_cl_hco3 = Param[2]
    param_nak = Param[3]
    param_kcl = Param[4]
    param_na_hco3 = Param[5]

    flux_na_gluc_mi = sglt_mi(mu_i[0], mu_i[4], mu_m[0], mu_m[4], lmi_nagluc/150, param_sglt_mi)

    flux_clhco3_mi = clhco3_mi(mu_m[2], mu_i[2], mu_m[3], mu_i[3], lmi_clhco3, param_clhco3_mi)
    active_transport_mi =[flux_na_gluc_mi[0], 0 * flux_na_gluc_mi[0], flux_clhco3_mi[0], flux_clhco3_mi[1], flux_na_gluc_mi[1]] #% need to change zero to a Zero vector
    fikm_c = convective_solute_flux(fivm, S_mi, cmi_bar, param_csf)
    fikm = solute_flux(fikm_c, mu_m, mu_i, cmi_bar, h_mi, ami, active_transport_mi)  # Apical Membrane

    flux_k_cl_is = k_cl(mu_i[1], mu_s[1], mu_i[2], mu_s[2],  lis_kcl, param_kcl)
    flux_na_k_is = na_k_is(mu_i[0], mu_s[0], mu_i[1], mu_s[1], lis_nak, A_ATP, param_nak)
    flux_na_hco3_is = na_hco3_is(mu_i[0], mu_i[3], mu_s[0], mu_s[3], lis_nahco3/10, param_na_hco3)
    flux_na_cl_hco3_is = na_cl_hco3_is(mu_i[0], mu_i[2], mu_i[3], mu_s[0], mu_s[2], mu_s[3],  100*lis_na_clhco3, param_na_cl_hco3)

    cis_bar = solute_concen_mean(ci, cs)
    active_transport_is =[flux_na_k_is[0]+flux_na_hco3_is[0] + flux_na_cl_hco3_is[0],  flux_k_cl_is[0] + flux_na_k_is[1],   flux_k_cl_is[1] + flux_na_cl_hco3_is[1] , flux_na_hco3_is[1] + flux_na_cl_hco3_is[2], 0]#[0 * i for i in S_es] #
    fiks_c = convective_solute_flux(fivs, S_is, cis_bar, param_csf)
    fiks = solute_flux(fiks_c, mu_i, mu_s, cis_bar, h_is, ais, active_transport_is)  # Basolateral Membrane

    flux_k_cl_ie = k_cl(mu_i[ 1 ], mu_e[ 1 ], mu_i[ 2 ], mu_e[ 2 ], lis_kcl*130, param_kcl)

    flux_na_k_ie = na_k_is(mu_i[ 0 ], mu_e[ 0 ], mu_i[ 1 ], mu_e[ 1 ], lis_nak/15, A_ATP, param_nak)

    flux_na_hco3_ie = na_hco3_is(mu_i[ 0 ], mu_i[ 3 ], mu_e[ 0 ], mu_e[ 3 ], lis_nahco3/100, param_na_hco3)

    flux_na_cl_hco3_ie = na_cl_hco3_is(mu_i[ 0 ], mu_i[ 2 ], mu_i[ 3 ], mu_e[ 0 ], mu_e[ 2 ], mu_e[ 3 ],
                                       20*lis_na_clhco3, param_na_cl_hco3)

    active_transport_ie =[ flux_na_k_ie[ 0 ] + flux_na_hco3_ie[ 0 ] + flux_na_cl_hco3_ie[ 0 ],
                            flux_k_cl_ie[ 0 ] + flux_na_k_ie[ 1 ], flux_k_cl_ie[ 1 ] + flux_na_cl_hco3_ie[ 1 ],
                            flux_na_hco3_ie[ 1 ] + flux_na_cl_hco3_ie[ 2 ], 0 ]  # [0 * i for i in S_es] #
    cie_bar = solute_concen_mean(ci, ce)
    fike_c = convective_solute_flux(five, S_ie, cie_bar, param_csf)
    fike = solute_flux(fike_c, mu_i, mu_e, cie_bar, h_ie, aie, active_transport_ie)  # Tight-Junction Membrane

    cme_bar = solute_concen_mean(cm, ce)
    active_transport_me =[0 * i for i in S_me]
    fekm_c = convective_solute_flux(fevm, S_me, cme_bar, param_csf)
    fekm = solute_flux(fekm_c, mu_m, mu_e, cme_bar, h_em, ame, active_transport_me)  # Tight-Junction Membrane

    ces_bar = solute_concen_mean(ce, cs)
    active_transport_es =[0 * i for i in S_es]
    feks_c = convective_solute_flux(fevs, S_es, ces_bar, param_csf)
    feks = solute_flux(feks_c, mu_e, mu_s, ces_bar, h_es, ae, active_transport_es)  # Membrane

    f1_i =[a + b for a, b in zip(fikm, fekm)]
    f1_v = fivm + fevm

    f2_i =[a - b for a, b in zip(fiks, fikm)]
    f2_i =[a + b for a, b in zip(f2_i, fike)]
    f2_v = fivs - fivm + five

    f3_i =[-a + b for a, b in zip(fiks, feks)]
    f3_v = -fivs + fevs

    f4_i =[-a - b for a, b in zip(fike, fekm)]
    f4_i =[a - b for a, b in zip(f4_i, feks)]
    f4_v = -five - fevm - fevs

    Cell = Slt_Cnst_FODE(f2_v, f2_i, clvl, ci)

    Cell =[num * scale_factor for num in Cell]
    ILS = Slt_Cnst_FODE(f4_v, f4_i, chvl, ce)
    ILS =[num * scale_factor for num in ILS]

    p_i = 0
    pe = 0
    clvl = 0
    chvl = 0
    vi = Voltage_FODE([a - b for a, b in zip(f2_i, f1_i)],[i1*ami for i1 in z], c_elec) * scale_factor
    ve =  Voltage_FODE(f4_i, z, c_elec) * scale_factor

    def flatten(x):
        result =[ ]
        for el in x:
            if hasattr(el, "__iter__") and not isinstance(el, str):
                result.extend(flatten(el))
            else:
                result.append(el)
        return result

    if solver == 1:
        rslts =[ Cell, ILS, clvl, chvl, vi, ve, p_i, pe]
        return flatten(rslts)
    else:
        rtrn = [flux_na_gluc_mi, flux_clhco3_mi, active_transport_mi, fikm_c, fikm, fivm, flux_k_cl_is, flux_na_k_is, flux_na_cl_hco3_is, flux_na_hco3_is, active_transport_is, fiks_c, fiks, fivs,  fike_c, fike, five,  fekm_c, fekm, fevm, feks_c, feks, fevs, f1_i, f1_v, f2_i, f2_v, f3_i, f3_v, f4_i, f4_v]
        return rtrn


tt = np.linspace(0, tf, T_F)

Figure_default = 0
if Figure_default:
    S_0 = (ci_na_init, ci_k_init, ci_cl_init, ci_hco3_init, ci_gluc_init, ce_na_init, ce_k_init, ce_cl_init, ce_hco3_init, ce_gluc_init,  clvl_init,  chvl0_init,  vi_init,    ve_init,  p_i_init, pe_init)
    Param =[1, 1, 1, 1, 1, 1]
    sol = odeint(dSdt, y0=S_0, t=tt, tfirst=True, args=(solver, Param))
    ci_na = sol.T[0]
    ci_k = sol.T[1]
    ci_cl = sol.T[2]
    ci_hco3 = sol.T[3]
    ci_gluc= sol.T[4]
    ce_na = sol.T[5]
    ce_k = sol.T[6]
    ce_cl = sol.T[7]
    ce_hco3 = sol.T[8]
    ce_gluc= sol.T[9]
    clvl = sol.T[10]
    chvl = sol.T[11]
    vi = sol.T[12]
    ve = sol.T[13]
    p_i = sol.T[14]
    pe = sol.T[15]

    S = [ci_na, ci_k, ci_cl, ci_hco3, ci_gluc, ce_na, ce_k, ce_cl, ce_hco3, ce_gluc, clvl,  chvl, vi, ve, p_i, pe]
    Results = dSdt(tt, S, solver, Param)
    solver = 0
    fluxes = dSdt(tt, S, solver, Param)
    na_nagluc_mi, gluc_nagluc_mi = fluxes[0]
    cl_clhco3_mi , hco3_clhco3_mi = fluxes[1]
    na_active_transport_mi, cl_active_transport_mi, k_active_transport_mi, hco3_active_transport_mi, gluc_active_transport_mi = fluxes[2]
    fikm_c_na, fikm_c_cl, fikm_c_k, fikm_c_hco3, fikm_c_gluc = fluxes[3]
    fikm_na, fikm_cl, fikm_k, fikm_hco3, fikm_gluc = fluxes[4]
    fivm = fluxes[5]
    k_kcl_is, cl_kcl_is  = fluxes[6]
    na_nak_is,  k_nak_is  =  fluxes[7]
    na_nacl_hco3_is,cl_nacl_hco3_is,hco3_nacl_hco3_is = fluxes[8]
    na_nahco3_is, hco3_nahco3_is = fluxes[9]
    na_active_transport_is, cl_active_transport_is, k_active_transport_is, hco3_active_transport_is, gluc_active_transport_is = fluxes[10]
    fiks_c_na, fiks_c_cl, fiks_c_k, fiks_c_hco3, fiks_c_gluc = fluxes[11]
    fiks_na, fiks_cl, fiks_k, fiks_hco3, fiks_gluc = fluxes[12]
    fivs = fluxes[13]

    fike_c_na, fike_c_cl, fike_c_k, fike_c_hco3, fike_c_gluc = fluxes[ 14 ]
    fike_na, fike_cl, fike_k, fike_hco3, fike_gluc = fluxes[ 15 ]
    five = fluxes[ 16 ]

    fekm_c_na, fekm_c_cl, fekm_c_k, fekm_c_hco3, fekm_c_gluc = fluxes[ 17 ]
    fekm_na, fekm_cl, fekm_k, fekm_hco3, fekm_gluc = fluxes[ 18 ]
    fevm = fluxes[ 19 ]


    feks_c_na, feks_c_cl, feks_c_k, feks_c_hco3, feks_c_gluc = fluxes[ 20 ]
    feks_na, feks_cl, feks_k, feks_hco3, feks_gluc = fluxes[ 21]
    fevs = fluxes[ 22 ]

    default = {'na':[ fekm_na[ -1 ], fikm_na[ -1 ], fiks_na[ -1 ], fike_na[ -1 ], feks_na[ -1 ] ],
                       'k':[ fekm_k[ -1 ], fikm_k[ -1 ], fiks_k[ -1 ], fike_k[ -1 ], feks_k[ -1 ] ],
                       'cl':[ fekm_cl[ -1 ], fikm_cl[ -1 ], fiks_cl[ -1 ], fike_cl[ -1 ], feks_cl[ -1 ] ],
                       'gluc':[ fekm_gluc[ -1 ], fikm_gluc[ -1 ], fiks_gluc[ -1 ], fike_gluc[ -1 ],
                                 feks_gluc[ -1 ] ],
                       'fekm':[ fekm_na[ -1 ], fekm_k[ -1 ], fekm_cl[ -1 ], fekm_gluc[ -1 ] ],
                       'fikm':[ fikm_na[ -1 ], fikm_k[ -1 ], fikm_cl[ -1 ], fikm_gluc[ -1 ] ],
                       'fiks':[ fiks_na[ -1 ], fiks_k[ -1 ], fiks_cl[ -1 ], fiks_gluc[ -1 ] ],
                       'fike':[ fike_na[ -1 ], fike_k[ -1 ], fike_cl[ -1 ], fike_gluc[ -1 ] ],
                       'feks':[ feks_na[ -1 ], feks_k[ -1 ], feks_cl[ -1 ], feks_gluc[ -1 ] ]}
    flux_na = feks_na[ -1 ] + fike_na[ -1 ] + fiks_na[ -1 ] + fekm_na[ -1 ] + fikm_na[ -1 ]
    slt_con_default = {'Na': ci_na[ -1 ], 'K': ci_k[ -1 ], 'Cl': ci_cl[ -1 ], 'Gluc': ci_gluc[ -1 ]}
    flx_na_mem_default = { 'feks_na':feks_na[ -1 ],  'fike_na':fike_na[ -1 ],  'fiks_na':fiks_na[ -1 ],  'fekm_na':fekm_na[ -1 ],  'fikm_na':fikm_na[ -1 ] }
    flx_Epithelial_na = fekm_na[ -1 ] + fikm_na[ -1 ]
    flx_Epithelial_Convective_na = fekm_c_na[ -1 ] + fikm_c_na[-1]
    flx_Epithelial_Passive_na = flx_Epithelial_na - flx_Epithelial_Convective_na
    epithelial_flx_variation_default_na = {'flx_Epithelial_na': flx_Epithelial_na, 'flx_Epithelial_Convective_na':flx_Epithelial_Convective_na,'flx_Epithelial_Passive_na':flx_Epithelial_Passive_na,
                                           'na_nagluc_mi':na_nagluc_mi[-1]}

    from collections import defaultdict
    Figure_default = defaultdict(list)
    for d in (default, slt_con_default , flx_na_mem_default ,  epithelial_flx_variation_default_na):
        for key, value in d.items():
            Figure_default[key].append(value)

    pickled_list = pickle.dumps(Figure_default)
    of = open('MI_Receptors_Default.py', 'wb')
    of.write(pickled_list)
    of.close()

    fig1 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig1)
    ax0 = fig1.add_subplot(spec2[ 0, 0])
    ax0.set_title('Cellular Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ci_na,'g',linewidth=2, label='Na_i')
    ax0.set_ylabel('ci_na', color='blue')

    ax1 = fig1.add_subplot(spec2[ 1, 0])
    ax1.plot(tt, ci_cl,'k',linewidth=2, label='Cl_i')
    ax1.set_ylabel('ci_cl', color='blue')

    ax2 = fig1.add_subplot(spec2[ 2, 0])
    ax2.plot(tt, ci_k,'r',linewidth=2, label='K_i')
    ax2.set_ylabel('ci_k ', color='blue')

    ax3 = fig1.add_subplot(spec2[ 3, 0])
    ax3.plot(tt, ci_hco3,'b',linewidth=2, label='hco3_i')
    ax3.set_ylabel('ci_hco3 ', color='blue')

    ax4 = fig1.add_subplot(spec2[ 4, 0])
    ax4.plot(tt, ci_gluc,'m',linewidth=2, label='gluc_i')
    ax4.set_ylabel('ci_gluc', color='blue')

    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig2)
    ax0 = fig2.add_subplot(spec2[ 0, 0])
    ax0.set_title('Interspace Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ce_na,'g',linewidth=2, label='Na_e')
    ax0.set_ylabel('ce_na', color='blue')

    ax1 = fig2.add_subplot(spec2[ 1, 0])
    ax1.plot(tt, ce_cl,'k',linewidth=2, label='Cl_e')
    ax1.set_ylabel('ce_cl', color='blue')

    ax2 = fig2.add_subplot(spec2[ 2, 0])
    ax2.plot(tt, ce_k,'r',linewidth=2, label='K_e')
    ax2.set_ylabel('ce_k', color='blue')

    ax3 = fig2.add_subplot(spec2[ 3, 0])
    ax3.plot(tt, ce_hco3,'b',linewidth=2, label='hco3_e')
    ax3.set_ylabel('ce_hco3', color='blue')
    ax4 = fig2.add_subplot(spec2[ 4, 0])
    ax4.plot(tt, ce_gluc,'m',linewidth=2, label='gluc_e')
    ax4.set_ylabel('ce_gluc', color='blue')
    fig4 = plt.figure(constrained_layout=False,  figsize=(10, 10))
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig4)
    ax1 = fig4.add_subplot(spec2[0, 0])
    ax1.set_title('Electrochemical Fluxes', color='#868686')


    x1 = ax1.plot(tt, fluxes[ 0 ][ 0 ], 'g')
    ax1.set_ylabel('flux_na_gluc_mi[M/sec]', color='#868686')
    plt.setp(plt.gca(), xticklabels=[ ])

    ax2 = fig4.add_subplot(spec2[1, 0 ])
    ax2.set_ylabel('flux_clhco3_mi[M/sec]', color='#868686')
    x2 = ax2.plot( tt, fluxes[ 1 ][ 0 ], 'g')

    ax3 = fig4.add_subplot(spec2[2, 0 ])
    ax3.set_ylabel('flux_k_cl_is[M/sec]', color='#868686')
    ax3 = ax3.plot(tt, fluxes[ 2 ][0], 'r', tt, fluxes[ 2 ][2], 'r', tt, fluxes[ 2 ][3], 'k', tt, fluxes[ 2 ][4], 'y')

    ax4 = fig4.add_subplot(spec2[ 3, 0 ])
    ax4.set_ylabel('ve', color='#868686')
    x4 = ax4.plot(tt, ve, 'g')

    ax5 = fig4.add_subplot(spec2[ 4, 0 ])
    ax5.set_ylabel('vi', color='#868686')
    x5 = ax5.plot(tt, vi, 'g')
    print(vi)
    plt.show()
Figure_No_ClHCO3 = 0
if Figure_No_ClHCO3:
    S_0 = (ci_na_init, ci_k_init, ci_cl_init, ci_hco3_init, ci_gluc_init, ce_na_init, ce_k_init, ce_cl_init , ce_hco3_init, ce_gluc_init,  clvl_init,  chvl0_init,  vi_init,    ve_init,  p_i_init, pe_init)
    tt = np.linspace(0, tf, T_F )
    Param =[0, 1, 1, 1, 1, 1]
    sol = odeint(dSdt, y0=S_0, t=tt, tfirst=True, args=(solver,Param))
    ci_na = sol.T[0]
    ci_k = sol.T[1]
    ci_cl = sol.T[2]
    ci_hco3 = sol.T[3]
    ci_gluc= sol.T[4]
    ce_na = sol.T[5]
    ce_k = sol.T[6]
    ce_cl = sol.T[7]
    ce_hco3 = sol.T[8]
    ce_gluc= sol.T[9]
    clvl = sol.T[10]
    chvl = sol.T[11]
    vi = sol.T[12]
    ve = sol.T[13]
    p_i = sol.T[14]
    pe = sol.T[15]
    S =[ci_na, ci_k, ci_cl, ci_hco3, ci_gluc, ce_na, ce_k, ce_cl, ce_hco3, ce_gluc, clvl,  chvl, vi,    ve,  p_i,  pe]
    Results = dSdt(tt, S, solver, Param)
    solver = 0
    fluxes = dSdt(tt, S, solver,Param)
    na_nagluc_mi, gluc_nagluc_mi = fluxes[0]
    cl_clhco3_mi , hco3_clhco3_mi = fluxes[1]
    na_active_transport_mi, cl_active_transport_mi, k_active_transport_mi, hco3_active_transport_mi, gluc_active_transport_mi = fluxes[2]
    fikm_c_na, fikm_c_cl, fikm_c_k, fikm_c_hco3, fikm_c_gluc = fluxes[3]
    fikm_na, fikm_cl, fikm_k, fikm_hco3, fikm_gluc = fluxes[4]
    fivm = fluxes[5]
    k_kcl_is, cl_kcl_is  = fluxes[6]
    na_nak_is,  k_nak_is  =  fluxes[7]
    na_nacl_hco3_is,cl_nacl_hco3_is,hco3_nacl_hco3_is = fluxes[8]
    na_nahco3_is, hco3_nahco3_is = fluxes[9]
    na_active_transport_is, cl_active_transport_is, k_active_transport_is, hco3_active_transport_is, gluc_active_transport_is = fluxes[10]
    fiks_c_na, fiks_c_cl, fiks_c_k, fiks_c_hco3, fiks_c_gluc = fluxes[11]
    fiks_na, fiks_cl, fiks_k, fiks_hco3, fiks_gluc = fluxes[12]
    fivs = fluxes[13]

    fike_c_na, fike_c_cl, fike_c_k, fike_c_hco3, fike_c_gluc = fluxes[ 14 ]
    fike_na, fike_cl, fike_k, fike_hco3, fike_gluc = fluxes[ 15 ]
    five = fluxes[ 16 ]

    fekm_c_na, fekm_c_cl, fekm_c_k, fekm_c_hco3, fekm_c_gluc = fluxes[ 17 ]
    fekm_na, fekm_cl, fekm_k, fekm_hco3, fekm_gluc = fluxes[ 18 ]
    fevm = fluxes[ 19 ]

    feks_c_na, feks_c_cl, feks_c_k, feks_c_hco3, feks_c_gluc = fluxes[ 20 ]
    feks_na, feks_cl, feks_k, feks_hco3, feks_gluc = fluxes[ 21]
    fevs = fluxes[ 22 ]

    No_ClHCO3 = {'na':[ fekm_na[ -1 ], fikm_na[ -1 ], fiks_na[ -1 ], fike_na[ -1 ], feks_na[ -1 ] ],
                       'k':[ fekm_k[ -1 ], fikm_k[ -1 ], fiks_k[ -1 ], fike_k[ -1 ], feks_k[ -1 ] ],
                       'cl':[ fekm_cl[ -1 ], fikm_cl[ -1 ], fiks_cl[ -1 ], fike_cl[ -1 ], feks_cl[ -1 ] ],
                       'gluc':[ fekm_gluc[ -1 ], fikm_gluc[ -1 ], fiks_gluc[ -1 ], fike_gluc[ -1 ],
                                 feks_gluc[ -1 ] ],
                       'fekm':[ fekm_na[ -1 ], fekm_k[ -1 ], fekm_cl[ -1 ], fekm_gluc[ -1 ] ],
                       'fikm':[ fikm_na[ -1 ], fikm_k[ -1 ], fikm_cl[ -1 ], fikm_gluc[ -1 ] ],
                       'fiks':[ fiks_na[ -1 ], fiks_k[ -1 ], fiks_cl[ -1 ], fiks_gluc[ -1 ] ],
                       'fike':[ fike_na[ -1 ], fike_k[ -1 ], fike_cl[ -1 ], fike_gluc[ -1 ] ],
                       'feks':[ feks_na[ -1 ], feks_k[ -1 ], feks_cl[ -1 ], feks_gluc[ -1 ] ]}
    flux_na_No_ClHCO3 = feks_na[ -1 ] + fike_na[ -1 ] + fiks_na[ -1 ] + fekm_na[ -1 ] + fikm_na[ -1 ]
    slt_con_No_ClHCO3 = {'Na': ci_na[ -1 ], 'K': ci_k[ -1 ], 'Cl': ci_cl[ -1 ], 'Gluc': ci_gluc[ -1 ]}
    flx_na_mem_No_ClHCO3 = {'feks_na':feks_na[ -1 ],  'fike_na':fike_na[ -1 ],  'fiks_na':fiks_na[ -1 ],  'fekm_na':fekm_na[ -1 ],  'fikm_na':fikm_na[ -1 ]}
    flx_Epithelial_na = fekm_na[ -1 ] + fikm_na[ -1 ]
    flx_Epithelial_Convective_na = fekm_c_na[ -1 ] + fikm_c_na[ -1 ]
    flx_Epithelial_Passive_na = flx_Epithelial_na - flx_Epithelial_Convective_na
    epithelial_flx_variation_na_No_ClHCO3 = {'flx_Epithelial_na': flx_Epithelial_na, 'flx_Epithelial_Convective_na':flx_Epithelial_Convective_na,'flx_Epithelial_Passive_na':flx_Epithelial_Passive_na,
                                            'na_nagluc_mi': 0}

    from collections import defaultdict
    Figure_No_ClHCO3 = defaultdict(list)
    for d in (No_ClHCO3, slt_con_No_ClHCO3 , flx_na_mem_No_ClHCO3 ,  epithelial_flx_variation_na_No_ClHCO3):
        for key, value in d.items():
            Figure_No_ClHCO3[key].append(value)

    pickled_list = pickle.dumps(Figure_No_ClHCO3)
    of = open('MI_Receptors_No_ClHCO3.py', 'wb')
    of.write(pickled_list)
    of.close()

    fig1 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig1)
    ax0 = fig1.add_subplot(spec2[ 0, 0])
    ax0.set_title('Cellular Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ci_na,'g',linewidth=2, label='Na_i')
    ax0.set_ylabel('ci_na', color='blue')

    ax1 = fig1.add_subplot(spec2[ 1, 0])
    ax1.plot(tt, ci_cl,'k',linewidth=2, label='Cl_i')
    ax1.set_ylabel('ci_cl', color='blue')

    ax2 = fig1.add_subplot(spec2[ 2, 0])
    ax2.plot(tt, ci_k,'r',linewidth=2, label='K_i')
    ax2.set_ylabel('ci_k ', color='blue')

    ax3 = fig1.add_subplot(spec2[ 3, 0])
    ax3.plot(tt, ci_hco3,'b',linewidth=2, label='hco3_i')
    ax3.set_ylabel('ci_hco3 ', color='blue')

    ax4 = fig1.add_subplot(spec2[ 4, 0])
    ax4.plot(tt, ci_gluc,'m',linewidth=2, label='gluc_i')
    ax4.set_ylabel('ci_gluc', color='blue')

    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig2)
    ax0 = fig2.add_subplot(spec2[ 0, 0])
    ax0.set_title('Interspace Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ce_na,'g',linewidth=2, label='Na_e')
    ax0.set_ylabel('ce_na', color='blue')

    ax1 = fig2.add_subplot(spec2[ 1, 0])
    ax1.plot(tt, ce_cl,'k',linewidth=2, label='Cl_e')
    ax1.set_ylabel('ce_cl', color='blue')

    ax2 = fig2.add_subplot(spec2[ 2, 0])
    ax2.plot(tt, ce_k,'r',linewidth=2, label='K_e')
    ax2.set_ylabel('ce_k', color='blue')

    ax3 = fig2.add_subplot(spec2[ 3, 0])
    ax3.plot(tt, ce_hco3,'b',linewidth=2, label='hco3_e')
    ax3.set_ylabel('ce_hco3', color='blue')
    ax4 = fig2.add_subplot(spec2[ 4, 0])
    ax4.plot(tt, ce_gluc,'m',linewidth=2, label='gluc_e')
    ax4.set_ylabel('ce_gluc', color='blue')
    fig4 = plt.figure(constrained_layout=False,  figsize=(10, 10))
    spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig4)
    ax1 = fig4.add_subplot(spec2[0, 0])
    ax1.set_title('Electrochemical Fluxes', color='#868686')
    plt.setp(plt.gca(), xticklabels=[ ])
    plt.show()

Figure_No_NaGluc = 1
if Figure_No_NaGluc:
    S_0 = (ci_na_init, ci_k_init, ci_cl_init, ci_hco3_init, ci_gluc_init, ce_na_init, ce_k_init, ce_cl_init , ce_hco3_init, ce_gluc_init,  clvl_init,  chvl0_init,  vi_init,    ve_init,  p_i_init, pe_init)
    tt = np.linspace(0, tf, T_F)
    Param =[1, 0, 1, 1, 1, 1]
    sol = odeint(dSdt, y0=S_0, t=tt, tfirst=True, args=(solver,Param))
    ci_na = sol.T[0]
    ci_k = sol.T[1]
    ci_cl = sol.T[2]
    ci_hco3 = sol.T[3]
    ci_gluc= sol.T[4]
    ce_na = sol.T[5]
    ce_k = sol.T[6]
    ce_cl = sol.T[7]
    ce_hco3 = sol.T[8]
    ce_gluc= sol.T[9]
    clvl = sol.T[10]
    chvl = sol.T[11]
    vi = sol.T[12]
    ve = sol.T[13]
    p_i = sol.T[14]
    pe = sol.T[15]
    S =[ci_na, ci_k, ci_cl, ci_hco3, ci_gluc, ce_na, ce_k, ce_cl, ce_hco3, ce_gluc, clvl,  chvl, vi,    ve,  p_i,  pe]
    Results = dSdt(tt, S, solver, Param)
    solver = 0
    fluxes = dSdt(tt, S, solver,Param)
    na_nagluc_mi, gluc_nagluc_mi = fluxes[0]
    cl_clhco3_mi , hco3_clhco3_mi = fluxes[1]
    na_active_transport_mi, cl_active_transport_mi, k_active_transport_mi, hco3_active_transport_mi, gluc_active_transport_mi = fluxes[2]
    fikm_c_na, fikm_c_cl, fikm_c_k, fikm_c_hco3, fikm_c_gluc = fluxes[3]
    fikm_na, fikm_cl, fikm_k, fikm_hco3, fikm_gluc = fluxes[4]
    fivm = fluxes[5]
    k_kcl_is, cl_kcl_is  = fluxes[6]
    na_nak_is,  k_nak_is  =  fluxes[7]
    na_nacl_hco3_is,cl_nacl_hco3_is,hco3_nacl_hco3_is = fluxes[8]
    na_nahco3_is, hco3_nahco3_is = fluxes[9]
    na_active_transport_is, cl_active_transport_is, k_active_transport_is, hco3_active_transport_is, gluc_active_transport_is = fluxes[10]
    fiks_c_na, fiks_c_cl, fiks_c_k, fiks_c_hco3, fiks_c_gluc = fluxes[11]
    fiks_na, fiks_cl, fiks_k, fiks_hco3, fiks_gluc = fluxes[12]
    fivs = fluxes[13]

    fike_c_na, fike_c_cl, fike_c_k, fike_c_hco3, fike_c_gluc = fluxes[ 14 ]
    fike_na, fike_cl, fike_k, fike_hco3, fike_gluc = fluxes[ 15 ]
    five = fluxes[ 16 ]

    fekm_c_na, fekm_c_cl, fekm_c_k, fekm_c_hco3, fekm_c_gluc = fluxes[ 17 ]
    fekm_na, fekm_cl, fekm_k, fekm_hco3, fekm_gluc = fluxes[ 18 ]
    fevm = fluxes[ 19 ]


    feks_c_na, feks_c_cl, feks_c_k, feks_c_hco3, feks_c_gluc = fluxes[ 20 ]
    feks_na, feks_cl, feks_k, feks_hco3, feks_gluc = fluxes[ 21]
    fevs = fluxes[ 22 ]


    No_NaGluc = {'na':[ fekm_na[ -1 ], fikm_na[ -1 ], fiks_na[ -1 ], fike_na[ -1 ], feks_na[ -1 ] ],
                       'k':[ fekm_k[ -1 ], fikm_k[ -1 ], fiks_k[ -1 ], fike_k[ -1 ], feks_k[ -1 ] ],
                       'cl':[ fekm_cl[ -1 ], fikm_cl[ -1 ], fiks_cl[ -1 ], fike_cl[ -1 ], feks_cl[ -1 ] ],
                       'gluc':[ fekm_gluc[ -1 ], fikm_gluc[ -1 ], fiks_gluc[ -1 ], fike_gluc[ -1 ],
                                 feks_gluc[ -1 ] ],
                       'fekm':[ fekm_na[ -1 ], fekm_k[ -1 ], fekm_cl[ -1 ], fekm_gluc[ -1 ] ],
                       'fikm':[ fikm_na[ -1 ], fikm_k[ -1 ], fikm_cl[ -1 ], fikm_gluc[ -1 ] ],
                       'fiks':[ fiks_na[ -1 ], fiks_k[ -1 ], fiks_cl[ -1 ], fiks_gluc[ -1 ] ],
                       'fike':[ fike_na[ -1 ], fike_k[ -1 ], fike_cl[ -1 ], fike_gluc[ -1 ] ],
                       'feks':[ feks_na[ -1 ], feks_k[ -1 ], feks_cl[ -1 ], feks_gluc[ -1 ] ]}
    flux_na_No_NaGluc = feks_na[ -1 ] + fike_na[ -1 ] + fiks_na[ -1 ] + fekm_na[ -1 ] + fikm_na[ -1 ]
    slt_con_No_NaGluc = {'Na': ci_na[ -1 ], 'K': ci_k[ -1 ], 'Cl': ci_cl[ -1 ], 'Gluc': ci_gluc[ -1 ]}
    flx_na_mem_No_NaGluc = { 'feks_na':feks_na[ -1 ],  'fike_na':fike_na[ -1 ],  'fiks_na':fiks_na[ -1 ],  'fekm_na':fekm_na[ -1 ],  'fikm_na':fikm_na[ -1 ] }
    flx_Epithelial_na = fekm_na[ -1 ] + fikm_na[ -1 ]
    flx_Epithelial_Convective_na = fekm_c_na[ -1 ] + fikm_c_na[ -1 ]
    flx_Epithelial_Passive_na = flx_Epithelial_na - flx_Epithelial_Convective_na
    epithelial_flx_variation_na_No_NaGluc = {'flx_Epithelial_na': flx_Epithelial_na, 'flx_Epithelial_Convective_na':flx_Epithelial_Convective_na,'flx_Epithelial_Passive_na':flx_Epithelial_Passive_na,
                                            'na_nagluc_mi':0 }

    from collections import defaultdict
    Figure_No_NaGluc = defaultdict(list)
    for d in (No_NaGluc, slt_con_No_NaGluc , flx_na_mem_No_NaGluc ,  epithelial_flx_variation_na_No_NaGluc):
        for key, value in d.items():
            Figure_No_NaGluc[key].append(value)

    pickled_list = pickle.dumps(Figure_No_NaGluc)
    of = open('MI_Receptors_No_NaGluc.py', 'wb')
    of.write(pickled_list)
    of.close()




    fig1 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig1)
    ax0 = fig1.add_subplot(spec2[ 0, 0])
    ax0.set_title('Cellular Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ci_na,'g',linewidth=2, label='Na_i')
    ax0.set_ylabel('ci_na', color='blue')

    ax1 = fig1.add_subplot(spec2[ 1, 0])
    ax1.plot(tt, ci_cl,'k',linewidth=2, label='Cl_i')
    ax1.set_ylabel('ci_cl', color='blue')

    ax2 = fig1.add_subplot(spec2[ 2, 0])
    ax2.plot(tt, ci_k,'r',linewidth=2, label='K_i')
    ax2.set_ylabel('ci_k ', color='blue')

    ax3 = fig1.add_subplot(spec2[ 3, 0])
    ax3.plot(tt, ci_hco3,'b',linewidth=2, label='hco3_i')
    ax3.set_ylabel('ci_hco3 ', color='blue')

    ax4 = fig1.add_subplot(spec2[ 4, 0])
    ax4.plot(tt, ci_gluc,'m',linewidth=2, label='gluc_i')
    ax4.set_ylabel('ci_gluc', color='blue')

    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig2)
    ax0 = fig2.add_subplot(spec2[ 0, 0])
    ax0.set_title('Interspace Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ce_na,'g',linewidth=2, label='Na_e')
    ax0.set_ylabel('ce_na', color='blue')

    ax1 = fig2.add_subplot(spec2[ 1, 0])
    ax1.plot(tt, ce_cl,'k',linewidth=2, label='Cl_e')
    ax1.set_ylabel('ce_cl', color='blue')

    ax2 = fig2.add_subplot(spec2[ 2, 0])
    ax2.plot(tt, ce_k,'r',linewidth=2, label='K_e')
    ax2.set_ylabel('ce_k', color='blue')

    ax3 = fig2.add_subplot(spec2[ 3, 0])
    ax3.plot(tt, ce_hco3,'b',linewidth=2, label='hco3_e')
    ax3.set_ylabel('ce_hco3', color='blue')
    ax4 = fig2.add_subplot(spec2[ 4, 0])
    ax4.plot(tt, ce_gluc,'m',linewidth=2, label='gluc_e')
    ax4.set_ylabel('ce_gluc', color='blue')
    fig4 = plt.figure(constrained_layout=False,  figsize=(10, 10))
    spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig4)
    ax1 = fig4.add_subplot(spec2[0, 0])
    ax1.set_title('Electrochemical Fluxes', color='#868686')
    plt.setp(plt.gca(), xticklabels=[ ])

    ax2 = fig4.add_subplot(spec2[1, 0 ])
    ax2.set_ylabel('flux_clhco3_mi[M/sec]', color='#868686')
    x2 = ax2.plot( tt, fluxes[ 1 ][ 0 ], 'g')
    ax2 = fig4.add_subplot(spec2[2, 0 ])
    ax2.set_ylabel('flux_k_cl_is[M/sec]', color='#868686')
    ax2 = ax2.plot( tt, fluxes[ 2 ][2], 'r', tt, fluxes[ 2 ][3], 'k')
    # ax2 = ax2.plot(tt, fluxes[ 2 ][0], 'r', tt, fluxes[ 2 ][2], 'r', tt, fluxes[ 2 ][3], 'k', tt, fluxes[ 2 ][4], 'y')
    plt.show()
Figure_No_Na_Cl_HCO3 = 0
if Figure_No_Na_Cl_HCO3:
    S_0 = (
        ci_na_init, ci_k_init, ci_cl_init, ci_hco3_init, ci_gluc_init, ce_na_init, ce_k_init, ce_cl_init,
        ce_hco3_init,
        ce_gluc_init, clvl_init, chvl0_init, vi_init, ve_init, p_i_init, pe_init)
    tt = np.linspace(0, tf, T_F)
    Param =[ 1, 1, 0, 1, 1, 1 ]
    # param_clhco3_mi = Param[0]
    # param_sglt_mi = Param[1]
    # param_na_cl_hco3 = Param[2]
    # param_nak = Param[3]
    # param_kcl = Param[4]
    # param_na_hco3 = Param[5]
    sol = odeint(dSdt, y0=S_0, t=tt, tfirst=True, args=(solver, Param))
    ci_na = sol.T[ 0 ]
    ci_k = sol.T[ 1 ]
    ci_cl = sol.T[ 2 ]
    ci_hco3 = sol.T[ 3 ]
    ci_gluc = sol.T[ 4 ]
    ce_na = sol.T[ 5 ]
    ce_k = sol.T[ 6 ]
    ce_cl = sol.T[ 7 ]
    ce_hco3 = sol.T[ 8 ]
    ce_gluc = sol.T[ 9 ]
    clvl = sol.T[ 10 ]
    chvl = sol.T[ 11 ]
    vi = sol.T[ 12 ]
    ve = sol.T[ 13 ]
    p_i = sol.T[ 14 ]
    pe = sol.T[ 15 ]
    S =[ ci_na, ci_k, ci_cl, ci_hco3, ci_gluc, ce_na, ce_k, ce_cl, ce_hco3, ce_gluc, clvl, chvl, vi, ve, p_i, pe ]
    Results = dSdt(tt, S, solver, Param)
    solver = 0
    fluxes = dSdt(tt, S, solver, Param)

    na_nagluc_mi, gluc_nagluc_mi = fluxes[ 0 ]
    cl_clhco3_mi, hco3_clhco3_mi = fluxes[ 1 ]
    na_active_transport_mi, cl_active_transport_mi, k_active_transport_mi, hco3_active_transport_mi, gluc_active_transport_mi = \
        fluxes[ 2 ]
    fikm_c_na, fikm_c_cl, fikm_c_k, fikm_c_hco3, fikm_c_gluc = fluxes[ 3 ]
    fikm_na, fikm_cl, fikm_k, fikm_hco3, fikm_gluc = fluxes[ 4 ]
    fivm = fluxes[ 5 ]
    k_kcl_is, cl_kcl_is = fluxes[ 6 ]
    na_nak_is, k_nak_is = fluxes[ 7 ]
    na_nacl_hco3_is, cl_nacl_hco3_is, hco3_nacl_hco3_is = fluxes[ 8 ]
    na_nahco3_is, hco3_nahco3_is = fluxes[ 9 ]
    na_active_transport_is, cl_active_transport_is, k_active_transport_is, hco3_active_transport_is, gluc_active_transport_is = \
        fluxes[ 10 ]
    fiks_c_na, fiks_c_cl, fiks_c_k, fiks_c_hco3, fiks_c_gluc = fluxes[ 11 ]
    fiks_na, fiks_cl, fiks_k, fiks_hco3, fiks_gluc = fluxes[ 12 ]
    fivs = fluxes[ 13 ]

    fike_c_na, fike_c_cl, fike_c_k, fike_c_hco3, fike_c_gluc = fluxes[ 14 ]
    fike_na, fike_cl, fike_k, fike_hco3, fike_gluc = fluxes[ 15 ]
    five = fluxes[ 16 ]

    fekm_c_na, fekm_c_cl, fekm_c_k, fekm_c_hco3, fekm_c_gluc = fluxes[ 17 ]
    fekm_na, fekm_cl, fekm_k, fekm_hco3, fekm_gluc = fluxes[ 18 ]
    fevm = fluxes[ 19 ]

    feks_c_na, feks_c_cl, feks_c_k, feks_c_hco3, feks_c_gluc = fluxes[ 20 ]
    feks_na, feks_cl, feks_k, feks_hco3, feks_gluc = fluxes[ 21 ]
    fevs = fluxes[ 22 ]

    No_Na_Cl_HCO3= {'na':[ fekm_na[ -1 ], fikm_na[ -1 ], fiks_na[ -1 ], fike_na[ -1 ], feks_na[ -1 ] ],
                       'k':[ fekm_k[ -1 ], fikm_k[ -1 ], fiks_k[ -1 ], fike_k[ -1 ], feks_k[ -1 ] ],
                       'cl':[ fekm_cl[ -1 ], fikm_cl[ -1 ], fiks_cl[ -1 ], fike_cl[ -1 ], feks_cl[ -1 ] ],
                       'gluc':[ fekm_gluc[ -1 ], fikm_gluc[ -1 ], fiks_gluc[ -1 ], fike_gluc[ -1 ],
                                 feks_gluc[ -1 ] ],
                       'fekm':[ fekm_na[ -1 ], fekm_k[ -1 ], fekm_cl[ -1 ], fekm_gluc[ -1 ] ],
                       'fikm':[ fikm_na[ -1 ], fikm_k[ -1 ], fikm_cl[ -1 ], fikm_gluc[ -1 ] ],
                       'fiks':[ fiks_na[ -1 ], fiks_k[ -1 ], fiks_cl[ -1 ], fiks_gluc[ -1 ] ],
                       'fike':[ fike_na[ -1 ], fike_k[ -1 ], fike_cl[ -1 ], fike_gluc[ -1 ] ],
                       'feks':[ feks_na[ -1 ], feks_k[ -1 ], feks_cl[ -1 ], feks_gluc[ -1 ] ]}
    flux_na = feks_na[ -1 ] + fike_na[ -1 ] + fiks_na[ -1 ] + fekm_na[ -1 ] + fikm_na[ -1 ]
    slt_con_No_Na_Cl_HCO3 = {'Na': ci_na[ -1 ], 'K': ci_k[ -1 ], 'Cl': ci_cl[ -1 ], 'Gluc': ci_gluc[ -1 ]}
    flx_na_mem_No_Na_Cl_HCO3 = {'feks_na': feks_na[ -1 ], 'fike_na': fike_na[ -1 ], 'fiks_na': fiks_na[ -1 ],
                          'fekm_na': fekm_na[ -1 ], 'fikm_na': fikm_na[ -1 ]}
    flx_Epithelial_na = fekm_na[ -1 ] + fikm_na[ -1 ]
    flx_Epithelial_Convective_na = fekm_c_na[ -1 ] + fikm_c_na[ -1 ]
    flx_Epithelial_Passive_na = flx_Epithelial_na - flx_Epithelial_Convective_na
    epithelial_flx_variation_na_No_Na_Cl_HCO3 = {'flx_Epithelial_na': flx_Epithelial_na,
                                           'flx_Epithelial_Convective_na': flx_Epithelial_Convective_na,
                                           'flx_Epithelial_Passive_na': flx_Epithelial_Passive_na,
                                           'na_nagluc_mi': na_nagluc_mi}

    from collections import defaultdict

    Figure_No_Na_Cl_HCO3= defaultdict(list)
    for d in (No_Na_Cl_HCO3 , slt_con_No_Na_Cl_HCO3, flx_na_mem_No_Na_Cl_HCO3 , epithelial_flx_variation_na_No_Na_Cl_HCO3):
        for key, value in d.items():
            Figure_No_Na_Cl_HCO3[ key ].append(value)

    pickled_list = pickle.dumps(Figure_No_Na_Cl_HCO3)
    of = open('MI_Receptors_No_Na_Cl_HCO3.py', 'wb')
    of.write(pickled_list)
    of.close()
    fig1 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig1)
    ax0 = fig1.add_subplot(spec2[ 0, 0 ])
    ax0.set_title('Cellular Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ci_na, 'g', linewidth=2, label='Na_i')
    ax0.set_ylabel('ci_na', color='blue')

    ax1 = fig1.add_subplot(spec2[ 1, 0 ])
    ax1.plot(tt, ci_cl, 'k', linewidth=2, label='Cl_i')
    ax1.set_ylabel('ci_cl', color='blue')

    ax2 = fig1.add_subplot(spec2[ 2, 0 ])
    ax2.plot(tt, ci_k, 'r', linewidth=2, label='K_i')
    ax2.set_ylabel('ci_k ', color='blue')

    ax3 = fig1.add_subplot(spec2[ 3, 0 ])
    ax3.plot(tt, ci_hco3, 'b', linewidth=2, label='hco3_i')
    ax3.set_ylabel('ci_hco3 ', color='blue')

    ax4 = fig1.add_subplot(spec2[ 4, 0 ])
    ax4.plot(tt, ci_gluc, 'm', linewidth=2, label='gluc_i')
    ax4.set_ylabel('ci_gluc', color='blue')

    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig2)
    ax0 = fig2.add_subplot(spec2[ 0, 0 ])
    ax0.set_title('Interspace Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ce_na, 'g', linewidth=2, label='Na_e')
    ax0.set_ylabel('ce_na', color='blue')

    ax1 = fig2.add_subplot(spec2[ 1, 0 ])
    ax1.plot(tt, ce_cl, 'k', linewidth=2, label='Cl_e')
    ax1.set_ylabel('ce_cl', color='blue')

    ax2 = fig2.add_subplot(spec2[ 2, 0 ])
    ax2.plot(tt, ce_k, 'r', linewidth=2, label='K_e')
    ax2.set_ylabel('ce_k', color='blue')

    ax3 = fig2.add_subplot(spec2[ 3, 0 ])
    ax3.plot(tt, ce_hco3, 'b', linewidth=2, label='hco3_e')
    ax3.set_ylabel('ce_hco3', color='blue')
    ax4 = fig2.add_subplot(spec2[ 4, 0 ])
    ax4.plot(tt, ce_gluc, 'm', linewidth=2, label='gluc_e')
    ax4.set_ylabel('ce_gluc', color='blue')
    fig4 = plt.figure(constrained_layout=False, figsize=(10, 10))
    spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig4)
    ax1 = fig4.add_subplot(spec2[ 0, 0 ])
    ax1.set_title('Electrochemical Fluxes', color='#868686')

    plt.setp(plt.gca(), xticklabels=[ ])

    ax2 = fig4.add_subplot(spec2[ 1, 0 ])
    ax2.set_ylabel('flux_clhco3_mi[M/sec]', color='#868686')
    x2 = ax2.plot(tt, fluxes[ 1 ][ 0 ], 'g')

    ax2 = fig4.add_subplot(spec2[ 2, 0 ])
    ax2.set_ylabel('flux_k_cl_is[M/sec]', color='#868686')
    ax2 = ax2.plot(tt, fluxes[ 2 ][ 0 ], 'b', tt, fluxes[ 2 ][ 1 ], 'r', tt, fluxes[ 2 ][ 2 ], 'g', tt,
                   fluxes[ 2 ][ 3 ], 'k', tt, fluxes[ 2 ][ 4 ], 'y')
    plt.show()
Figure_No_NaK = 0
if Figure_No_NaK:
    S_0 = (
        ci_na_init, ci_k_init, ci_cl_init, ci_hco3_init, ci_gluc_init, ce_na_init, ce_k_init, ce_cl_init, ce_hco3_init,
        ce_gluc_init, clvl_init, chvl0_init, vi_init, ve_init, p_i_init, pe_init)
    tt = np.linspace(0, tf, T_F)
    Param =[ 1, 1, 1, 0, 1, 1 ]
    sol = odeint(dSdt, y0=S_0, t=tt, tfirst=True, args=(solver, Param))
    ci_na = sol.T[ 0 ]
    ci_k = sol.T[ 1 ]
    ci_cl = sol.T[ 2 ]
    ci_hco3 = sol.T[ 3 ]
    ci_gluc = sol.T[ 4 ]
    ce_na = sol.T[ 5 ]
    ce_k = sol.T[ 6 ]
    ce_cl = sol.T[ 7 ]
    ce_hco3 = sol.T[ 8 ]
    ce_gluc = sol.T[ 9 ]
    clvl = sol.T[ 10 ]
    chvl = sol.T[ 11 ]
    vi = sol.T[ 12 ]
    ve = sol.T[ 13 ]
    p_i = sol.T[ 14 ]
    pe = sol.T[ 15 ]
    S =[ ci_na, ci_k, ci_cl, ci_hco3, ci_gluc, ce_na, ce_k, ce_cl, ce_hco3, ce_gluc, clvl, chvl, vi, ve, p_i, pe ]
    Results = dSdt(tt, S, solver, Param)
    solver = 0
    fluxes = dSdt(tt, S, solver, Param)
    na_nagluc_mi, gluc_nagluc_mi = fluxes[ 0 ]
    cl_clhco3_mi, hco3_clhco3_mi = fluxes[ 1 ]
    na_active_transport_mi, cl_active_transport_mi, k_active_transport_mi, hco3_active_transport_mi, gluc_active_transport_mi = \
        fluxes[ 2 ]
    fikm_c_na, fikm_c_cl, fikm_c_k, fikm_c_hco3, fikm_c_gluc = fluxes[ 3 ]
    fikm_na, fikm_cl, fikm_k, fikm_hco3, fikm_gluc = fluxes[ 4 ]
    fivm = fluxes[ 5 ]
    k_kcl_is, cl_kcl_is = fluxes[ 6 ]
    na_nak_is, k_nak_is = fluxes[ 7 ]
    na_nacl_hco3_is, cl_nacl_hco3_is, hco3_nacl_hco3_is = fluxes[ 8 ]
    na_nahco3_is, hco3_nahco3_is = fluxes[ 9 ]
    na_active_transport_is, cl_active_transport_is, k_active_transport_is, hco3_active_transport_is, gluc_active_transport_is = \
        fluxes[ 10 ]
    fiks_c_na, fiks_c_cl, fiks_c_k, fiks_c_hco3, fiks_c_gluc = fluxes[ 11 ]
    fiks_na, fiks_cl, fiks_k, fiks_hco3, fiks_gluc = fluxes[ 12 ]
    fivs = fluxes[ 13 ]

    fike_c_na, fike_c_cl, fike_c_k, fike_c_hco3, fike_c_gluc = fluxes[ 14 ]
    fike_na, fike_cl, fike_k, fike_hco3, fike_gluc = fluxes[ 15 ]
    five = fluxes[ 16 ]

    fekm_c_na, fekm_c_cl, fekm_c_k, fekm_c_hco3, fekm_c_gluc = fluxes[ 17 ]
    fekm_na, fekm_cl, fekm_k, fekm_hco3, fekm_gluc = fluxes[ 18 ]
    fevm = fluxes[ 19 ]

    feks_c_na, feks_c_cl, feks_c_k, feks_c_hco3, feks_c_gluc = fluxes[ 20 ]
    feks_na, feks_cl, feks_k, feks_hco3, feks_gluc = fluxes[ 21 ]
    fevs = fluxes[ 22 ]

    No_NaK = {'na':[ fekm_na[ -1 ], fikm_na[ -1 ], fiks_na[ -1 ], fike_na[ -1 ], feks_na[ -1 ] ],
               'k':[ fekm_k[ -1 ], fikm_k[ -1 ], fiks_k[ -1 ], fike_k[ -1 ], feks_k[ -1 ] ],
               'cl':[ fekm_cl[ -1 ], fikm_cl[ -1 ], fiks_cl[ -1 ], fike_cl[ -1 ], feks_cl[ -1 ] ],
               'gluc':[ fekm_gluc[ -1 ], fikm_gluc[ -1 ], fiks_gluc[ -1 ], fike_gluc[ -1 ],
                         feks_gluc[ -1 ] ],
               'fekm':[ fekm_na[ -1 ], fekm_k[ -1 ], fekm_cl[ -1 ], fekm_gluc[ -1 ] ],
               'fikm':[ fikm_na[ -1 ], fikm_k[ -1 ], fikm_cl[ -1 ], fikm_gluc[ -1 ] ],
               'fiks':[ fiks_na[ -1 ], fiks_k[ -1 ], fiks_cl[ -1 ], fiks_gluc[ -1 ] ],
               'fike':[ fike_na[ -1 ], fike_k[ -1 ], fike_cl[ -1 ], fike_gluc[ -1 ] ],
               'feks':[ feks_na[ -1 ], feks_k[ -1 ], feks_cl[ -1 ], feks_gluc[ -1 ] ]}
    flux_na = feks_na[ -1 ] + fike_na[ -1 ] + fiks_na[ -1 ] + fekm_na[ -1 ] + fikm_na[ -1 ]
    slt_con_No_NaK = {'Na': ci_na[ -1 ], 'K': ci_k[ -1 ], 'Cl': ci_cl[ -1 ], 'Gluc': ci_gluc[ -1 ]}
    flx_na_mem_No_NaK = {'feks_na': feks_na[ -1 ], 'fike_na': fike_na[ -1 ], 'fiks_na': fiks_na[ -1 ],
                          'fekm_na': fekm_na[ -1 ], 'fikm_na': fikm_na[ -1 ]}
    flx_Epithelial_na = fekm_na[ -1 ] + fikm_na[ -1 ]
    flx_Epithelial_Convective_na = fekm_c_na[ -1 ] + fikm_c_na[ -1 ]
    flx_Epithelial_Passive_na = flx_Epithelial_na - flx_Epithelial_Convective_na
    epithelial_flx_variation_na_No_NaK = {'flx_Epithelial_na': flx_Epithelial_na,
                                           'flx_Epithelial_Convective_na': flx_Epithelial_Convective_na,
                                           'flx_Epithelial_Passive_na': flx_Epithelial_Passive_na,
                                           'na_nagluc_mi': na_nagluc_mi}

    from collections import defaultdict

    Figure_No_NaK = defaultdict(list)
    for d in (No_NaK, slt_con_No_NaK, flx_na_mem_No_NaK, epithelial_flx_variation_na_No_NaK):
        for key, value in d.items():
            Figure_No_NaK[ key ].append(value)

    pickled_list = pickle.dumps(Figure_No_NaK)
    of = open('MI_Receptors_No_NaK.py', 'wb')
    of.write(pickled_list)
    of.close()
    fig1 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig1)
    ax0 = fig1.add_subplot(spec2[ 0, 0 ])
    ax0.set_title('Cellular Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ci_na, 'g', linewidth=2, label='Na_i')
    ax0.set_ylabel('ci_na', color='blue')

    ax1 = fig1.add_subplot(spec2[ 1, 0 ])
    ax1.plot(tt, ci_cl, 'k', linewidth=2, label='Cl_i')
    ax1.set_ylabel('ci_cl', color='blue')

    ax2 = fig1.add_subplot(spec2[ 2, 0 ])
    ax2.plot(tt, ci_k, 'r', linewidth=2, label='K_i')
    ax2.set_ylabel('ci_k ', color='blue')

    ax3 = fig1.add_subplot(spec2[ 3, 0 ])
    ax3.plot(tt, ci_hco3, 'b', linewidth=2, label='hco3_i')
    ax3.set_ylabel('ci_hco3 ', color='blue')

    ax4 = fig1.add_subplot(spec2[ 4, 0 ])
    ax4.plot(tt, ci_gluc, 'm', linewidth=2, label='gluc_i')
    ax4.set_ylabel('ci_gluc', color='blue')

    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig2)
    ax0 = fig2.add_subplot(spec2[ 0, 0 ])
    ax0.set_title('Interspace Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ce_na, 'g', linewidth=2, label='Na_e')
    ax0.set_ylabel('ce_na', color='blue')

    ax1 = fig2.add_subplot(spec2[ 1, 0 ])
    ax1.plot(tt, ce_cl, 'k', linewidth=2, label='Cl_e')
    ax1.set_ylabel('ce_cl', color='blue')

    ax2 = fig2.add_subplot(spec2[ 2, 0 ])
    ax2.plot(tt, ce_k, 'r', linewidth=2, label='K_e')
    ax2.set_ylabel('ce_k', color='blue')

    ax3 = fig2.add_subplot(spec2[ 3, 0 ])
    ax3.plot(tt, ce_hco3, 'b', linewidth=2, label='hco3_e')
    ax3.set_ylabel('ce_hco3', color='blue')
    ax4 = fig2.add_subplot(spec2[ 4, 0 ])
    ax4.plot(tt, ce_gluc, 'm', linewidth=2, label='gluc_e')
    ax4.set_ylabel('ce_gluc', color='blue')
    fig4 = plt.figure(constrained_layout=False, figsize=(10, 10))
    spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig4)
    ax1 = fig4.add_subplot(spec2[ 0, 0 ])
    ax1.set_title('Electrochemical Fluxes', color='#868686')

    plt.setp(plt.gca(), xticklabels=[ ])

    ax2 = fig4.add_subplot(spec2[ 1, 0 ])
    ax2.set_ylabel('flux_clhco3_mi[M/sec]', color='#868686')
    x2 = ax2.plot(tt, fluxes[ 1 ][ 0 ], 'g')

    ax2 = fig4.add_subplot(spec2[ 2, 0 ])
    ax2.set_ylabel('flux_k_cl_is[M/sec]', color='#868686')
    ax2 = ax2.plot(tt, fluxes[ 2 ][ 0 ], 'b', tt, fluxes[ 2 ][ 1 ], 'r', tt, fluxes[ 2 ][ 2 ], 'g', tt,
                   fluxes[ 2 ][ 3 ], 'k', tt, fluxes[ 2 ][ 4 ], 'y')
    plt.show()

Figure_No_KCl = 0
if Figure_No_KCl:
    S_0 = (
        ci_na_init, ci_k_init, ci_cl_init, ci_hco3_init, ci_gluc_init, ce_na_init, ce_k_init, ce_cl_init,
        ce_hco3_init,
        ce_gluc_init, clvl_init, chvl0_init, vi_init, ve_init, p_i_init, pe_init)
    tt = np.linspace(0, tf, T_F)
    Param =[ 1, 1, 1, 1, 0, 1 ]
    sol = odeint(dSdt, y0=S_0, t=tt, tfirst=True, args=(solver, Param))
    ci_na = sol.T[ 0 ]
    ci_k = sol.T[ 1 ]
    ci_cl = sol.T[ 2 ]
    ci_hco3 = sol.T[ 3 ]
    ci_gluc = sol.T[ 4 ]
    ce_na = sol.T[ 5 ]
    ce_k = sol.T[ 6 ]
    ce_cl = sol.T[ 7 ]
    ce_hco3 = sol.T[ 8 ]
    ce_gluc = sol.T[ 9 ]
    clvl = sol.T[ 10 ]
    chvl = sol.T[ 11 ]

    vi = sol.T[ 12 ]
    ve = sol.T[ 13 ]
    p_i = sol.T[ 14 ]
    pe = sol.T[ 15 ]
    S =[ ci_na, ci_k, ci_cl, ci_hco3, ci_gluc, ce_na, ce_k, ce_cl, ce_hco3, ce_gluc, clvl, chvl, vi, ve, p_i, pe ]
    Results = dSdt(tt, S, solver, Param)
    solver = 0
    fluxes = dSdt(tt, S, solver, Param)

    na_nagluc_mi, gluc_nagluc_mi = fluxes[ 0 ]
    cl_clhco3_mi, hco3_clhco3_mi = fluxes[ 1 ]
    na_active_transport_mi, cl_active_transport_mi, k_active_transport_mi, hco3_active_transport_mi, gluc_active_transport_mi = \
        fluxes[ 2 ]
    fikm_c_na, fikm_c_cl, fikm_c_k, fikm_c_hco3, fikm_c_gluc = fluxes[ 3 ]
    fikm_na, fikm_cl, fikm_k, fikm_hco3, fikm_gluc = fluxes[ 4 ]
    fivm = fluxes[ 5 ]
    k_kcl_is, cl_kcl_is = fluxes[ 6 ]
    na_nak_is, k_nak_is = fluxes[ 7 ]
    na_nacl_hco3_is, cl_nacl_hco3_is, hco3_nacl_hco3_is = fluxes[ 8 ]
    na_nahco3_is, hco3_nahco3_is = fluxes[ 9 ]
    na_active_transport_is, cl_active_transport_is, k_active_transport_is, hco3_active_transport_is, gluc_active_transport_is = \
        fluxes[ 10 ]
    fiks_c_na, fiks_c_cl, fiks_c_k, fiks_c_hco3, fiks_c_gluc = fluxes[ 11 ]
    fiks_na, fiks_cl, fiks_k, fiks_hco3, fiks_gluc = fluxes[ 12 ]
    fivs = fluxes[ 13 ]

    fike_c_na, fike_c_cl, fike_c_k, fike_c_hco3, fike_c_gluc = fluxes[ 14 ]
    fike_na, fike_cl, fike_k, fike_hco3, fike_gluc = fluxes[ 15 ]
    five = fluxes[ 16 ]

    fekm_c_na, fekm_c_cl, fekm_c_k, fekm_c_hco3, fekm_c_gluc = fluxes[ 17 ]
    fekm_na, fekm_cl, fekm_k, fekm_hco3, fekm_gluc = fluxes[ 18 ]
    fevm = fluxes[ 19 ]

    feks_c_na, feks_c_cl, feks_c_k, feks_c_hco3, feks_c_gluc = fluxes[ 20 ]
    feks_na, feks_cl, feks_k, feks_hco3, feks_gluc = fluxes[ 21 ]
    fevs = fluxes[ 22 ]

    No_KCl = {'na':[ fekm_na[ -1 ], fikm_na[ -1 ], fiks_na[ -1 ], fike_na[ -1 ], feks_na[ -1 ] ],
                       'k':[ fekm_k[ -1 ], fikm_k[ -1 ], fiks_k[ -1 ], fike_k[ -1 ], feks_k[ -1 ] ],
                       'cl':[ fekm_cl[ -1 ], fikm_cl[ -1 ], fiks_cl[ -1 ], fike_cl[ -1 ], feks_cl[ -1 ] ],
                       'gluc':[ fekm_gluc[ -1 ], fikm_gluc[ -1 ], fiks_gluc[ -1 ], fike_gluc[ -1 ],
                                 feks_gluc[ -1 ] ],
                       'fekm':[ fekm_na[ -1 ], fekm_k[ -1 ], fekm_cl[ -1 ], fekm_gluc[ -1 ] ],
                       'fikm':[ fikm_na[ -1 ], fikm_k[ -1 ], fikm_cl[ -1 ], fikm_gluc[ -1 ] ],
                       'fiks':[ fiks_na[ -1 ], fiks_k[ -1 ], fiks_cl[ -1 ], fiks_gluc[ -1 ] ],
                       'fike':[ fike_na[ -1 ], fike_k[ -1 ], fike_cl[ -1 ], fike_gluc[ -1 ] ],
                       'feks':[ feks_na[ -1 ], feks_k[ -1 ], feks_cl[ -1 ], feks_gluc[ -1 ] ]}
    flux_na = feks_na[ -1 ] + fike_na[ -1 ] + fiks_na[ -1 ] + fekm_na[ -1 ] + fikm_na[ -1 ]
    slt_con_No_KCl = {'Na': ci_na[ -1 ], 'K': ci_k[ -1 ], 'Cl': ci_cl[ -1 ], 'Gluc': ci_gluc[ -1 ]}
    flx_na_mem_No_KCl = {'feks_na': feks_na[ -1], 'fike_na': fike_na[ -1 ], 'fiks_na': fiks_na[ -1 ],
                          'fekm_na': fekm_na[ -1 ], 'fikm_na': fikm_na[ -1 ]}
    flx_Epithelial_na = fekm_na[ -1 ] + fikm_na[ -1 ]
    flx_Epithelial_Convective_na = fekm_c_na[ -1 ] + fikm_c_na[ -1 ]
    flx_Epithelial_Passive_na = flx_Epithelial_na - flx_Epithelial_Convective_na
    epithelial_flx_variation_na_No_KCl = {'flx_Epithelial_na': flx_Epithelial_na,
                                           'flx_Epithelial_Convective_na': flx_Epithelial_Convective_na,
                                           'flx_Epithelial_Passive_na': flx_Epithelial_Passive_na,
                                           'na_nagluc_mi': na_nagluc_mi}

    from collections import defaultdict

    Figure_No_KCl = defaultdict(list)
    for d in (No_KCl, slt_con_No_KCl, flx_na_mem_No_KCl, epithelial_flx_variation_na_No_KCl):
        for key, value in d.items():
            Figure_No_KCl[key].append(value)

    pickled_list = pickle.dumps(Figure_No_KCl)
    of = open('MI_Receptors_No_KCl.py', 'wb')
    of.write(pickled_list)
    of.close()
    fig1 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig1)
    ax0 = fig1.add_subplot(spec2[ 0, 0 ])
    ax0.set_title('Cellular Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ci_na, 'g', linewidth=2, label='Na_i')
    ax0.set_ylabel('ci_na', color='blue')

    ax1 = fig1.add_subplot(spec2[ 1, 0 ])
    ax1.plot(tt, ci_cl, 'k', linewidth=2, label='Cl_i')
    ax1.set_ylabel('ci_cl', color='blue')

    ax2 = fig1.add_subplot(spec2[ 2, 0 ])
    ax2.plot(tt, ci_k, 'r', linewidth=2, label='K_i')
    ax2.set_ylabel('ci_k ', color='blue')

    ax3 = fig1.add_subplot(spec2[ 3, 0 ])
    ax3.plot(tt, ci_hco3, 'b', linewidth=2, label='hco3_i')
    ax3.set_ylabel('ci_hco3 ', color='blue')

    ax4 = fig1.add_subplot(spec2[ 4, 0 ])
    ax4.plot(tt, ci_gluc, 'm', linewidth=2, label='gluc_i')
    ax4.set_ylabel('ci_gluc', color='blue')

    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig2)
    ax0 = fig2.add_subplot(spec2[ 0, 0 ])
    ax0.set_title('Interspace Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ce_na, 'g', linewidth=2, label='Na_e')
    ax0.set_ylabel('ce_na', color='blue')

    ax1 = fig2.add_subplot(spec2[ 1, 0 ])
    ax1.plot(tt, ce_cl, 'k', linewidth=2, label='Cl_e')
    ax1.set_ylabel('ce_cl', color='blue')

    ax2 = fig2.add_subplot(spec2[ 2, 0 ])
    ax2.plot(tt, ce_k, 'r', linewidth=2, label='K_e')
    ax2.set_ylabel('ce_k', color='blue')

    ax3 = fig2.add_subplot(spec2[ 3, 0 ])
    ax3.plot(tt, ce_hco3, 'b', linewidth=2, label='hco3_e')
    ax3.set_ylabel('ce_hco3', color='blue')
    ax4 = fig2.add_subplot(spec2[ 4, 0 ])
    ax4.plot(tt, ce_gluc, 'm', linewidth=2, label='gluc_e')
    ax4.set_ylabel('ce_gluc', color='blue')
    fig4 = plt.figure(constrained_layout=False, figsize=(10, 10))
    spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig4)
    ax1 = fig4.add_subplot(spec2[ 0, 0 ])
    ax1.set_title('Electrochemical Fluxes', color='#868686')

    plt.setp(plt.gca(), xticklabels=[ ])

    ax2 = fig4.add_subplot(spec2[ 1, 0 ])
    ax2.set_ylabel('flux_clhco3_mi[M/sec]', color='#868686')
    x2 = ax2.plot(tt, fluxes[ 1 ][ 0 ], 'g')

    ax2 = fig4.add_subplot(spec2[ 2, 0 ])
    ax2.set_ylabel('flux_k_cl_is[M/sec]', color='#868686')
    ax2 = ax2.plot(tt, fluxes[ 2 ][ 0 ], 'b', tt, fluxes[ 2 ][ 1 ], 'r', tt, fluxes[ 2 ][ 2 ], 'g', tt,
                   fluxes[ 2 ][ 3 ], 'k', tt, fluxes[ 2 ][ 4 ], 'y')
    plt.show()
Figure_No_Na_HCO3 = 0
if Figure_No_Na_HCO3:
    S_0 = (
        ci_na_init, ci_k_init, ci_cl_init, ci_hco3_init, ci_gluc_init, ce_na_init, ce_k_init, ce_cl_init,
        ce_hco3_init,
        ce_gluc_init, clvl_init, chvl0_init, vi_init, ve_init, p_i_init, pe_init)
    tt = np.linspace(0, tf, T_F)
    Param =[ 1, 1, 1, 1, 1, 0 ]
    sol = odeint(dSdt, y0=S_0, t=tt, tfirst=True, args=(solver, Param))
    ci_na = sol.T[ 0 ]
    ci_k = sol.T[ 1 ]
    ci_cl = sol.T[ 2 ]
    ci_hco3 = sol.T[ 3 ]
    ci_gluc = sol.T[ 4 ]
    ce_na = sol.T[ 5 ]
    ce_k = sol.T[ 6 ]
    ce_cl = sol.T[ 7 ]
    ce_hco3 = sol.T[ 8 ]
    ce_gluc = sol.T[ 9 ]
    clvl = sol.T[ 10 ]
    chvl = sol.T[ 11 ]
    vi = sol.T[ 12 ]
    ve = sol.T[ 13 ]
    p_i = sol.T[ 14 ]
    pe = sol.T[ 15 ]
    S =[ ci_na, ci_k, ci_cl, ci_hco3, ci_gluc, ce_na, ce_k, ce_cl, ce_hco3, ce_gluc, clvl, chvl, vi, ve, p_i, pe ]
    Results = dSdt(tt, S, solver, Param)
    solver = 0
    fluxes = dSdt(tt, S, solver, Param)

    na_nagluc_mi, gluc_nagluc_mi = fluxes[ 0 ]
    cl_clhco3_mi, hco3_clhco3_mi = fluxes[ 1 ]
    na_active_transport_mi, cl_active_transport_mi, k_active_transport_mi, hco3_active_transport_mi, gluc_active_transport_mi = \
        fluxes[ 2 ]
    fikm_c_na, fikm_c_cl, fikm_c_k, fikm_c_hco3, fikm_c_gluc = fluxes[ 3 ]
    fikm_na, fikm_cl, fikm_k, fikm_hco3, fikm_gluc = fluxes[ 4 ]
    fivm = fluxes[ 5 ]
    k_kcl_is, cl_kcl_is = fluxes[ 6 ]
    na_nak_is, k_nak_is = fluxes[ 7 ]
    na_nacl_hco3_is, cl_nacl_hco3_is, hco3_nacl_hco3_is = fluxes[ 8 ]
    na_nahco3_is, hco3_nahco3_is = fluxes[ 9 ]
    na_active_transport_is, cl_active_transport_is, k_active_transport_is, hco3_active_transport_is, gluc_active_transport_is = \
        fluxes[ 10 ]
    fiks_c_na, fiks_c_cl, fiks_c_k, fiks_c_hco3, fiks_c_gluc = fluxes[ 11 ]
    fiks_na, fiks_cl, fiks_k, fiks_hco3, fiks_gluc = fluxes[ 12 ]
    fivs = fluxes[ 13 ]

    fike_c_na, fike_c_cl, fike_c_k, fike_c_hco3, fike_c_gluc = fluxes[ 14 ]
    fike_na, fike_cl, fike_k, fike_hco3, fike_gluc = fluxes[ 15 ]
    five = fluxes[ 16 ]

    fekm_c_na, fekm_c_cl, fekm_c_k, fekm_c_hco3, fekm_c_gluc = fluxes[ 17 ]
    fekm_na, fekm_cl, fekm_k, fekm_hco3, fekm_gluc = fluxes[ 18 ]
    fevm = fluxes[ 19 ]

    feks_c_na, feks_c_cl, feks_c_k, feks_c_hco3, feks_c_gluc = fluxes[ 20 ]
    feks_na, feks_cl, feks_k, feks_hco3, feks_gluc = fluxes[ 21 ]
    fevs = fluxes[ 22 ]

    No_na_hco3 = {'na':[ fekm_na[ -1 ], fikm_na[ -1 ], fiks_na[ -1 ], fike_na[ -1 ], feks_na[ -1 ] ],
                       'k':[ fekm_k[ -1 ], fikm_k[ -1 ], fiks_k[ -1 ], fike_k[ -1 ], feks_k[ -1 ] ],
                       'cl':[ fekm_cl[ -1 ], fikm_cl[ -1 ], fiks_cl[ -1 ], fike_cl[ -1 ], feks_cl[ -1 ] ],
                       'gluc':[ fekm_gluc[ -1 ], fikm_gluc[ -1 ], fiks_gluc[ -1 ], fike_gluc[ -1 ],
                                 feks_gluc[ -1 ] ],
                       'fekm':[ fekm_na[ -1 ], fekm_k[ -1 ], fekm_cl[ -1 ], fekm_gluc[ -1 ] ],
                       'fikm':[ fikm_na[ -1 ], fikm_k[ -1 ], fikm_cl[ -1 ], fikm_gluc[ -1 ] ],
                       'fiks':[ fiks_na[ -1 ], fiks_k[ -1 ], fiks_cl[ -1 ], fiks_gluc[ -1 ] ],
                       'fike':[ fike_na[ -1 ], fike_k[ -1 ], fike_cl[ -1 ], fike_gluc[ -1 ] ],
                       'feks':[ feks_na[ -1 ], feks_k[ -1 ], feks_cl[ -1 ], feks_gluc[ -1 ] ]}
    flux_naNo_na_hco3 = feks_na[ -1 ] + fike_na[ -1 ] + fiks_na[ -1 ] + fekm_na[ -1 ] + fikm_na[ -1 ]
    slt_con_No_na_hco3 = {'Na': ci_na[ -1 ], 'K': ci_k[ -1 ], 'Cl': ci_cl[ -1 ], 'Gluc': ci_gluc[ -1 ]}
    flx_na_mem_No_na_hco3 = {'feks_na': feks_na[ -1 ], 'fike_na': fike_na[ -1 ], 'fiks_na': fiks_na[ -1 ],
                          'fekm_na': fekm_na[ -1 ], 'fikm_na': fikm_na[ -1 ]}
    flx_Epithelial_na = fekm_na[ -1 ] + fikm_na[ -1 ]
    flx_Epithelial_Convective_na = fekm_c_na[ -1 ] + fikm_c_na[ -1 ]
    flx_Epithelial_Passive_na = flx_Epithelial_na - flx_Epithelial_Convective_na
    epithelial_flx_variation_na_No_na_hco3 = {'flx_Epithelial_na': flx_Epithelial_na,
                                           'flx_Epithelial_Convective_na': flx_Epithelial_Convective_na,
                                           'flx_Epithelial_Passive_na': flx_Epithelial_Passive_na,
                                           'na_nagluc_mi': na_nagluc_mi}

    from collections import defaultdict

    Figure5_No_na_hco3 = defaultdict(list)
    for d in (No_na_hco3 , slt_con_No_na_hco3 , flx_na_mem_No_na_hco3 , epithelial_flx_variation_na_No_na_hco3 ):
        for key, value in d.items():
            Figure5_No_na_hco3[ key ].append(value)

    pickled_list = pickle.dumps(Figure5_No_na_hco3)
    of = open('MI_Receptors_No_Na_HCO3.py', 'wb')
    of.write(pickled_list)
    of.close()
    fig1 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig1)
    ax0 = fig1.add_subplot(spec2[ 0, 0 ])
    ax0.set_title('Cellular Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ci_na, 'g', linewidth=2, label='Na_i')
    ax0.set_ylabel('ci_na', color='blue')

    ax1 = fig1.add_subplot(spec2[ 1, 0 ])
    ax1.plot(tt, ci_cl, 'k', linewidth=2, label='Cl_i')
    ax1.set_ylabel('ci_cl', color='blue')

    ax2 = fig1.add_subplot(spec2[ 2, 0 ])
    ax2.plot(tt, ci_k, 'r', linewidth=2, label='K_i')
    ax2.set_ylabel('ci_k ', color='blue')

    ax3 = fig1.add_subplot(spec2[ 3, 0 ])
    ax3.plot(tt, ci_hco3, 'b', linewidth=2, label='hco3_i')
    ax3.set_ylabel('ci_hco3 ', color='blue')

    ax4 = fig1.add_subplot(spec2[ 4, 0 ])
    ax4.plot(tt, ci_gluc, 'm', linewidth=2, label='gluc_i')
    ax4.set_ylabel('ci_gluc', color='blue')

    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=5, figure=fig2)
    ax0 = fig2.add_subplot(spec2[ 0, 0 ])
    ax0.set_title('Interspace Concentration[mmol/l]', color='#868686')
    ax0.plot(tt, ce_na, 'g', linewidth=2, label='Na_e')
    ax0.set_ylabel('ce_na', color='blue')

    ax1 = fig2.add_subplot(spec2[ 1, 0 ])
    ax1.plot(tt, ce_cl, 'k', linewidth=2, label='Cl_e')
    ax1.set_ylabel('ce_cl', color='blue')

    ax2 = fig2.add_subplot(spec2[ 2, 0 ])
    ax2.plot(tt, ce_k, 'r', linewidth=2, label='K_e')
    ax2.set_ylabel('ce_k', color='blue')

    ax3 = fig2.add_subplot(spec2[ 3, 0 ])
    ax3.plot(tt, ce_hco3, 'b', linewidth=2, label='hco3_e')
    ax3.set_ylabel('ce_hco3', color='blue')
    ax4 = fig2.add_subplot(spec2[ 4, 0 ])
    ax4.plot(tt, ce_gluc, 'm', linewidth=2, label='gluc_e')
    ax4.set_ylabel('ce_gluc', color='blue')
    fig4 = plt.figure(constrained_layout=False, figsize=(10, 10))
    spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig4)
    ax1 = fig4.add_subplot(spec2[ 0, 0 ])
    ax1.set_title('Electrochemical Fluxes', color='#868686')

    plt.setp(plt.gca(), xticklabels=[ ])

    ax2 = fig4.add_subplot(spec2[ 1, 0 ])
    ax2.set_ylabel('flux_clhco3_mi[M/sec]', color='#868686')
    x2 = ax2.plot(tt, fluxes[ 1 ][ 0 ], 'g')

    ax2 = fig4.add_subplot(spec2[ 2, 0 ])
    ax2.set_ylabel('flux_k_cl_is[M/sec]', color='#868686')
    ax2 = ax2.plot(tt, fluxes[ 2 ][ 0 ], 'b', tt, fluxes[ 2 ][ 1 ], 'r', tt, fluxes[ 2 ][ 2 ], 'g', tt,
                   fluxes[ 2 ][ 3 ], 'k', tt, fluxes[ 2 ][ 4 ], 'y')
    plt.show()


Figure_6A = 0
Figure_6B = 0
Figure_8A = 0
Figure_8B = 1

if Figure_6A:
    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_Default.py', 'rb')
    read_file = f.read()
    my_loaded_list0 = pickle.loads(read_file)
    f.close()


    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_No_NaK.py', 'rb')
    read_file = f.read()
    my_loaded_list1 = pickle.loads(read_file)
    f.close()



    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_No_KCl.py', 'rb')
    read_file = f.read()
    my_loaded_list2 = pickle.loads(read_file)
    f.close()
    scale_factor = 1e9




    flux_me = (np.array(my_loaded_list0[ 'fekm']),np.array(my_loaded_list1['fekm']),np.array(my_loaded_list2['fekm']))
    flux_me = tuple([x * scale_factor for x in flux_me])
    print(type(flux_me))

    flux_me = np.reshape(flux_me, (1,12))
    flux_me = np.reshape(flux_me, (3,4))


    flux_mi = (np.array(my_loaded_list0[ 'fikm' ]),np.array(my_loaded_list1[ 'fikm' ]),np.array(my_loaded_list2[ 'fikm' ]))
    flux_mi = tuple([ x * scale_factor for x in flux_mi ])
    flux_mi = np.reshape(flux_mi, (1,12))
    flux_mi = np.reshape(flux_mi, (3,4))



    flux_is = (np.array(my_loaded_list0[ 'fiks' ]),np.array(my_loaded_list1[ 'fiks' ]),np.array(my_loaded_list2[ 'fiks' ]))
    flux_is = tuple([ x * scale_factor for x in flux_is ])
    flux_is = np.reshape(flux_is, (1,12))
    flux_is = np.reshape(flux_is, (3,4))


    flux_ie = (np.array(my_loaded_list0[ 'fike' ]),np.array(my_loaded_list1[ 'fike' ]),np.array(my_loaded_list2[ 'fike' ]))

    flux_ie = tuple([ x * scale_factor for x in flux_ie ])
    flux_ie = np.reshape(flux_ie, (1,12))
    flux_ie = np.reshape(flux_ie, (3,4))


    flux_es = (np.array(my_loaded_list0[ 'feks' ]), np.array(my_loaded_list1[ 'feks' ]), np.array(my_loaded_list1[ 'feks' ]))
    print('flux_es', flux_es)
    flux_es = tuple([ x * scale_factor for x in flux_es ])
    flux_es = np.reshape(flux_es, (1,12))
    flux_es = np.reshape(flux_es, (3,4))


    print('flux_es', flux_es)
    df1 = pd.DataFrame(flux_me, index=["(a) Original", "(b) NaK = 0", " (c) KCl = 0"],
                       columns=[ "Na", "K", "Cl", "Gluc" ])
    df2 = pd.DataFrame(flux_is, index=["(a) Original", "(b) NaK = 0", " (c) KCl = 0"],
                       columns=[ "Na", "K", "Cl", "Gluc" ])
    df3 = pd.DataFrame(flux_ie, index=["(a) Original", "(b) NaK = 0", " (c) KCl = 0"],
                       columns=[ "Na", "K", "Cl", "Gluc" ])
    df4 = pd.DataFrame(flux_es, index=["(a) Original", "(b) NaK = 0", " (c) KCl = 0"],
                       columns=[ "Na", "K", "Cl", "Gluc" ])
    df5 = pd.DataFrame( flux_mi, index=["(a) Original", "(b) NaK = 0", " (c) KCl = 0"],
                       columns=[ "Na", "K", "Cl", "Gluc" ])
    def prep_df(df, name):
        df = df.stack().reset_index()
        df.columns =[ 'c1', 'c2', 'values' ]
        df[ 'Fluxes' ] = name
        return df
    df1 = prep_df(df1, 'ME')
    df2 = prep_df(df2, 'IS')
    df3 = prep_df(df3, 'IE')
    df4 = prep_df(df4, 'ES')
    df5 = prep_df(df5, 'MI')
    df = pd.concat([ df1, df2, df3, df4, df5 ])

    pd.set_option('display.max_rows', 60)
    pd.set_option('display.max_columns',5)
    print(df)
    alt.renderers.enable('altair_viewer')
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('c2', title=None, sort=[ "Na", "K", "Cl", "Gluc" ]),
        alt.Y('sum(values)',
              axis=alt.Axis(
                  grid=False,
                  title="Membrane Fluxes[Pmol/s.cm^2]")),

        alt.Column('c1', title=None, sort =["(a) Original", "(b) NaK = 0", " (c) KCl = 0"]),
        alt.Color('Fluxes',
                  scale=alt.Scale(
                      range =['#6E7F5C', '#868606', '#F0E442', '#007282', '#2B9F78']
                  ),
                  )) \
        .configure_view(
    ).configure_axis(
        grid=False,
        labelFont = 'Times New Roman',  # set font style to Helvetica
        labelFontSize = 12  # set font size to 14
    )
    chart.show()

if Figure_6B:
    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_Default.py', 'rb')
    read_file = f.read()
    my_loaded_list0 = pickle.loads(read_file)
    f.close()


    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_No_NaK.py', 'rb')
    read_file = f.read()
    my_loaded_list1 = pickle.loads(read_file)
    f.close()



    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_No_KCl.py', 'rb')
    read_file = f.read()
    my_loaded_list2 = pickle.loads(read_file)
    f.close()
    scale_factor = 1e3

    Na =[ np.array(my_loaded_list0[ 'Na' ]), np.array(my_loaded_list1[ 'Na' ]),
           np.array(my_loaded_list2[ 'Na' ]) ]
    Na =[ scale_factor * x for x in Na]
    K = (
        np.array(my_loaded_list0[ 'K' ]), np.array(my_loaded_list1[ 'K' ]), np.array(my_loaded_list2[ 'K' ]))

    K =[ scale_factor * x for x in K]

    Cl = (
        np.array(my_loaded_list0[ 'Cl' ]), np.array(my_loaded_list1[ 'Cl' ]), np.array(my_loaded_list2[ 'Cl' ]))

    Cl =[ scale_factor * x for x in Cl]
    Gluc = (
        np.array(my_loaded_list0[ 'Gluc' ]), np.array(my_loaded_list1[ 'Gluc' ]),np.array(my_loaded_list2[ 'Gluc' ]))

    Gluc =[ scale_factor * x for x in Gluc]

    df = pd.DataFrame({
        'index':[ "(a) Original", "(b) NaK = 0", "(c) KCl = 0" ],
        'Na': Na,
        'K': K,
        'Cl': Cl,
        'Gluc': Gluc
    })

    # set display options to show up to 10 rows and 100 columns
    pd.set_option('display.max_rows', 4)
    pd.set_option('display.max_columns',5)
    print(df)# pprint(df)
    chart = alt.Chart(df.melt('index')).mark_bar().encode(
        alt.X('variable:N', axis=alt.Axis(title=''), sort=alt.Sort(None)),
        alt.Y('value:Q', axis=alt.Axis(title='Cellular Concentration[mmol/l]', grid=False)),
        column='index:N'
    ).configure_view(
    ).encode(
        color=alt.Color('variable:N', scale=alt.Scale(range=[ '#868686', '#d6d743', '#007282', '#2B9F78' ]))
    )
    chart.show()

if Figure_8A:
    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_Default.py', 'rb')
    read_file = f.read()
    my_loaded_list0 = pickle.loads(read_file)
    f.close()

    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_No_NaGluc.py', 'rb')
    read_file = f.read()
    my_loaded_list1 = pickle.loads(read_file)
    f.close()

    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_No_ClHCO3.py', 'rb')
    read_file = f.read()
    my_loaded_list2 = pickle.loads(read_file)
    f.close()
    scale_factor = 1e9


    flux_me = (np.array(my_loaded_list0[ 'fekm']),np.array(my_loaded_list1['fekm']),np.array(my_loaded_list2['fekm']))
    flux_me = tuple([x * scale_factor for x in flux_me])
    print(type(flux_me))

    flux_me = np.reshape(flux_me, (1,12))
    flux_me = np.reshape(flux_me, (3,4))


    flux_mi = (np.array(my_loaded_list0[ 'fikm' ]),np.array(my_loaded_list1[ 'fikm' ]),np.array(my_loaded_list2[ 'fikm' ]))
    flux_mi = tuple([ x * scale_factor for x in flux_mi ])
    flux_mi = np.reshape(flux_mi, (1,12))
    flux_mi = np.reshape(flux_mi, (3,4))



    flux_is = (np.array(my_loaded_list0[ 'fiks' ]),np.array(my_loaded_list1[ 'fiks' ]),np.array(my_loaded_list2[ 'fiks' ]))
    flux_is = tuple([ x * scale_factor for x in flux_is ])
    flux_is = np.reshape(flux_is, (1,12))
    flux_is = np.reshape(flux_is, (3,4))


    flux_ie = (np.array(my_loaded_list0[ 'fike' ]),np.array(my_loaded_list1[ 'fike' ]),np.array(my_loaded_list2[ 'fike' ]))

    flux_ie = tuple([ x * scale_factor for x in flux_ie ])
    flux_ie = np.reshape(flux_ie, (1,12))
    flux_ie = np.reshape(flux_ie, (3,4))


    flux_es = (np.array(my_loaded_list0[ 'feks' ]), np.array(my_loaded_list1[ 'feks' ]), np.array(my_loaded_list1[ 'feks' ]))
    print('flux_es', flux_es)
    flux_es = tuple([ x * scale_factor for x in flux_es ])
    flux_es = np.reshape(flux_es, (1,12))
    flux_es = np.reshape(flux_es, (3,4))


    print('flux_es', flux_es)
    df1 = pd.DataFrame(flux_me, index=["(a) Original", "(b) SGLT = 0", "(c) ClHCO3 = 0"],
                       columns=[ "Na", "K", "Cl", "Gluc" ])
    df2 = pd.DataFrame(flux_is, index=["(a) Original", "(b) SGLT = 0", "(c) ClHCO3 = 0"],
                       columns=[ "Na", "K", "Cl", "Gluc" ])
    df3 = pd.DataFrame(flux_ie, index=["(a) Original", "(b) SGLT = 0", "(c) ClHCO3 = 0"],
                       columns=[ "Na", "K", "Cl", "Gluc" ])
    df4 = pd.DataFrame(flux_es, index=["(a) Original","(b) SGLT = 0", "(c) ClHCO3 = 0"],
                       columns=[ "Na", "K", "Cl", "Gluc" ])
    df5 = pd.DataFrame( flux_mi, index=["(a) Original","(b) SGLT = 0", "(c) ClHCO3 = 0"],
                       columns=[ "Na", "K", "Cl", "Gluc" ])
    def prep_df(df, name):
        df = df.stack().reset_index()
        df.columns =[ 'c1', 'c2', 'values' ]
        df[ 'Fluxes' ] = name
        return df
    df1 = prep_df(df1, 'ME')
    df2 = prep_df(df2, 'IS')
    df3 = prep_df(df3, 'IE')
    df4 = prep_df(df4, 'ES')
    df5 = prep_df(df5, 'MI')
    df = pd.concat([ df1, df2, df3, df4, df5 ])
    pd.set_option('display.max_rows', 60)
    pd.set_option('display.max_columns', 5)
    print(df)  # pprint(df)

    alt.renderers.enable('altair_viewer')
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('c2', title=None, sort=[ "Na", "K", "Cl", "Gluc" ]),
        alt.Y('sum(values)',
              axis=alt.Axis(
                  grid=False,
                  title="Membrane Fluxes[pmol/s.cm^2]")),

        alt.Column('c1', title=None, sort =["(a) Original", "(b) SGLT = 0", "(c) ClHCO3 = 0"]),
        alt.Color('Fluxes',
                  scale=alt.Scale(
                      range =['#6E7F5C', '#868606', '#F0E442', '#007282', '#2B9F78']
                  ),
                  )) \
        .configure_view(
    ).configure_axis(
        grid = False,
        labelFont = 'Times New Roman',  # set font style to Helvetica
        labelFontSize = 12  # set font size to 14
    )
    chart.show()

if Figure_8B:
    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_Default.py', 'rb')
    read_file = f.read()
    my_loaded_list0 = pickle.loads(read_file)
    f.close()

    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_No_NaGluc.py', 'rb')
    read_file = f.read()
    my_loaded_list1 = pickle.loads(read_file)
    f.close()

    f = open('C:/Users/lnor300/Documents/BondGraph_Project/Leyla_BondGraph_W_PCT_E_Model/W_PCT_E_PYTHON/MI_Receptors_No_ClHCO3.py', 'rb')
    read_file = f.read()
    my_loaded_list2 = pickle.loads(read_file)
    f.close()
    scale_factor = 1e3



    Na =[ np.array(my_loaded_list0[ 'Na' ]), np.array(my_loaded_list1[ 'Na' ]),
           np.array(my_loaded_list2[ 'Na' ]) ]
    Na =[ scale_factor * x for x in Na]
    K = (
        np.array(my_loaded_list0[ 'K' ]), np.array(my_loaded_list1[ 'K' ]), np.array(my_loaded_list2[ 'K' ]))

    K =[ scale_factor * x for x in K]

    Cl = (
        np.array(my_loaded_list0[ 'Cl' ]), np.array(my_loaded_list1[ 'Cl' ]), np.array(my_loaded_list2[ 'Cl' ]))

    Cl =[ scale_factor * x for x in Cl]
    Gluc = (
        np.array(my_loaded_list0[ 'Gluc' ]), np.array(my_loaded_list1[ 'Gluc' ]),np.array(my_loaded_list2[ 'Gluc' ]))

    Gluc =[ scale_factor * x for x in Gluc]


    df = pd.DataFrame({
        'index':[ "(a) Original", "(b) SGLT = 0", "(c) ClHCO3 = 0"],
        'Na': Na,
        'K': K,
        'Cl': Cl,
        'Gluc': Gluc
    })
    pd.set_option('display.max_rows', 60)
    pd.set_option('display.max_columns', 5)
    print(df)
    chart = alt.Chart(df.melt('index')).mark_bar().encode(
        alt.X('variable:N', axis=alt.Axis(title=''), sort=alt.Sort(None)),
        alt.Y('value:Q', axis=alt.Axis(title='Cellular Concentration[mmol/l]', grid=False)),
        column='index:N'
    ).configure_view(
    ).encode(
        color=alt.Color('variable:N', scale=alt.Scale(range=[ '#868686', '#d6d743', '#007282', '#2B9F78' ]))
    )
    chart.show()

