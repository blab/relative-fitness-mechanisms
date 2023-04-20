from jax import jit
import jax.numpy as jnp

# Make classes here with relevant ODE models and frequency only models


@jit
def two_variant_model(u, t, θ):
    # Susceptible, Infectious wild type, Infectious variant, Recovered wild type, Recovered variant
    S, I_wt, I_v, R_wt, R_v, Inc_wt, Inc_v = (
        u[0],
        u[1],
        u[2],
        u[3],
        u[4],
        u[5],
        u[6],
    )

    # Transmissibility, recovery rate
    # eta_T is added variant transmissibility in susceptibles,
    # eta_E is escape in recovered wt hosts
    # eta_gam is the ratio between recovery rates
    beta_wt, gam_wt, eta_T, eta_E, eta_gam = (θ[0], θ[1], θ[2], θ[3], θ[4])

    lam_wt = beta_wt * S * I_wt  # Wild-type transmission in naive
    lam_vs = (beta_wt * eta_T) * S * I_v  # Variant transmission in naive
    lam_vr = (
        (beta_wt * eta_T) * (eta_E * R_wt) * I_v
    )  # Variant transmission in wt recovered

    recov_wt = gam_wt * I_wt
    recov_v = gam_wt * eta_gam * I_v

    # Defining differential equations
    dS = -lam_wt - lam_vs
    dI_wt = lam_wt - recov_wt
    dI_v = lam_vs + lam_vr - recov_v
    dR_wt = recov_wt - lam_vr
    dR_v = recov_v

    # New compartments for capturing cummulative incidence
    dIncidence_wt = lam_wt
    dIncidence_v = lam_vs + lam_vr

    return jnp.stack(
        [dS, dI_wt, dI_v, dR_wt, dR_v, dIncidence_wt, dIncidence_v]
    )


def three_variant_model(u, t, θ):
    # Susceptible, Infectious wild type, Infectious variant, Recovered wild type, Recovered variant
    S, I_wt, I_ve, I_vt, R_wt, R_ve, R_vt, Inc_wt, Inc_ve, Inc_vt = (
        u[0],
        u[1],
        u[2],
        u[3],
        u[4],
        u[5],
        u[6],
        u[7],
        u[8],
        u[9],
    )

    # Transmissibility, recovery rate
    # eta_T is added variant transmissibility in susceptibles,
    # eta_E is escape in recovered wt hosts
    beta_wt, gam_wt, eta_T, eta_E = (θ[0], θ[1], θ[2], θ[3])

    lam_wt = beta_wt * S * I_wt  # Wild-type transmission in naive
    lam_ves = beta_wt * S * I_ve  # Variant transmission in naive
    lam_vts = (beta_wt * eta_T) * S * I_vt
    # lam_ver = beta_wt * (eta_E * R_wt) * I_ve # Variant transmission in wt recovered
    lam_verw = (
        beta_wt * eta_E * R_wt * I_ve
    )  # Variant transmission in wt recovered
    lam_vert = (
        beta_wt * eta_E * R_vt * I_ve
    ) # Transmission in variant T recovered

    recov_wt = gam_wt * I_wt
    recov_ve = gam_wt * I_ve
    recov_vt = gam_wt * I_vt

    # Defining differential equations
    dS = -lam_wt - lam_ves - lam_vts

    dI_wt = lam_wt - recov_wt
    dI_ve = lam_ves + lam_verw + lam_vert - recov_ve
    dI_vt = lam_vts - recov_vt

    dR_wt = recov_wt - lam_verw
    dR_ve = recov_ve
    dR_vt = recov_vt - lam_vert

    # New compartments for capturing cummulative incidence
    dInc_wt = lam_wt
    dInc_ve = lam_ves + lam_verw + lam_vert
    dInc_vt = lam_vts

    return jnp.stack(
        [
            dS,
            dI_wt,
            dI_ve,
            dI_vt,
            dR_wt,
            dR_ve,
            dR_vt,
            dInc_wt,
            dInc_ve,
            dInc_vt,
        ]
    )
