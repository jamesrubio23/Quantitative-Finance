import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Sde_class import GBM, OrnsteinUhlenbeck, CIR
from scipy.stats import norm, skew, kurtosis

def run_simulation(model_type, params, drift, volatility):
    #Function that runs the class selected
    if model_type == "Geometric Brownian Motion":
        model = GBM(drift, volatility, **params)
    elif model_type == "Ornstein-Uhlenbeck":
        model = OrnsteinUhlenbeck(drift, volatility, **params)
    elif model_type == "Cox-Ingersoll-Ross":
        model = CIR(drift, volatility, **params)
    else:
        raise ValueError("Unknown model type")
    model.simulate_paths()
    return model


def plot_model_output(model):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(min(model.n_paths, 50)):
        axes[0].plot(model.t, model.store_Xts[i], alpha=0.7)
    axes[0].set_title("Simulated Paths")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Xt")

    axes[1].hist(model.final_values_Xt, bins=50, edgecolor='black')
    axes[1].set_title("Final Value Distribution")
    axes[1].set_xlabel("Xt(T)")
    axes[1].set_ylabel("Frequency")
    st.pyplot(fig)

def option_pricing(model, K, r, option_type):
    if not model.simulated_paths:
        model.simulate_paths()
    if option_type == "Call":
        payoff = np.maximum(model.final_values_Xt - K, 0)
    else:
        payoff = np.maximum(K - model.final_values_Xt, 0)
    return np.exp(-r * model.T) * np.mean(payoff)







def main():


    def display_analytics(model, S0, K, T, r, sigma):
        def black_scholes_call(S, K, T, r, sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        st.subheader("Statistical Properties of Xt(T)")
        mean = np.mean(model.final_values_Xt)
        std = np.std(model.final_values_Xt)
        skewness = skew(model.final_values_Xt)
        kurt = kurtosis(model.final_values_Xt)
        st.write(f"Mean: {mean:.4f}")
        st.write(f"Std Dev: {std:.4f}")
        st.write(f"Skewness: {skewness:.4f}")
        st.write(f"Kurtosis: {kurt:.4f}")

        if model.name == "Geometric Brownian Motion":
            st.subheader("Black-Scholes vs Monte Carlo")
            mc_price = np.exp(-r * T) * np.mean(np.maximum(model.final_values_Xt - K, 0))
            bs_price = black_scholes_call(S0, K, T, r, sigma)
            st.write(f"Monte Carlo Price: {mc_price:.4f}")
            st.write(f"Black-Scholes Price: {bs_price:.4f}")
            st.write(f"Absolute Error: {abs(mc_price - bs_price):.4f}")






    ###Model and paramaters for SDE_class
    model_type = st.sidebar.selectbox("Select Model", ["Geometric Brownian Motion", "Ornstein-Uhlenbeck", "Cox-Ingersoll-Ross"])
    X0 = st.sidebar.number_input("Initial Value (X0)", value=1.0)
    T = st.sidebar.number_input("Time Horizon (T)", value=1.0)
    dt = st.sidebar.number_input("Time Step (dt)", value=0.01)
    n_paths = st.sidebar.number_input("Number of Paths", value=1000, step=100)
    seed = st.sidebar.number_input("Random Seed", value=15, step=1)


    if "option_priced" not in st.session_state:
        st.session_state["option_priced"] = False



    st.sidebar.title("Model Configuration")


    # Parámetros de cada función
    if model_type == "Geometric Brownian Motion":
        drift_type = st.sidebar.selectbox("Drift Function", ["Constant", "Linear in t", "Quadratic in t"])
        vol_type = st.sidebar.selectbox("Volatility Function", ["Constant", "Linear in t", "Sqrt(t)", "Quadratic in t"])

        if drift_type == "Constant":
            drift_val = st.sidebar.number_input("Drift (μ)", value=0.1)
            drift = lambda t: drift_val
        elif drift_type == "Linear in t":
            slope = st.sidebar.number_input("Slope of Drift (a)", value=0.1)
            drift = lambda t: slope * t
        elif drift_type == "Quadratic in t":
            a = st.sidebar.number_input("Quadratic Drift Coef (a)", value=1.0)
            b = st.sidebar.number_input("Quadratic Drift Linear Coef (b)", value=0.0)
            c = st.sidebar.number_input("Quadratic Drift Const (c)", value=0.0)
            drift = lambda t: a*t**2 + b*t + c

        if vol_type == "Constant":
            vol_val = st.sidebar.number_input("Volatility", value=0.2)
            volatility = lambda t: vol_val
        elif vol_type == "Linear in t":
            slope_vol = st.sidebar.number_input("Slope of Volatility", value=0.1)
            volatility = lambda t: slope_vol * t
        elif vol_type == "Sqrt(t)":
            volatility = lambda t: np.sqrt(t)
        elif vol_type == "Quadratic in t":
            a = st.sidebar.number_input("Quadratic Vol Coef (a)", value=1.0)
            b = st.sidebar.number_input("Quadratic Vol Linear Coef (b)", value=0.0)
            c = st.sidebar.number_input("Quadratic Vol Const (c)", value=0.0)
            volatility = lambda t: a*t**2 + b*t + c

    elif model_type == "Ornstein-Uhlenbeck":
        st.sidebar.markdown("**OU drift = θ(μ - Xt)**")
        st.sidebar.markdown("**OU volatility = σ**")
        theta = st.sidebar.number_input("Theta", value=0.7)
        mu = st.sidebar.number_input("Mu", value=1.2)
        sigma = st.sidebar.number_input("Sigma", value=0.3)
        drift = lambda t, Xt: theta * (mu - Xt)
        volatility = lambda t, Xt: sigma

    elif model_type == "Cox-Ingersoll-Ross":
        st.sidebar.markdown("**CIR drift = θ(μ - Xt)**")
        st.sidebar.markdown("**CIR volatility = σ √Xt**")
        theta = st.sidebar.number_input("Theta", value=0.7)
        mu = st.sidebar.number_input("Mu", value=1.2)
        sigma = st.sidebar.number_input("Sigma", value=0.3)
        drift = lambda t, Xt: theta * (mu - Xt)
        volatility = lambda t, Xt: sigma * np.sqrt(max(Xt, 0))

################
###Parameters###
################
    params = {
        "X0": X0,
        "T": T,
        "dt": dt,
        "n_paths": int(n_paths),
        "seed": seed,
    }

    if model_type != "Geometric Brownian Motion":
        params.update({"theta": theta, "mu": mu, "sigma": sigma})

    if st.sidebar.button("Run Simulation"):
        model = run_simulation(model_type, params, drift, volatility)
        st.session_state["model"] = model
        st.session_state["simulation_done"] = True

    if "model" in st.session_state:
        model = st.session_state["model"]
        st.subheader(f"{model.name} Simulation Results")
        st.write(f"Mean: {model.mean():.4f}")
        st.write(f"Standard Deviation: {model.standard_dev():.4f}")
        plot_model_output(model)

    
    if "model" in st.session_state and st.session_state["model"].simulated_paths:
        st.subheader("Option Pricing")
        K = st.number_input("Strike Price (K)", value=1.0, key="K")
        r = st.number_input("Risk-Free Rate (r)", value=0.05, key="r")
        option_type = st.selectbox("Option Type", ["Call", "Put"], key="option_type")

        if st.button("Price Option"):
            price = option_pricing(st.session_state["model"], K, r, option_type)
            st.write(f"{option_type} Option Price: {price:.4f}")
            st.session_state["option_priced"] = True
    else:
        st.warning("First run a simulation to enable option pricing.")

    if st.session_state.get("option_priced", False):
        if st.button("Compare Option"):
            sigma_to_pass = vol_val if model_type == "Geometric Brownian Motion" else sigma
            display_analytics(st.session_state["model"], X0, K, T, r, sigma_to_pass)

      
    

        



if __name__ == "__main__":
    main()
