import streamlit as st
import numpy as np
import pandas as pd
import math
import altair as alt

# ---- Metrics ----
def dissimilarity_index(df, g1, g2):
    X, Y = df[g1].sum(), df[g2].sum()
    return 0.5 * np.sum(np.abs(df[g1]/X - df[g2]/Y))

def entropy_index(df, g1, g2):
    # totals
    X, Y = df[g1].sum(), df[g2].sum()
    T = X + Y
    # overall entropy
    pX, pY = X/T, Y/T
    Et = -(pX*math.log(pX,2) + pY*math.log(pY,2)) if pX>0 and pY>0 else 0
    # within-domain entropy
    Ei = []
    for _, row in df.iterrows():
        ti = row[g1] + row[g2]
        if ti == 0: continue
        piX, piY = row[g1]/ti, row[g2]/ti
        e = 0
        if piX>0: e += -piX*math.log(piX,2)
        if piY>0: e += -piY*math.log(piY,2)
        Ei.append((ti/T)*e)
    Ew = np.sum(Ei)
    return (Et - Ew)/Et if Et>0 else 0

def visit_weighted_segregation(df, g1, g2):
    rows = df.to_dict(orient="records")
    C = sum(r[g1] for r in rows)
    L = sum(r[g2] for r in rows)
    term1 = sum((r[g1]/C) * (r[g1]/(r[g1]+r[g2]))
                for r in rows if (r[g1]+r[g2])>0 and C>0)
    term2 = sum((r[g2]/L) * (r[g1]/(r[g1]+r[g2]))
                for r in rows if (r[g1]+r[g2])>0 and L>0)
    return term1 - term2

# ---- Simulation ----
def simulate_data(n_domains, g1_size, g2_size, dist):
    if dist == "Uniform":
        g1 = np.random.multinomial(g1_size, [1/n_domains]*n_domains)
        g2 = np.random.multinomial(g2_size, [1/n_domains]*n_domains)
    elif dist == "Power law":
        probs = np.array([1/(i+1) for i in range(n_domains)])
        probs = probs / probs.sum()
        g1 = np.random.multinomial(g1_size, probs)
        g2 = np.random.multinomial(g2_size, probs)
    elif dist == "Clustered":
        g1 = np.zeros(n_domains, dtype=int)
        g2 = np.zeros(n_domains, dtype=int)
        g1[0] = g1_size
        g2[-1] = g2_size
    elif dist == "Segregated Uniform":
        g1 = np.zeros(n_domains, dtype=int)
        g2 = np.zeros(n_domains, dtype=int)
        half = n_domains // 2
        g1_share = g1_size // half if half > 0 else g1_size
        g2_share = g2_size // (n_domains - half) if n_domains - half > 0 else g2_size
        g1[:half] = g1_share
        g2[half:] = g2_share
    elif dist == "Sparse Long Tail":
        # give each user their own unique domain until we run out
        g1 = np.zeros(n_domains, dtype=int)
        g2 = np.zeros(n_domains, dtype=int)
        # assign one user per domain for group 1
        for i in range(min(g1_size, n_domains)):
            g1[i] = 1
        # assign one user per domain for group 2 (continue from end)
        for j in range(min(g2_size, n_domains)):
            g2[-(j+1)] = 1
    return pd.DataFrame({"domain": range(n_domains), "Group1": g1, "Group2": g2})





# ---- Dashboard ----
st.title("Segregation Metrics Explorer")

# Inputs
n_domains = st.slider("Number of domains", 2, 10000, 6)   # now up to 10k
g1_size = st.slider("Size of Group 1", 50, 10000, 500)    # now up to 10k
g2_size = st.slider("Size of Group 2", 50, 10000, 500)    # now up to 10k
dist = st.selectbox(
    "Activity distribution",
    ["Uniform", "Power law", "Clustered", "Segregated Uniform", "Sparse Long Tail"]
)
# Simulate
df = simulate_data(n_domains, g1_size, g2_size, dist)

# Compute indices
D = dissimilarity_index(df, "Group1", "Group2")
H = entropy_index(df, "Group1", "Group2")
VWS = visit_weighted_segregation(df, "Group1", "Group2")

st.subheader("Results")
st.write(f"**Dissimilarity:** {D:.3f}")
st.write(f"**Entropy (Theil’s H):** {H:.3f}")
st.write(f"**Visit-weighted segregation (S):** {VWS:.3f}")

# ---- Altair Bar Chart ----
st.subheader("Domain Distribution")

df_melt = df.melt(id_vars="domain", value_vars=["Group1", "Group2"], 
                  var_name="Group", value_name="Count")

color_scale = alt.Scale(domain=["Group1", "Group2"],
                        range=["blue", "red"])

chart = alt.Chart(df_melt).mark_bar().encode(
    x="domain:N",
    y="Count:Q",
    color=alt.Color("Group:N", scale=color_scale)
).properties(width=600)

st.altair_chart(chart, use_container_width=True)
