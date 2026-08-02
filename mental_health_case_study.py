"""Single-file Mental Health in Tech 2016 case-study analysis.

Run from the repository root:
    python mental_health_case_study.py

The script performs EDA, preprocessing, PCA, model selection, K-Means
clustering, cluster profiling, and publication-ready visualization.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "mental_health_in_tech_2016.csv"
OUTPUT = ROOT / "single_file_outputs"

SELF_EMPLOYED = "Are you self-employed?"
AGE = "What is your age?"
CURRENT = "Do you currently have a mental health disorder?"
TREATMENT = "Have you ever sought treatment for a mental health issue from a mental health professional?"


def feature(alias, domain, column, mapping):
    return alias, domain, column, mapping


YN_SUPPORT = {"Yes": 0.0, "I don't know": 0.5, "No": 1.0}
YN_RISK = {"No": 0.0, "Maybe": 0.5, "Yes": 1.0}
YN_COMFORT = {"Yes": 0.0, "Maybe": 0.5, "No": 1.0}

FEATURES = [
    feature("benefits_gap", "Workplace support", "Does your employer provide mental health benefits as part of healthcare coverage?", {"Yes": 0, "I don't know": .5, "Not eligible for coverage / N/A": .75, "No": 1}),
    feature("coverage_options_unclear", "Workplace support", "Do you know the options for mental health care available under your employer-provided coverage?", {"Yes": 0, "I am not sure": .5, "No": 1}),
    feature("no_formal_discussion", "Workplace support", "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?", YN_SUPPORT),
    feature("no_resources", "Workplace support", "Does your employer offer resources to learn more about mental health concerns and options for seeking help?", YN_SUPPORT),
    feature("anonymity_uncertain", "Workplace support", "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?", YN_SUPPORT),
    feature("leave_difficulty", "Workplace support", "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:", {"Very easy": 0, "Somewhat easy": .25, "Neither easy nor difficult": .5, "I don't know": .5, "Somewhat difficult": .75, "Very difficult": 1}),
    feature("employer_consequence_fear", "Disclosure climate", "Do you think that discussing a mental health disorder with your employer would have negative consequences?", YN_RISK),
    feature("coworker_discomfort", "Disclosure climate", "Would you feel comfortable discussing a mental health disorder with your coworkers?", YN_COMFORT),
    feature("supervisor_discomfort", "Disclosure climate", "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?", YN_COMFORT),
    feature("mental_health_not_equal", "Workplace support", "Do you feel that your employer takes mental health as seriously as physical health?", YN_SUPPORT),
    feature("observed_workplace_consequence", "Disclosure climate", "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?", {"No": 0, "Yes": 1}),
    feature("career_harm_expectation", "Stigma", "Do you feel that being identified as a person with a mental health issue would hurt your career?", {"No, it has not": 0, "No, I don't think it would": 0, "Maybe": .5, "Yes, I think it would": .75, "Yes, it has": 1}),
    feature("coworker_stigma_expectation", "Stigma", "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?", {"No, they do not": 0, "No, I don't think they would": 0, "Maybe": .5, "Yes, I think they would": .75, "Yes, they do": 1}),
    feature("private_disclosure_reluctance", "Stigma", "How willing would you be to share with friends and family that you have a mental illness?", {"Very open": 0, "Somewhat open": .25, "Neutral": .5, "Somewhat not open": .75, "Not open at all": 1, "Not applicable to me (I do not have a mental illness)": 0}),
    feature("unsupportive_response", "Disclosure climate", "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?", {"No": 0, "Maybe/Not sure": .5, "Yes, I observed": .75, "Yes, I experienced": 1}),
    feature("family_history", "Personal need", "Do you have a family history of mental illness?", {"No": 0, "I don't know": .5, "Yes": 1}),
    feature("past_disorder", "Personal need", "Have you had a mental health disorder in the past?", YN_RISK),
    feature("current_disorder", "Personal need", CURRENT, YN_RISK),
    feature("professional_diagnosis", "Personal need", "Have you been diagnosed with a mental health condition by a medical professional?", {"No": 0, "Yes": 1}),
    feature("treatment_history", "Personal need", TREATMENT, {0: 0, 1: 1}),
    feature("treated_work_interference", "Work impact", "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?", {"Not applicable to me": 0, "Never": 0, "Rarely": .25, "Sometimes": .65, "Often": 1}),
    feature("untreated_work_interference", "Work impact", "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?", {"Not applicable to me": 0, "Never": 0, "Rarely": .25, "Sometimes": .65, "Often": 1}),
]


def stability_score(matrix, k):
    labels = [KMeans(n_clusters=k, n_init=20, random_state=s).fit_predict(matrix) for s in range(10)]
    return np.mean([adjusted_rand_score(labels[a], labels[b]) for a, b in combinations(range(10), 2)])


def main():
    OUTPUT.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # EDA and text normalization
    raw = pd.read_csv(DATA)
    if raw.shape != (1433, 63):
        raise ValueError(f"Expected the OSMI 2016 1433 x 63 CSV; found {raw.shape}.")
    text_columns = raw.select_dtypes(include="object").columns
    raw[text_columns] = raw[text_columns].apply(lambda col: col.str.strip())
    raw["age_clean"] = pd.to_numeric(raw[AGE], errors="coerce").where(lambda x: x.between(18, 75))

    missing = raw.drop(columns="age_clean").isna().mean().nlargest(15).sort_values()
    ax = (missing * 100).plot.barh(figsize=(10, 6), color="#4C78A8", title="Most incomplete survey questions")
    ax.set_xlabel("Missing responses (%)")
    plt.tight_layout(); plt.savefig(OUTPUT / "01_missingness.png", dpi=220); plt.close()

    status = raw[CURRENT].value_counts().reindex(["Yes", "Maybe", "No"])
    ax = status.plot.bar(figsize=(7, 4), color=["#4C78A8", "#D08C20", "#159A74"], title="Current mental health status")
    ax.set_ylabel("Participants"); ax.tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.savefig(OUTPUT / "02_current_status.png", dpi=220); plt.close()

    # Employer-focused sample and ordinal encoding (0 = lower risk, 1 = higher risk)
    employees = raw.loc[raw[SELF_EMPLOYED].eq(0)].copy().reset_index(drop=True)
    risk = pd.DataFrame(index=employees.index)
    domains = {}
    for alias, domain, column, mapping in FEATURES:
        unmapped = set(employees[column].dropna().unique()) - set(mapping)
        if unmapped:
            raise ValueError(f"Unmapped values in {alias}: {sorted(map(str, unmapped))}")
        risk[alias] = employees[column].map(mapping)
        domains[alias] = domain

    imputed = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(risk), columns=risk.columns)
    scaled = StandardScaler().fit_transform(imputed)

    # PCA retains the minimum number of components explaining at least 80% variance
    full_pca = PCA().fit(scaled)
    n_components = max(2, np.searchsorted(np.cumsum(full_pca.explained_variance_ratio_), .80) + 1)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)

    # Compare k=2...8; select stable, actionable near-best solution
    metric_rows = []
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(scores)
        counts = np.bincount(labels)
        metric_rows.append({
            "k": k,
            "silhouette": silhouette_score(scores, labels),
            "calinski_harabasz": calinski_harabasz_score(scores, labels),
            "davies_bouldin": davies_bouldin_score(scores, labels),
            "stability_ari": stability_score(scores, k),
            "smallest_cluster_share": counts.min() / counts.sum(),
        })
    metrics = pd.DataFrame(metric_rows)
    eligible = metrics.query("stability_ari >= .80 and smallest_cluster_share >= .08")
    near_best = eligible[eligible.silhouette >= eligible.silhouette.max() * .90]
    selected_k = int(near_best.sort_values("davies_bouldin").iloc[0].k)

    labels = KMeans(n_clusters=selected_k, n_init=100, random_state=42).fit_predict(scores)
    old_order = imputed.assign(cluster=labels).groupby("cluster").mean().mean(axis=1).sort_values().index
    relabel = {old: new + 1 for new, old in enumerate(old_order)}
    clusters = pd.Series([relabel[x] for x in labels], name="cluster")

    # Cluster profiles in original interpretable features and HR domains
    feature_profiles = imputed.assign(cluster=clusters).groupby("cluster").mean().T
    domain_profiles = feature_profiles.assign(domain=pd.Series(domains)).groupby("domain").mean().T
    summary = pd.DataFrame({
        "cluster": sorted(clusters.unique()),
        "participants": clusters.value_counts().sort_index().values,
        "share": clusters.value_counts(normalize=True).sort_index().values,
        "current_disorder_yes_share": employees.assign(cluster=clusters).groupby("cluster")[CURRENT].apply(lambda x: x.eq("Yes").mean()).values,
        "treatment_share": employees.assign(cluster=clusters).groupby("cluster")[TREATMENT].mean().values,
    }).merge(domain_profiles.reset_index(), on="cluster")

    metrics.to_csv(OUTPUT / "model_selection_metrics.csv", index=False)
    summary.to_csv(OUTPUT / "cluster_summary.csv", index=False)
    feature_profiles.to_csv(OUTPUT / "cluster_feature_profiles.csv")

    plot = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "Cluster": clusters.astype(str)})
    ax = sns.scatterplot(data=plot, x="PC1", y="PC2", hue="Cluster", alpha=.7, s=35, palette="colorblind")
    ax.set_title(f"K-Means clusters in PCA space (k={selected_k})")
    plt.tight_layout(); plt.savefig(OUTPUT / "03_pca_clusters.png", dpi=220); plt.close()

    plt.figure(figsize=(9, 5))
    sns.heatmap(domain_profiles, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1)
    plt.title("HR risk-domain profiles by cluster"); plt.xlabel("Risk domain"); plt.ylabel("Cluster")
    plt.tight_layout(); plt.savefig(OUTPUT / "04_cluster_profiles.png", dpi=220); plt.close()

    print(f"Rows: {len(raw)} | Employee sample: {len(employees)} | Features: {risk.shape[1]}")
    print(f"PCA: {n_components} components, {pca.explained_variance_ratio_.sum():.2%} variance")
    print(f"Selected K-Means solution: k={selected_k}\n")
    print(summary.to_string(index=False))
    print("\nHR interpretation: Cluster 1 = lower need/mixed support; Cluster 2 = higher need/supportive climate; Cluster 3 = higher need/adverse climate.")
    print("Use clusters only for aggregate program planning, never diagnosis or employment decisions.")


if __name__ == "__main__":
    main()
