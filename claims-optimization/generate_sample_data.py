import numpy as np, pandas as pd
from pathlib import Path

def main():
    rng = np.random.default_rng(7)
    N = 5000
    df = pd.DataFrame({
        "claim_id": [f"C{100000+i}" for i in range(N)],
        "claim_amount": rng.gamma(2.0, 500, N).round(2),
        "patient_age": rng.integers(18, 90, N),
        "patient_gender": rng.choice(["F","M","O"], N, p=[0.49,0.49,0.02]),
        "insurance_provider": rng.choice(["Aetna","BlueCross","Cigna","United","Kaiser"], N),
        "diagnosis_code": rng.choice([f"I{n:03d}" for n in range(100)], N),
        "procedure_code": rng.choice([f"P{n:03d}" for n in range(200)], N),
        "in_network": rng.integers(0,2,N),
        "preauth_obtained": rng.integers(0,2,N),
        "prior_denials_count": rng.poisson(0.3, N),
        "days_since_service": rng.integers(0, 120, N)
    })
    logit = (
        -2.0
        + 0.002 * (5000 - df["claim_amount"].clip(0,5000))
        + 0.03 * (65 - df["patient_age"])
        + 0.8  * df["in_network"]
        + 1.0  * df["preauth_obtained"]
        - 0.4  * df["prior_denials_count"]
        - 0.01 * df["days_since_service"]
    )
    p_approved = 1/(1+np.exp(-logit))
    df["status"] = np.where(rng.random(N) < p_approved, "Approved", "Denied")
    denial_map = ["Missing preauth","Out-of-network","Invalid code","Exceeded limits","Other"]
    df["denial_reason"] = np.where(df["status"]=="Denied", rng.choice(denial_map, size=N), "")
    Path("data").mkdir(exist_ok=True, parents=True)
    out = Path("data/claims.csv")
    df.to_csv(out, index=False)
    print(f"Saved {out.resolve()}")

if __name__ == "__main__":
    main()