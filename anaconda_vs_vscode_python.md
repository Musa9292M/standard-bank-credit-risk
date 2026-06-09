# Anaconda vs VS Code Built-in Python

**Prepared by:** Musawenkosi Bongumusa (Musa) Vilakazi  
**Position:** Senior Risk Analytics Engineer  
**Company:** Standard Bank Group  
**Date:** June 08, 2026

## Why Anaconda + VS Code is the Best Setup for Risk Analytics

### Comparison Table

| Aspect                        | **Anaconda (Recommended)**                          | **VS Code Built-in / System Python**              |
|-------------------------------|-----------------------------------------------------|---------------------------------------------------|
| Package Management            | Excellent (conda + pip)                             | Only pip — frequent conflicts                     |
| Environment Isolation         | Very easy (`conda create -n risk_model`)            | Manual and error-prone                            |
| Complex Package Installation  | Reliable (XGBoost, SHAP, Streamlit, etc.)          | Often fails due to dependency issues              |
| Reproducibility               | High (`environment.yml`)                            | Low                                               |
| Stability for Banking Work    | Preferred in financial institutions                 | Riskier for production models                     |
| Learning Curve                | Medium at start                                     | Simpler initially                                 |

### Key Advantages of Using Anaconda

1. **Isolated Environments** — Prevents package conflicts between projects.
2. **Reliable Installation** of data science libraries (especially XGBoost, SHAP, etc.).
3. **Better for Regulatory Work** — Easier to maintain and share consistent environments.
4. **Long-term Maintainability** — Export your full environment with one command:
   ```bash
   conda env export > environment.yml