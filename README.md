
# Polypharmacy and pre-ICU length of hospital stay are associated with one-year mortality in critical COVID-19 in a retrospective exploratory explainable machine-learning survival analysis
One-year mortality risk ML survival analysis and explainable AI in patients with critical COVID-19.<br>

---

## Scripts

<code>c19_survival_XGBoost_hp.py</code> <br>
- Hyperparameter search

<code>c19_survival_XGBoost_train.py</code> <br>
- Training the final model

<code>c19_survival_XGBoost_XAI.py</code> <br>
- XAI and SHAP-derived HRs

<code>c19_survival_forest_plot.R</code> <br>
- Making the forest plot of the SHAP-derived HRs

<code>c19shapenvironment.yml</code> <br>
- Configurations

The Cox PH model is trained and evaluated by replacing the XGBoost model with the Cox PH model from the scikit-survival Python module.

The code for preprocessing the data is not available due to the sensitivity of the data. 

---


## Contact
<address>
<a href="mailto:malin.hjartstrom@med.lu.se">Malin Hjärtström</a><br>
</address> 




