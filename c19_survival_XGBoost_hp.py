# -------------------------------------------------------------------
# Import required packages and setup environment
# -------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import pandas as pd
from optuna import create_study, Trial
from sksurv.metrics import concordance_index_censored
from xgboost import XGBRegressor
from org import excel, write_to_file # local scripts for organising output
from sklearn.model_selection import RepeatedStratifiedKFold
from datetime import datetime

# -------------------------------------------------------------------
# Global parameters and constants
# -------------------------------------------------------------------
date_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') # Date and time to save output
REPETITIONS = 5
FOLDS = 5
n_files = 10  # Number of imputed datasets
nbr_of_trials = 100
SEED = 42
rng = np.random.default_rng(seed=SEED)
NAME = 'name_to_save_output'

# Mapping of internal variable names to human-readable names for plots
change_names = {'clinical_frailty_scale': 'CFS', 'leukocytes': 'Leukocytes', 'albumin': 'Albumin', 'ngal': 'NGAL',
                'ddimer': 'D-dimer', 'il6': 'IL-6', 'age_at_admission': 'Age', 'meds_per_pat': '#Meds/Patient',
                'time_at_hospital_before_admission': '#Days in Hosp. before ICU', 'no_CPR': 'DNR order',
                'thrombocytes': 'Thrombocytes', 'neutrofila': 'Neutrophils',
                'cystatin_c': 'Cystatin C',
                'smoker': 'Smoker',
                'congestive_heart_failure': 'Congestive heart failure',
                'cancer': 'Cancer',
                'hypertension': 'Hypertension',
                'myocardial_infarction': 'Myocardial infarction',
                'peripheral_vascular_disease': 'Peripheral vascular disease',
                'cerebrovascular_insult': 'Cerebrovascular insult',
                'chronic_pulmonary_disease': 'Chronic pulmonary disease',
                'rheumatic_disease': 'Rheumatic disease',
                'peptic_ulcer_disease': 'Peptic ulcer disease',
                'diabetes_mellitus_uncompl': 'Diabetes mellitus (uncompl.)',
                'diabetes_mellitus_compl': 'Diabetes mellitus (compl.)',
                'chronic_kidney_disease': 'Chronic kidney disease',
                'imv_within_24h': 'IMV within 24h from ICU admission',
                'noradrenaline': 'Noradrenaline',
                'sex_male': 'Sex',
                'GCS': 'Glasgow Coma Scale',
                'cardiovascular_map': 'Cardiovascular mean arterial pressure',
                'creatinine': 'Creatinine',
                'bilirubin': 'Bilirubin',
                'ferritin': 'Ferritin',
                'laktat': 'Lactate',
                'pct': 'Procalcitonin',
                'pao2': 'Pa0$_2$',
                'paco2': 'PaC0$_2$',
                'aB_pH': 'pH (arterial blood)',
                'lymphocytes': 'Lymphocytes',
                'body_temperature': 'Body temperature (\N{DEGREE SIGN}C)',
                'heart_frequency': 'Heart rate (bpm)',
                'systolic_blood_pressure': 'Systolic blood pressure (mmHg)',
                'arterial_oxygen_tension': 'Arterial oxygen tension (mmHg)',
                'symptomatic_days': '#Symptomatic days',
                'habitual_creatinine': 'Habitual creatinine',
                'ICU_strain': 'ICU burden',
                'endostatin': 'Endostatin',
                'vcam': 'VCAM',
                'icam': 'ICAM',
                'calprotectin': 'Calprotectin',
                'pf_quotient': 'Pa0$_2$/FiO$_2$ Ratio',
                'survival_days_from_icu_start': '#Survival days from ICU admission',
                'mort': 'Dead within one year',
                'study_month': 'Study month'
                }

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def plot_names(variables_to_rename, feat_name, feat_names):
    """
    Replace technical variable names with human-readable names for plotting.

    Args:
        variables_to_rename (dict): Mapping of old → new feature names.
        feat_name (str): A single feature name to rename.
        feat_names (list[str]): List of feature names to check and rename.

    Returns:
        tuple: Updated (feat_name, feat_names) with renamed variables.
    """
    for old, new in variables_to_rename.items():
        if feat_name == old:
            feat_name = new
        elif old in feat_names:
            feat_names = [new if x == old else x for x in feat_names]
    return feat_name, feat_names


# -------------------------------------------------------------------
# Data Import and Preparation
# -------------------------------------------------------------------
# Create a list to store DataFrames
c19_survival_dict = {}
path_to_folder = Path(r"path_to_imputed_datasets")

# Loop to read each Excel file
for i in range(n_files):
    filename = os.path.join(path_to_folder, f'imputed_values_{i}.xlsx')
    df = pd.read_excel(filename, engine='openpyxl')   # Read Excel file into a DataFrame
    c19_survival_dict[i] = df             # Add to dict

# Load dataset containing missing values (pre-imputation)
c19_survival_missing = pd.read_excel(r"path_to_non_imputed_data", engine='openpyxl')

# Identify patients with missing data to identify which patients should be in internal validation dataset
missing_idx = c19_survival_missing[c19_survival_missing.isna().any(axis=1)].index

# -------------------------------------------------------------------
# SHAP label dictionary for consistent plotting and naming
# -------------------------------------------------------------------
labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "SHAP value",
    'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
    'VALUE_FOR': "SHAP value for\n%s",
    'PLOT_FOR': "SHAP plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Feature value",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'JOINT_VALUE': "Joint SHAP value",
    'MODEL_OUTPUT': "Model output value"
}

# Create hp tuning pipeline
def instantiate_xgboost(trial : Trial) -> XGBRegressor:
  params = {
        'max_depth': trial.suggest_int('max_depth', 1, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1),
        'gamma': trial.suggest_float('min_split_loss', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 100, log=True), # L2
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 100, log=True), # L1
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'tree_method': 'hist'
    }
  return XGBRegressor(**params)


# -------------------------------------------------------------------
# Objective Function for Training and Evaluation
# -------------------------------------------------------------------
def objective(data_deque, non_full_index, trial) -> float:

    rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPETITIONS,
                                   random_state=SEED)  # Repeats Stratified K-Fold n times with different randomization in each repetition.

    number_of_patients = len(data_deque[0])

    # Prepare dicts to save stuff for CI and so on
    train_predictions = {key: [] for key in range(number_of_patients)}
    val_predictions = {key: [] for key in range(number_of_patients)}

    ## 1) First loop: imputed datasets
    # Get the dataset out of a deque
    for i_loop, data in data_deque.items():
        # Drop unnecessary column
        if 'index' in data.columns.to_list():
            data = data.drop('index', axis=1)

        # Divide into x and y
        XX = data.drop(['mort'], axis=1)
        yy = data.copy(deep=True)

        for cat in data.columns.to_list():
            if cat not in ['mort']:
                yy = yy.drop(cat, axis=1)

        # Instantiate new model for this dataset
        model = instantiate_xgboost(trial)

        ## 2) Second loop: Fold*repetition loop
        for i, (train_index, test_index) in enumerate(rskf.split(XX, yy)):
            X_train = XX.iloc[train_index]
            y_train = yy.iloc[train_index]
            X_test = XX.iloc[test_index]

            # Both time and mortality in y
            y_train.loc[:, 'survival_days_from_icu_start'] = X_train['survival_days_from_icu_start']
            X_train = X_train.drop('survival_days_from_icu_start', axis=1)
            X_test = X_test.drop('survival_days_from_icu_start', axis=1)

            y_train_cox = np.where(
                y_train["mort"].eq(1),
                y_train["survival_days_from_icu_start"],
                -y_train["survival_days_from_icu_start"]
            )

            # fit model
            model.fit(X_train, y_train_cox, verbose=True)

            # predict on train and test data
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_test)

            # save training predictions from this fold
            for nbr, idx in enumerate(train_index):
                train_predictions[idx].append(train_pred[nbr])

            # save validation predictions from this fold
            for nbr, idx in enumerate(test_index):
                if idx in non_full_index:
                    continue  # skip adding this prediction if it is not a full-case patient
                else:
                    val_predictions[idx].append(val_pred[nbr])

        # End of loop 2

    # End of loop 1 (imputed datasets)

    # average predictions over all repeated folds
    train_predictions_mean = {}
    val_predictions_mean = {}
    for idx, predictions in train_predictions.items():
        mean_predictions = np.mean(predictions)
        train_predictions_mean[idx] = mean_predictions
    for idx, predictions in val_predictions.items():
        mean_predictions = np.mean(predictions)
        val_predictions_mean[idx] = mean_predictions


    # from dict to df
    train_predictions_mean_df = pd.DataFrame.from_dict(train_predictions_mean, orient='index')
    val_predictions_mean_df = pd.DataFrame.from_dict(val_predictions_mean, orient='index')

    # combine with y_true into one df
    predictions_and_targets = train_predictions_mean_df.copy(deep=True)
    predictions_and_targets.rename(columns={0: 'training_predictions'}, inplace=True)
    predictions_and_targets.insert(1, 'validation_predictions', val_predictions_mean_df)
    true_values = XX.copy(deep=True)
    for cat in XX.columns.to_list():
        if cat not in ['survival_days_from_icu_start']:
            true_values = true_values.drop(cat, axis=1)
    true_mort = yy.copy(deep=True)
    for cat in yy.columns.to_list():
        if cat not in ['mort']:
            true_mort = true_mort.drop(cat, axis=1)
    predictions_and_targets.insert(2, 'survival_days_from_icu_start', true_values[
        'survival_days_from_icu_start'])  # since only one var can be in the split, this is taken from X and not y
    predictions_and_targets.insert(3, 'mort', true_mort['mort'])


    predictions_and_targets_train = predictions_and_targets.copy(deep=True)
    # Drop imputed patients that were just used for training
    predictions_and_targets_val = predictions_and_targets.dropna(axis=0)

    # Save
    excel(date_time, predictions_and_targets_train, NAME + '_train')
    excel(date_time, predictions_and_targets_val, NAME+ '_val')

    # c-index, nbr of concordant pairs, nbr of discordant pairs, nbr of pairs with tied risk scores, nbr of comparable pairs with tied survival times
    train_cindex, _, _, _, _ = concordance_index_censored(predictions_and_targets_train['mort'].astype(bool),
                                                          predictions_and_targets_train['survival_days_from_icu_start'],
                                                          predictions_and_targets_train['training_predictions'])

    val_cindex, _, _, _, _ = concordance_index_censored(predictions_and_targets_val['mort'].astype(bool),
                                                        predictions_and_targets_val['survival_days_from_icu_start'],
                                                        predictions_and_targets_val['validation_predictions'])

    # Log additional information
    trial.set_user_attr(
        'train_cindex',
        train_cindex)

    return val_cindex

# -------------------------------------------------------------------
# Optuna Study Runner
# -------------------------------------------------------------------
def xgb_train(data):
    # A study corresponds to an optimization task, i.e., a set of trials.
    study = create_study(study_name=str(NAME) + '_optimisation', direction='maximize')  # want to maximize the validation C-index
    study.optimize(lambda trial: objective(data, missing_idx, trial),
                   n_trials=nbr_of_trials)

    # Save study parameters and results
    trials_df = study.trials_dataframe()
    if 'value' in trials_df.columns:
        trials_df.rename(columns={'value': 'validation_cindex'}, inplace=True)
    if 'number' in trials_df.columns:
        trials_df.rename(columns={'number': 'trial_number'}, inplace=True)
    if 'user_attrs_train_cindex' in trials_df.columns:
        trials_df.rename(columns={'user_attrs_train_cindex': 'train_cindex'}, inplace=True)
    excel(date_time, trials_df, 'trials', str(NAME) + '_hp_opt')

    best_params = study.best_params
    best_params_df = pd.DataFrame.from_dict(best_params, orient='index')
    excel(date_time, best_params_df, 'best_params', str(NAME) + '_hp_opt')

    # Save AUC and features used
    best_value = study.best_value
    text = 'C-index: ' + str(round(best_value, 3)) + '\nfeatures: ' + str(data[0].columns.tolist())
    write_to_file(date_time, str(NAME), text, read_text=True, subfolder=str(NAME) + '_hp_opt')

# -------------------------------------------------------------------
# Run the pipeline
# -------------------------------------------------------------------
xgb_train(c19_survival_dict)

