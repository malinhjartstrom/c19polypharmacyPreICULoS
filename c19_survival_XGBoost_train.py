# -------------------------------------------------------------------
# Import required packages and setup environment
# -------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import pandas as pd
from optuna import create_study
from sksurv.metrics import concordance_index_censored
import shap
from collections import deque
from xgboost import XGBRegressor
from org import excel, write_to_file # local scripts for organising output
from sklearn.model_selection import RepeatedStratifiedKFold
from datetime import datetime

# -------------------------------------------------------------------
# Global parameters and constants
# -------------------------------------------------------------------
date_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') # Date and time to save output
REPETITIONS = 5
FOLDS =5
SEED = 42
nbr_of_trials = 1 # Since we are not doing a trial but running the pipeline once
n_bootstraps = 1000
n_files = 10
NAME = 'name_to_save_output'
rng = np.random.default_rng(seed=SEED)

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

# -------------------------------------------------------------------
# SHAP and Model Helper Functions
# -------------------------------------------------------------------
def get_shap_values(
        estimator,
        X_train,
        X_test
):
    explainer = shap.TreeExplainer(
        estimator,
        data=X_train,
        feature_perturbation="interventional",
        model_output="raw"
    )

    explanation = explainer(X_test)

    return explanation

# -------------------------------------------------------------------
# Model Initialization and Hyperparameters
# -------------------------------------------------------------------
def best_params_xgboost() -> XGBRegressor:

    best_params = pd.read_excel(
        r"paths_to_the_best_parameters_from_hyperparameter_optimization", engine='openpyxl')
    best_params = best_params.set_index('Unnamed: 0')

    params = {
        'max_depth': int(best_params.loc['max_depth'].values[0]),
        'learning_rate': best_params.loc['learning_rate'].values[0],
        'n_estimators': int(best_params.loc['n_estimators'].values[0]),
        'subsample': best_params.loc['subsample'].values[0],
        'min_split_loss': best_params.loc['min_split_loss'].values[0],
        'min_child_weight': best_params.loc['min_child_weight'].values[0],
        'reg_lambda': best_params.loc['reg_lambda'].values[0],  # L2
        'reg_alpha': best_params.loc['reg_alpha'].values[0],  # L1
        'colsample_bytree': best_params.loc['colsample_bytree'].values[0],
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'tree_method': 'hist'
    }

    print(params)
    return XGBRegressor(**params)

# -------------------------------------------------------------------
# Objective Function for Training and Evaluation
# -------------------------------------------------------------------
def objective(data_deque, non_full_index) -> float:

    rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPETITIONS,
                                   random_state=SEED)  # Repeats Stratified K-Fold n times with different randomization in each repetition.

    number_of_patients = len(data_deque[0])
    # Prepare dicts to save stuff for CI and so on
    train_predictions = {key: [] for key in range(number_of_patients)}
    val_predictions = {key: [] for key in range(number_of_patients)}

    shap_values = {key: deque() for key in range(number_of_patients)}
    feature_values = {key: deque() for key in range(number_of_patients)}
    base_values = {key: [] for key in range(number_of_patients)}
    feature_names = []

    ## 1) First loop
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
        model = best_params_xgboost()  # instantiate_xgboost(trial) #

        # Loop for imputation C-index
        ctrain_predictions = {key: [] for key in range(number_of_patients)}
        cval_predictions = {key: [] for key in range(number_of_patients)}

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

            # Calculate SHAP values for each validation dataset
            explanation = get_shap_values(model, X_train, X_test)


            # predict on train and test data
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_test)

            # save training predictions from this fold
            for nbr, idx in enumerate(train_index):
                train_predictions[idx].append(train_pred[nbr])
                ctrain_predictions[idx].append(train_pred[nbr])

            # save validation predictions from this fold
            for nbr, idx in enumerate(test_index):
                if idx in non_full_index:
                    continue  # skip adding this prediction if it is not a full-case patient
                else:
                    val_predictions[idx].append(val_pred[nbr])
                    cval_predictions[idx].append(val_pred[nbr])
                    # Store validation SHAP values
                    shap_values[idx].append(explanation[nbr].values)
                    feature_values[idx].append(explanation[nbr].data)
                    base_values[idx].append(explanation[nbr].base_values)

            if not len(feature_names) is None:
                feature_names = explanation.feature_names

        # End of loop 2 (folds*repetitions)

        # ------ Save results (predictions and c-index) for each imputed dataset (comment from reviewer 1) -------
        # average predictions over all repeated folds for this imputed dataset (i_loop)
        train_predictions_i_loop_mean = {}
        val_predictions_i_loop_mean = {}
        for idx, predictions in ctrain_predictions.items():
            mean_predictions_i_loop = np.mean(predictions)
            train_predictions_i_loop_mean[idx] = mean_predictions_i_loop
        for idx, predictions in cval_predictions.items():
            mean_predictions_i_loop = np.mean(predictions)
            val_predictions_i_loop_mean[idx] = mean_predictions_i_loop

        # Make df
        train_predictions_i_loop_df = pd.DataFrame.from_dict(train_predictions_i_loop_mean, orient='index')
        val_predictions_i_loop_df = pd.DataFrame.from_dict(val_predictions_i_loop_mean, orient='index')
        excel(date_time, train_predictions_i_loop_df, str(NAME) + '_train_predictions_imputed_dataset_' + str(i_loop), subfolder=NAME)
        excel(date_time, val_predictions_i_loop_df, str(NAME) + '_val_predictions_imputed_dataset_' + str(i_loop), subfolder=NAME)

        # combine with y_true into one df
        predictions_and_targets_i_loop = train_predictions_i_loop_df.copy(deep=True)
        predictions_and_targets_i_loop.rename(columns={0: 'training_predictions'}, inplace=True)
        predictions_and_targets_i_loop.insert(1, 'validation_predictions', val_predictions_i_loop_df)
        true_values_i_loop = XX.copy(deep=True)
        for cat in XX.columns.to_list():
            if cat not in ['survival_days_from_icu_start']:
                true_values_i_loop = true_values_i_loop.drop(cat, axis=1)
        true_mort_i_loop = yy.copy(deep=True)
        for cat in yy.columns.to_list():
            if cat not in ['mort']:
                true_mort_i_loop = true_mort_i_loop.drop(cat, axis=1)
        predictions_and_targets_i_loop.insert(2, 'survival_days_from_icu_start', true_values_i_loop[
            'survival_days_from_icu_start'])  # since only one var can be in the split, this is taken from X and not y
        predictions_and_targets_i_loop.insert(3, 'mort', true_mort_i_loop['mort'])

        # Drop imputed patients that were just used for training
        predictions_and_targets_val = predictions_and_targets_i_loop.dropna(axis=0)

        excel(date_time, predictions_and_targets_i_loop, str(NAME) + '_predictions_and_targets_train' + str(i_loop),
              subfolder=NAME)
        excel(date_time, predictions_and_targets_val, str(NAME) + '_predictions_and_targets_val' + str(i_loop),
              subfolder=NAME)

        # Store c-index for each impute dataset(comment from reviewer 1)
        # c-index, nbr of concordant pairs, nbr of discordant pairs, nbr of pairs with tied risk scores, nbr of comparable pairs with tied survival times
        train_loop_cindex, _, _, _, _ = concordance_index_censored(predictions_and_targets_i_loop['mort'].astype(bool),
                                                              predictions_and_targets_i_loop['survival_days_from_icu_start'],
                                                              predictions_and_targets_i_loop["training_predictions"])

        val_loop_cindex, _, _, _, _ = concordance_index_censored(predictions_and_targets_val['mort'].astype(bool),
                                                              predictions_and_targets_val['survival_days_from_icu_start'],
                                                              predictions_and_targets_val["validation_predictions"])
        print(train_loop_cindex, val_loop_cindex)
        text_i_loop = '\n\nImputed dataset ' + str(i_loop) + '\nC-index val: ' + str(round(val_loop_cindex, 3)) + '\nC-index train: ' + str(round(train_loop_cindex, 3))
        write_to_file(date_time, str(NAME), text_i_loop, read_text=True, subfolder=str(NAME) + '_c-index_imputed_datasets')
        # --------- End of saving c-index for each imputed dataset separately ---------

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

    # Average SHAP over validation results
    shap_mean = {}
    feature_mean = {} # also over the differently imputed datasets
    base_mean = {}

    # SHAP values
    for idx, predictions in shap_values.items():
        mean_predictions = np.mean(predictions, axis=0)  # element-wise over each matrix in each deque
        shap_mean[idx] = mean_predictions
    # Data values
    for idx, predictions in feature_values.items():
        mean_predictions = np.mean(predictions, axis=0)  # element-wise over each matrix in each deque
        feature_mean[idx] = mean_predictions
    # Base values
    for idx, predictions in base_values.items():
        mean_predictions = np.mean(predictions)
        base_mean[idx] = mean_predictions

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

    # Drop imputed patients that were just used for training
    predictions_and_targets_val = predictions_and_targets.dropna(axis=0)

    excel(date_time, predictions_and_targets_val, NAME + '_val')
    excel(date_time, predictions_and_targets, NAME+ '_train')

    # c-index, nbr of concordant pairs, nbr of discordant pairs, nbr of pairs with tied risk scores, nbr of comparable pairs with tied survival times
    train_cindex, _, _, _, _ = concordance_index_censored(predictions_and_targets['mort'].astype(bool),
                                                          predictions_and_targets['survival_days_from_icu_start'],
                                                          predictions_and_targets['training_predictions'])

    val_cindex, _, _, _, _ = concordance_index_censored(predictions_and_targets_val['mort'].astype(bool),
                                                        predictions_and_targets_val['survival_days_from_icu_start'],
                                                        predictions_and_targets_val['validation_predictions'])

    print(train_cindex, val_cindex)

    # CI for C-index
    train_c = []
    val_c = []
    for re in range(n_bootstraps):
        bs_sample_train = predictions_and_targets.sample(n=len(predictions_and_targets),
                                                   replace=True) # Sampling with replacement
        bs_sample_val = predictions_and_targets_val.sample(n=len(predictions_and_targets_val),
                                                   replace=True) # Sampling with replacement
        cindex_train = concordance_index_censored(bs_sample_train['mort'].astype(bool),
                                                  bs_sample_train['survival_days_from_icu_start'],
                                                  bs_sample_train['training_predictions'])
        cindex_val = concordance_index_censored(bs_sample_val['mort'].astype(bool),
                                                bs_sample_val['survival_days_from_icu_start'],
                                                bs_sample_val['validation_predictions'])
        train_c.append(cindex_train[0])
        val_c.append(cindex_val[0])

    # Pointwise percentile bootstrap 95% CI
    ci_lower_train = np.percentile(train_c, 2.5, axis=0)
    ci_upper_train = np.percentile(train_c, 97.5, axis=0)
    ci_lower_val = np.percentile(val_c, 2.5, axis=0)
    ci_upper_val = np.percentile(val_c, 97.5, axis=0)


    print(ci_lower_train, ci_upper_train)
    print(ci_lower_val, ci_upper_val)

    text = '\nC-index val: ' + str(round(val_cindex, 3)) + ' (' + str(round(ci_lower_val, 3)) + '-' + str(
        round(ci_upper_val, 3)) + ')'
    write_to_file(date_time, str(NAME), text, read_text=True, subfolder=str(NAME) + '_hp_opt')
    text = '\nC-index train: ' + str(round(train_cindex, 3)) + ' (' + str(round(ci_lower_train, 3)) + '-' + str(
        round(ci_upper_train, 3)) + ')' + '\n\n'
    write_to_file(date_time, str(NAME), text, read_text=True, subfolder=str(NAME) + '_hp_opt')

    # Remove nans
    full_val_shap_mean = {k: v for k, v in shap_mean.items() if not isinstance(v, float)}
    # From dict to df for shap values
    shap_mean_df = pd.DataFrame.from_dict(full_val_shap_mean, orient='index', columns=feature_names)

    # Remove empty deques
    full_val_shap_values = {k: v for k, v in shap_values.items() if
                            bool(v)}  # bool(v) is True if the deque is not empty
    full_val_feature_values = {k: v for k, v in feature_values.items() if
                               bool(v)}  # bool(v) is True if the deque is not empty

    # get all the SHAP values and data from all folds and repetitions into one column per variable
    stored_shap_values = {key: [] for key in range(len(feature_names))}
    stored_feature_values = {key: [] for key in range(len(feature_names))}
    for variable_nbr in range(len(feature_names)):
        print('Variable number', variable_nbr)
        # For SHAP
        for dict_idx, patient_deque in full_val_shap_values.items():
            print('Deque', dict_idx)
            for iteration in range(len(patient_deque)):
                print('Array', iteration)
                stored_shap_values[variable_nbr].append(patient_deque[iteration][variable_nbr])
        # For data
        for dict_idx, patient_deque in full_val_feature_values.items():
            print('Deque', dict_idx)
            for iteration in range(len(patient_deque)):
                print('Array', iteration)
                stored_feature_values[variable_nbr].append(patient_deque[iteration][variable_nbr])

    # from dict to df
    stored_shap_values_df = pd.DataFrame.from_dict(stored_shap_values, orient='columns')
    stored_feature_values_df = pd.DataFrame.from_dict(stored_feature_values, orient='columns')

    # rename columns
    stored_shap_values_df = stored_shap_values_df.set_axis(feature_names, axis=1)
    stored_feature_values_df = stored_feature_values_df.set_axis(feature_names, axis=1)

    ### SAVE for later ###
    # Save shap values and data over all folds
    excel(date_time, stored_shap_values_df, str(NAME) + '_all_shap_values', subfolder=NAME)
    excel(date_time, stored_feature_values_df, str(NAME) + '_all_fold_data', subfolder=NAME)

    # Save the mean of all rounds of SHAP values
    excel(date_time, shap_mean_df, str(NAME) + '_mean_shap_values', subfolder=NAME)

    # Save data (a mean over the 10 imputed datasets)
    # Remove nans
    full_val_feature_mean = {k: v for k, v in feature_mean.items() if not isinstance(v, float)}
    # From dict to df for shap values
    feature_mean_df = pd.DataFrame.from_dict(full_val_feature_mean, orient='index', columns=feature_names)
    excel(date_time, feature_mean_df, str(NAME) + '_features_val_mean', subfolder=NAME)

    return val_cindex

# -------------------------------------------------------------------
# Optuna Study Runner
# -------------------------------------------------------------------
def xgb_train(data):
    # A study corresponds to an optimization task, i.e., a set of trials.
    study = create_study(study_name=str(NAME) + '_optimisation', direction='maximize')  # want to maximize the validation C-index
    study.optimize(lambda trial: objective(data, missing_idx),
                   n_trials=nbr_of_trials)

    # Save study parameters and results
    trials_df = study.trials_dataframe()
    if 'value' in trials_df.columns:
        trials_df.rename(columns={'value': 'validation_cindex'}, inplace=True)
    if 'number' in trials_df.columns:
        trials_df.rename(columns={'number': 'trial_number'}, inplace=True)
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

