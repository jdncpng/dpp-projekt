# defining numeric, categorical and binary columns
num_cols = ['Age', 'BMI', 'Pack_Years', 'Symptom_Count', 'Risk_Count']

cat_cols = ['Smoking_Status', 'Air_Pollution_Exposure', 'Alcohol_Use', 
            'Exercise_Frequency', 'Genetic_Mutation', 'NSCLC_Subtype', 
             'Diagnosis_Method', 'Treatment']

binary_cols = ['Gender', 'Secondhand_Smoke', 'Family_History', 
               'Occupational_Hazard', 'Chronic_Lung_Disease', 
               'Asbestos_Exposure', 'Radon_Exposure', 
               'Previous_Cancer_History', 'Coughing','Shortness_of_Breath', 
               'Chest_Pain', 'Coughing_Blood', 'Fatigue', 'Weight_Loss', 
               'Wheezing', 'Recurrent_Infections', 'Swallowing_Difficulty', 
               'Finger_Clubbing', 'Metastasis', 'Cancer_Type']

ord_cols = ['Cancer_Stage_Numeric']

def clean_data(X):
    """
    Drops columns with no predictive value:
    - Patient_ID: only an identifier
    - Diagnosis_Year/Date: redundant temporal info
    - Survival_Months: target leakage. 
    """
    X = X.drop(['Patient_ID', 'Diagnosis_Year', 
                              'Diagnosis_Date', 'Survival_Months'], axis=1)
    return X

# removing columns again to reduce correlation instead of using PCA
def engineer_X(X):

    # adding pack_years as a feature (a standard clinical metric)
    X['Pack_Years'] = (X['Cigarettes_Per_Day'] / 20) * X['Years_Smoking']

    # changing cancer stage to ordinal values
    stage_map = {'Stage I': 1,
                 'Stage II': 2,
                 'Stage III': 3,
                 'Stage IV': 4}
    X['Cancer_Stage_Numeric'] = X['Cancer_Stage'].map(stage_map)

    # changing binary values to 1 and 0
    binary_cols = ['Secondhand_Smoke', 'Family_History', 'Occupational_Hazard',
                'Chronic_Lung_Disease', 'Asbestos_Exposure', 'Radon_Exposure',
                'Previous_Cancer_History', 'Coughing','Shortness_of_Breath', 
                'Chest_Pain', 'Coughing_Blood', 'Fatigue', 'Weight_Loss', 
                'Wheezing', 'Recurrent_Infections', 'Swallowing_Difficulty', 
                'Finger_Clubbing', 'Metastasis']

    X[binary_cols] = X[binary_cols].eq("Yes").astype(int)

    # converting gender values
    X['Gender'] = X['Gender'].eq("Male").astype(int)
    
    # converting cancer types
    X['Cancer_Type'] = X['Cancer_Type'].eq("NSCLC").astype(int)

    # creating a count column for symptoms and risk
    symptom_cols = ['Coughing','Shortness_of_Breath', 'Chest_Pain', 
                    'Coughing_Blood', 'Fatigue', 'Weight_Loss', 'Wheezing', 
                    'Recurrent_Infections', 'Swallowing_Difficulty', 
                    'Finger_Clubbing', 'Metastasis']
    X['Symptom_Count'] = X[symptom_cols].sum(axis=1).astype(int)

    risk_cols = ['Secondhand_Smoke', 'Family_History', 'Occupational_Hazard',
                'Chronic_Lung_Disease', 'Asbestos_Exposure', 'Radon_Exposure',
                'Previous_Cancer_History']
    X['Risk_Count'] = X[risk_cols].sum(axis=1).astype(int)

    X = X.drop(columns=['Cigarettes_Per_Day', 'Years_Smoking', 
                        'Cancer_Stage', 'Tumor_Size_cm', 'Country',
                        'WHO_Region'])

    return X


def bool_target(y):
    """Converts target column from 'Yes'/'No' to 1/0."""
    y['Survived'] = y['Survived'].eq("Yes").astype(int)
    return y
