import pandas as pd

list_of_dicts = pd.DataFrame([
    {'StudentID': 101, 'Name': 'Alice', 'Score': 85, 'Major': 'Computer Science'},
    {'StudentID': 102, 'Name': 'Bob', 'Score': 92, 'Major': 'Physics'},
    {'StudentID': 103, 'Name': 'Charlie', 'Score': 78, 'Major': 'Mathematics'},
    {'StudentID': 104, 'Name': 'David', 'Score': 95, 'Major': 'Computer Science'},
    {'StudentID': 105, 'Name': 'Eve', 'Score': 88, 'Major': 'Biology'},
    {'StudentID': 106, 'Name': 'Frank', 'Score': 72, 'Major': 'Mathematics'},
    {'StudentID': 107, 'Name': 'Grace', 'Score': 99, 'Major': 'Physics'},
    {'StudentID': 108, 'Name': 'Heidi', 'Score': 81, 'Major': 'Biology'},
    {'StudentID': 109, 'Name': 'Ivan', 'Score': 90, 'Major': 'Computer Science'},
    {'StudentID': 110, 'Name': 'Judy', 'Score': 84, 'Major': 'Mathematics'}
])
list_of_dicts

dict_of_lists = pd.DataFrame({
    'StudentID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Heidi', 'Ivan', 'Judy'],
    'Score': [85, 92, 78, 95, 88, 72, 99, 81, 90, 84],
    'Major': ['Computer Science', 'Physics', 'Mathematics', 'Computer Science', 'Biology', 'Mathematics', 'Physics', 'Biology', 'Computer Science', 'Mathematics']
})
dict_of_lists

list_of_lists = pd.DataFrame([
    [101, 'Alice', 85, 'Computer Science'],
    [102, 'Bob', 92, 'Physics'],
    [103, 'Charlie', 78, 'Mathematics'],
    [104, 'David', 95, 'Computer Science'],
    [105, 'Eve', 88, 'Biology'],
    [106, 'Frank', 72, 'Mathematics'],
    [107, 'Grace', 99, 'Physics'],
    [108, 'Heidi', 81, 'Biology'],
    [109, 'Ivan', 90, 'Computer Science'],
    [110, 'Judy', 84, 'Mathematics']
])
list_of_lists.columns = ['StudentID', 'Name', 'Score', 'Major']
list_of_lists

student_df = list_of_lists.copy(deep=True)
print(student_df.loc[3, :], end="\n\n")
print(student_df.loc[:, 'Score'], end="\n\n")
print(student_df.loc[6, 'Major'], end="\n\n")
print(student_df.loc[2:5, :], end="\n\n")
print(student_df.loc[[0, 4, 9], ['Name', 'Major']], end="\n\n")

df_reindexed = student_df.set_index("StudentID", drop=True)
df_reindexed

df_reindexed.loc[[102, 104, 107], :]

df_reindexed.iloc[[1, 3, 6], :]

math_majors = df_reindexed['Major'] == 'Mathematics'
print(df_reindexed[math_majors], end="\n\n")

print(df_reindexed.query('Score > 85'), end="\n\n")

def find_top_students_in_major(df, major_name):
    return df[df['Major'] == major_name].sort_values(by='Score', ascending=False).head(1)[['Name', 'Score']]

print(find_top_students_in_major(df_reindexed, 'Computer Science'), end="\n\n")
print(find_top_students_in_major(df_reindexed, 'Physics'), end="\n\n")
print(find_top_students_in_major(df_reindexed, 'Chemistry'), end="\n\n")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
duckdb = None

stroke_df = pd.read_csv('data/hw3/stroke/healthcare-dataset-stroke-data.csv')
location_codes_df = pd.read_csv('data/hw3/stroke/location_codes.csv')
stroke_df.head()

def get_cardiovascular_disease(row):
    if row['hypertension'] == 1 or row['heart_disease'] == 1:
        return 1
    else:
        return 0

stroke_df['cardiovascular_disease'] = stroke_df.apply(get_cardiovascular_disease, axis=1)

print(stroke_df[['hypertension', 'heart_disease', 'cardiovascular_disease']].head(), end="\n\n")
print(stroke_df['cardiovascular_disease'].value_counts(), end="\n\n")

def plot_histograms_by_category(df, continuous_col, categorical_col, categorical_threshold=10):
    if continuous_col not in df.columns:
        print(f"Column '{continuous_col}' not found in the dataframe.")
        return None
    if categorical_col not in df.columns:
        print(f"Column '{categorical_col}' not found in the dataframe.")
        return None

    if df[continuous_col].nunique() < categorical_threshold:
        print(f"'{continuous_col}' looks categorical (fewer than {categorical_threshold} unique values), not continuous.")
        return None
    if df[categorical_col].nunique() >= categorical_threshold:
        print(f"'{categorical_col}' looks continuous ({df[categorical_col].nunique()} unique values), not categorical.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = df[categorical_col].unique()
    for category in categories:
        subset = df[df[categorical_col] == category]
        ax.hist(subset[continuous_col], alpha=0.6, label=f'{categorical_col} = {category}')

    ax.set_title(f'Distribution of {continuous_col} by {categorical_col}')
    ax.set_xlabel(continuous_col)
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.show()

plot_histograms_by_category(stroke_df, 'age', 'stroke')
plot_histograms_by_category(stroke_df, 'bmi', 'gender')
plot_histograms_by_category(stroke_df, 'stroke', 'age')

def calculate_stroke_prevalence(df, group_by_cols, categorical_threshold=10):
    for col in group_by_cols:
        if col not in df.columns:
            print(f"Column '{col}' not found in the dataframe.")
            return None
    for col in group_by_cols:
        if df[col].nunique() >= categorical_threshold:
            print(f"'{col}' looks continuous ({df[col].nunique()} unique values), not categorical.")
            return None

    group_sizes = df.groupby(group_by_cols).size()
    group_stroke_counts = df.groupby(group_by_cols)['stroke'].sum()
    prevalence = group_stroke_counts / group_sizes

    prevalence_df = prevalence.to_frame(name='stroke_prevalence').reset_index()
    return prevalence_df

print(calculate_stroke_prevalence(stroke_df, ['smoking_status']), end="\n\n")
print(calculate_stroke_prevalence(stroke_df, ['ever_married', 'work_type']), end="\n\n")
print(calculate_stroke_prevalence(stroke_df, ['age']), end="\n\n")
print(calculate_stroke_prevalence(stroke_df, ['residence_type']), end="\n\n")

def preprocess_for_analysis(df, cols_to_impute, cols_to_scale, cols_to_encode, cols_to_drop):
    processed_df = df.copy()

    processed_df = processed_df.drop(columns=cols_to_drop)

    for col in cols_to_impute:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mean())

    for col in cols_to_scale:
        mean = processed_df[col].mean()
        std = processed_df[col].std()
        processed_df[f'{col}_scaled'] = (processed_df[col] - mean) / std
        processed_df = processed_df.drop(columns=[col])

    encoded_df = pd.get_dummies(processed_df[cols_to_encode], drop_first=True)
    processed_df = processed_df.drop(columns=cols_to_encode)
    processed_df = processed_df.join(encoded_df)

    return processed_df

impute_list = ['bmi']
scale_list = ['age', 'avg_glucose_level', 'bmi']
encode_list = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
drop_list = ['id', 'location_code']

processed_stroke_df = preprocess_for_analysis(stroke_df, impute_list, scale_list, encode_list, drop_list)
print(processed_stroke_df.head(), end="\n\n")
processed_stroke_df.info()

merged_df = stroke_df.merge(location_codes_df, on='location_code', how='left')
merged_df.head()

sql_query = """
SELECT *
FROM stroke_df AS l
LEFT JOIN location_codes_df AS r
ON l.location_code = r.location_code
WHERE r.state = 'NY'
"""

sql_result_df = duckdb.query(sql_query).to_df()

print(sql_result_df.head(), end="\n\n")
print(sql_result_df['state'].unique(), end="\n\n")