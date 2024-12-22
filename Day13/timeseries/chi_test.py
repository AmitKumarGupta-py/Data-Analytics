from pandas import read_csv, crosstab
from scipy.stats import chi2_contingency


df = read_csv("airline_passenger_satisfaction.csv")

male_df = df[df["Gender"] == "Male"]

category1 = "Class"
category2 = "On-board Service"

crosstab = crosstab(male_df[category1],male_df[category2])

chi2 , pval, degree_of_freedom, expected_counts = chi2_contingency(crosstab)

print(f"\n category1:{category1},category2:{category2}(male passenger)")

print(f"chi square{chi2:.2f}, pvalue:{pval:.4f},degree of freedom: {degree_of_freedom}")

if pval < 0.05:
    print("reject null hypothesis : there's significant association b/w class and on board service(p < 0.05)")
else:
    print("fail to reject null hypothesis : might be independent class and on board service(p < 0.05)")