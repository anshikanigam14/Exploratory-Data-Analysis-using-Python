import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from scipy.stats import stats
import copy
from scipy.stats import chi2_contingency

insurance_df = pd.read_csv('insurance-data.csv')
print(insurance_df.head())
print(insurance_df.info())

# Check for Missing values

print(insurance_df.isna().apply(pd.value_counts))

# Check for Outliers

plt.figure(figsize=(20,15))
plt.subplot(3,3,1)
sns.boxplot(x = insurance_df['age'], color='red')

plt.subplot(3,3,2)
sns.boxplot(x = insurance_df['bmi'], color='blue')

plt.subplot(3,3,3)
sns.boxplot(x = insurance_df['charges'], color='yellow')

plt.show()
plt.close()

print(insurance_df.describe())
# Analysis:
# -All the statistics seem reasonable.
# -Age column data looks representative of the true age distribution of the adult population with 39 as mean
# -Children column: Few people have more than 2 children (75% of the people have 2 or less children)
# -The claimed amount is highly skewed as most people would require basic medi-care and only few would suffer from diseases which cost more.

########################################################################################
# Create Visual methods to analyse the data

plt.figure(figsize=(20,15))
plt.subplot(3,3,1)
plt.hist(insurance_df['bmi'], color='lightblue', edgecolor = 'black', alpha = 0.7)
plt.xlabel('BMI')

plt.subplot(3,3,2)
plt.hist(insurance_df['age'], color='lightblue', edgecolor = 'black', alpha = 0.7)
plt.xlabel('Age')

plt.subplot(3,3,3)
plt.hist(insurance_df['charges'], color='lightblue', edgecolor = 'black', alpha = 0.7)
plt.xlabel('Charges')
plt.show()
plt.close()

# Check the Skewness of the variables

Skewness = Skewness = pd.DataFrame([stats.skew(insurance_df['bmi']),stats.skew(insurance_df['age']), stats.skew(insurance_df['charges'])], index= ['BMI', 'AGE', 'CHARGES'], columns =['Skewness'])
print(Skewness)

# Skewness of BMI is very low
# Age is uniformly distributed and thus not skewed
# Skewness of Charges is > 1 and thus is highly skewed
# If skewness is less than -1 or greater than 1, the distribution is highly skewed.
# If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.
# If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.

##################################################################################
# Data Visualisation

plt.figure(figsize= (15,10))

x = insurance_df['smoker'].value_counts().index
y = [insurance_df['smoker'].value_counts()[i] for i in x]

plt.subplot(4, 2 , 1)
plt.bar(x,y,align = 'center', color = 'black', edgecolor = 'black', alpha = 0.7)
plt.xlabel('Smoker')
plt.xlabel('Count')
plt.title('Smoker Distribution')

x1 = insurance_df['sex'].value_counts().index
y1 = [insurance_df['sex'].value_counts()[j] for j in x1]

plt.subplot(4, 2 , 2)
plt.bar(x1,y1,align = 'center', color = 'orange', edgecolor = 'black', alpha = 0.7)
plt.xlabel('Gender')
plt.xlabel('Count')
plt.title('Gender Distribution')

x2 = insurance_df['region'].value_counts().index
y2 = [insurance_df['region'].value_counts()[k] for k in x2]

plt.subplot(4, 2 , 3)
plt.bar(x2,y2,align = 'center', color = 'red', edgecolor = 'black', alpha = 0.7)
plt.xlabel('Region')
plt.xlabel('Count')
plt.title('Region Distribution')


x3 = insurance_df['children'].value_counts().index
y3 = [insurance_df['children'].value_counts()[l] for l in x3]

plt.subplot(4, 2 , 4)
plt.bar(x3,y3,align = 'center', color = 'green', edgecolor = 'black', alpha = 0.7)
plt.xlabel('Children')
plt.xlabel('Count')
plt.title('Children Distribution')

plt.show()
plt.close()

# ANALYSIS:
# -Analysis shows that there are more non-smokers than smokers.
# -Instances are distributed evenly across all regions.
# -Gender is also distributed evenly.
# -Most instances have less than three children and very few have 4/5.

################################################################################
# Label encoding before doing a pairplot because pairplot ignores strings

insurance_df_encoded = copy.deepcopy(insurance_df)
insurance_df_encoded.loc[:, ['sex','smoker', 'region']] = insurance_df_encoded.loc[:, ['sex','smoker', 'region']].apply(LabelEncoder().fit_transform)
sns.pairplot(insurance_df_encoded)

plt.show()
plt.close()

# -There is an obvious coorelation between charges and smoker.
# -Looks like smokers claimed more money than non-smokers
# -Age and charges has an interesting pattern, older people are charged more than younger people

# Analysing trends, patterns, and relationships

print('Do charges of people who smoke differ from those who dont?')
print(insurance_df['smoker'].value_counts())

plt.figure(figsize=(8,6))
sns.scatterplot(insurance_df['age'], insurance_df['charges'], hue = insurance_df['smoker'],palette=['red','black'] )
plt.title('Difference between charges of a smoker and a non-smoker')
plt.show()
plt.close()
# SHows that smokers are charged more than non smokers


plt.figure(figsize=(8,6))
sns.scatterplot(insurance_df['age'], insurance_df['charges'], hue = insurance_df['sex'],palette=['red','black'] )
plt.title('Difference between charges of a smoker and a non-smoker')
plt.show()
plt.close()
# No apparent relationship between gender and charges

###########################################################################
# 1. T-test to check the dependancy of smoking and charges
h0 = "Charges of smoker and non-smoker are the same"
h1 = "Charges of smoker and non-smoker are not the same"

# selecting charges corressponding to smokers as an array
x = np.array(insurance_df[insurance_df['smoker']  == 'yes']['charges'])
# selecting charges corressponding to non-smokers as an array
y = np.array(insurance_df[insurance_df['smoker']  == 'no']['charges'])

t , p_value = stats.ttest_ind(x, y, axis = 0)

# For significance level of 5%

if p_value < 0.05:
    print(f'{h1} as the p_value {p_value.__round__(3)} < 0.05')
else:
    print(f'{h0} as the p_value {p_value.__round__(3)} > 0.05')

print('Analysis: Charges of smoker and non-smoker are not the same as p_value < 0.05')
#############################################################################

# 2.  BMI of males differ from females significantly

insurance_df['sex'].value_counts()
# T-test to check the dependancy of BMI and sex
h0 = "BMI of male and females are the same"
h1 = "BMI of male and females are not the same"

# selecting bmi corressponding to male as an array
x1 = np.array(insurance_df[insurance_df['sex']  == 'male']['bmi'])
# selecting bmi corressponding to female as an array
y1 = np.array(insurance_df[insurance_df['sex']  == 'female']['bmi'])

t , p_value = stats.ttest_ind(x1, y1, axis = 0)

# For significance level of 5%

if p_value < 0.05:
    print(f'{h1} as the p_value {p_value.__round__(3)} < 0.05')
else:
    print(f'{h0} as the p_value {p_value.__round__(3)} > 0.05')

print('Analysis: BMI of male and females are the same as p_value > 0.05')

#############################################################################

# Proportion of smokers is different in different genders
# Chi Square test:
h0 = 'Gender has no effect on smoking'
h1 = 'Gender has an effect on smoking'
crosstab = pd.crosstab(insurance_df['sex'], insurance_df['smoker'])
chi, p_value, dof, expected = chi2_contingency(crosstab)

# interpret p-value
if p_value < 0.05:
    print(f'{h1} as the p_value {p_value.__round__(3)} < 0.05')
else:
    print(f'{h0} as the p_value {p_value.__round__(3)} > 0.05')

print(crosstab)

#################################################################

# Region has no effect on smoking habits
# Chi Square test:
h0 = 'Region has no effect on smoking'
h1 = 'Region has an effect on smoking'
crosstab = pd.crosstab(insurance_df['region'], insurance_df['smoker'])
chi, p_value, dof, expected = chi2_contingency(crosstab)

# interpret p-value
if p_value < 0.05:
    print(f'{h1} as the p_value {p_value.__round__(3)} < 0.05')
else:
    print(f'{h0} as the p_value {p_value.__round__(3)} > 0.05')

print(crosstab)

###################################################################


