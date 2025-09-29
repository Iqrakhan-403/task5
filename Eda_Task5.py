import pandas as pd
import matplotlib.pyplot as plt

def load_data(url):
    return pd.read_csv(url)

def explore_data(df):
    print(df.head(), "\n")
    print(df.info(), "\n")
    print(df.describe(), "\n")
    print("Missing Values:\n", df.isnull().sum(), "\n")
    for col in ['Sex', 'Pclass']:
        print(f"{col} Value Counts:\n", df[col].value_counts(), "\n")

def plot_hist(df, col, bins=20, color='skyblue', xlabel=None, ylabel='Count', title=None):
    plt.hist(df[col].dropna(), bins=bins, color=color)
    plt.xlabel(xlabel if xlabel else col)
    plt.ylabel(ylabel)
    plt.title(title if title else col)
    plt.show()

def plot_bar(df, col, color_list=None, title=None, ylabel=None):
    df[col].value_counts().sort_index().plot(kind='bar', color=color_list)
    plt.title(title if title else col)
    if ylabel: plt.ylabel(ylabel)
    plt.show()

def plot_grouped_bar(df, group_col, target_col='Survived', color_list=None, title=None, ylabel='Survival Rate'):
    df.groupby(group_col)[target_col].mean().plot(kind='bar', color=color_list)
    plt.title(title if title else f'{target_col} by {group_col}')
    plt.ylabel(ylabel)
    plt.show()

def plot_scatter(df, x, y, hue_col='Survived', alpha=0.5):
    colors = df[hue_col].map({0:'red', 1:'green'})
    plt.scatter(df[x], df[y], c=colors, alpha=alpha)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y} (Red=Not Survived, Green=Survived)')
    plt.show()

def plot_corr_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    print("Correlation Matrix:\n", corr, "\n")
    plt.matshow(corr, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title('Correlation Heatmap', pad=20)
    plt.show()

def summarize_findings():
    print("""
Summary of Findings:
1. Majority of passengers were male and in 3rd class.
2. Women had higher survival rates than men.
3. First-class passengers had better survival chances than 2nd and 3rd class.
4. Younger adults (20-40 years) formed the largest age group.
5. Fare positively correlates with Pclass and survival.
6. Family size could be explored further as a survival factor.
""")

# ==========================
# Main Execution
# ==========================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = load_data(url)

explore_data(df)

plot_hist(df, 'Age', bins=20, color='skyblue', title='Age Distribution')
plot_bar(df, 'Sex', color_list=['orange','green'], title='Count by Sex')
plot_bar(df, 'Pclass', color_list=['purple','blue','cyan'], title='Count by Passenger Class')

plot_grouped_bar(df, 'Sex', color_list=['pink','lightblue'], title='Survival Rate by Sex')
plot_grouped_bar(df, 'Pclass', color_list=['gold','silver','brown'], title='Survival Rate by Passenger Class')

plot_scatter(df, 'Age', 'Fare')

numeric_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
plot_corr_heatmap(df, numeric_cols)

summarize_findings()
