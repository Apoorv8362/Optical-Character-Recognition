#!/usr/bin/env python
# coding: utf-8

# # Question 2. Modelling Stress Scenarios

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


# In[24]:


def read_csv_to_dataframe(file_path):

    try:
        # Basic read - assumes first row contains headers
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None 


# In[25]:


file_path = "behltreu_ahwm5_data.csv"  # Replace with your CSV file path
merged_df = read_csv_to_dataframe(file_path)


# In[33]:


# Show correlation if both columns have data

if not merged_df['bond_price'].isna().all() and not merged_df['future_price'].isna().all():
    correlation = merged_df['bond_price'].corr(merged_df['future_price'])
    print(f"\nCorrelation between bond and future prices: {correlation:.4f}")


# In[34]:


def compute_nbd_stress(df, variable, nbd):

    stress = df[variable]/df[variable].shift(nbd) - 1
    
    return stress


# In[35]:


def model_and_plot(df, nbd, advanced = False, poly_deg = 2):
    
    df = df.copy()
    df['bond_stress'] = compute_nbd_stress(df, 'bond_price', nbd)
    df['future_stress'] = compute_nbd_stress(df, 'future_price', nbd)
    
    df = df.dropna()
    
    X = df[['bond_stress']].values
    y = df[['future_stress']].values
    
    df = df.dropna(subset = ['bond_stress', 'future_stress'])
    
    #Linear Regression
    linreg = LinearRegression()
    linreg.fit(X,y)
    y_pred_lin = linreg.predict(X)
    
    print(f"\n {nbd}BD Linear Model")
    
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['bond_stress'], y=df['future_stress'], label='Data Points')
    plt.plot(df['bond_stress'], y_pred_lin, color='red', label='Linear Fit', linewidth=2)
    plt.title(f'{nbd}BD Stress: Bond vs Future')
    plt.xlabel('Bond Stress')
    plt.ylabel('Future Stress')
    plt.legend()
    plt.grid(True)
    plt.show()

    if advanced:
        #Polynomial Regression
        poly = PolynomialFeatures(degree=poly_deg, include_bias=False)
        X_poly = poly.fit_transform(X)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, y)
        y_pred_poly = poly_reg.predict(X_poly)

        print(f"\n---- {nbd}BD Polynomial Degree {poly_deg} Model ----")
        print(f"R^2: {r2_score(y, y_pred_poly):.4f}")

        #Ridge Regression
        ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)
        ridge.fit(X_poly, y)
        y_pred_ridge = ridge.predict(X_poly)

        print(f"\n---- {nbd}BD Ridge Regression ----")
        print(f"Alpha: {ridge.alpha_:.6f}")
        print(f"R^2: {r2_score(y, y_pred_ridge):.4f}")

        # Plot polynomial vs linear comparison
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df['bond_stress'], y=df['future_stress'], label='Data Points')
        plt.plot(df['bond_stress'], y_pred_lin, color='red', label='Linear Fit', linewidth=2)
        plt.scatter(df['bond_stress'], y_pred_poly, color='green', s=10, label=f'Polynomial Degree {poly_deg}')
        plt.scatter(df['bond_stress'], y_pred_ridge, color='orange', s=10, label='Ridge Regression')
        plt.title(f'{nbd}BD Stress Comparison: Bond vs Future')
        plt.xlabel('Bond Stress')
        plt.ylabel('Future Stress')
        plt.legend()
        plt.grid(True)
        plt.show()


# # 3BD Stress Modelling 

# In[36]:


model_and_plot(merged_df, nbd=3, advanced=True, poly_deg=2)


# # 7BD Stress Modelling
# 

# In[37]:


model_and_plot(merged_df, nbd=7, advanced=True, poly_deg=2)


# # 14BD Stress Modelling
# 

# In[31]:


model_and_plot(merged_df, nbd=14, advanced=True, poly_deg=2)

