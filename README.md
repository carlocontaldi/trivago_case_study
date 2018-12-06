# Trivago Case Study
Data science case study targeting hotel entry click rate, based on real-world Trivago data.

The report includes an in-depth description of my work and thought process.

This project has been carried out in one week (time allotted for the challenge).

# Prerequisites
I used Conda 4.5.11 with Python 3.6.5 on a machine implementing Windows 10 64-bit to set up my PyData stack. I worked within a new environment defined and activated via terminal as in the following:
> conda create --name trivago python=3.6.5 numpy pandas matplotlib seaborn scikit-learn py-xgboost

> conda activate trivago

The "main.py" script executes the whole workflow and saves the output in a logfile.
> python main.py

# Highlights
- EDA: leveraged log-discretized pairplots with multiple KDE for regression analysis.
- My pipeline involved stratified splitting, one-hot encoding, normalization, grid/randomized search, linear and polynomial regression with shrinkage, SVR, random forest and XGBoost with early stopping.
- Significantly improved results by applying weighted oversampling strategy.
- Achieved a weighted MSE of 0.91, which is statistically significantly better than the random baseline (2.25).
