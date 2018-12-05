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
- EDA: framed the problem and leveraged some features by thinking out of the box.
- Feature-engineered a new dataset more suitable to the task at hand and applied relevant models.
- My pipeline involved one-hot encoding, normalization, multicollinearity assessment, logistic regression with grid search and XGBoost with early stopping.
- Achieved 81.5% accuracy on the test set.
