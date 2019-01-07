# Trivago Case Study
Data science case study targeting hotel entry click rate, based on real-world [Trivago](https://www.trivago.com/ "Trivago Homepage") data. The report includes an in-depth description of my work and thought process. This project has been carried out in one week (time allotted for the challenge).

# Prerequisites
I used Conda 4.5.11 with Python 3.6.5 on a machine implementing Windows 10 64-bit to set up my PyData stack. I worked within a new environment defined and activated via terminal as in the following:<br/>
`> conda create --name trivago python=3.6.5 numpy pandas matplotlib seaborn scikit-learn py-xgboost`<br/>
`> conda activate trivago`

The "main.py" script executes the whole workflow and saves the output in a logfile.<br/>
`> python main.py`

# Highlights
- EDA: characterized unknown feature by means of violin plots.
- EDA: leveraged log-discretized pairplots with multiple KDE for regression analysis.
- My pipeline involved stratified splitting, one-hot encoding, normalization, grid/randomized search, linear and polynomial regression with shrinkage, SVR, random forest and XGBoost with early stopping.
- Significantly improved results by applying weighted oversampling strategy.
- My best model achieved a weighted MSE of 0.91 and performed statistically significantly better than the naïve baseline on the test set (2.25).
