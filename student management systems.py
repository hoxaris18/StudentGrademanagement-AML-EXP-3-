import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_and_evaluate(x, y, title):

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    equation = f"y = {model.intercept_:.4f}"

    for coef, col in zip(model.coef_, x.columns):
        equation += f" + ({coef:.4f} × {col})"

    print("\n===================================")
    print(title)
    print("===================================")

    print("Regression Equation:")
    print(equation)

    print("\nPerformance")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE :", rmse)
    print("R2 Score :", r2)

    return model, x_test, y_test, predictions, [mae, mse, rmse, r2]


def plot_chart(chart_type, x_data, y_data,
               title, xlabel, ylabel,
               predictions=None):

    plt.figure(figsize=(6,4))

    if chart_type == "scatter_line":

        plt.scatter(x_data,
                    y_data,
                    color="blue",
                    label="Actual")

        plt.plot(x_data,
                 predictions,
                 color="red",
                 linewidth=2,
                 label="Regression Line")

        plt.legend()

    elif chart_type == "actual_vs_pred":

        plt.scatter(x_data,
                    y_data,
                    color="purple")

        plt.plot(
            [x_data.min(), x_data.max()],
            [x_data.min(), x_data.max()],
            "r--"
        )

    elif chart_type == "metrics_bar":

        bars = plt.bar(
            ["MAE","MSE","RMSE","R2"],
            x_data,
            color=["blue","orange","green","red"]
        )

        for bar in bars:

            plt.text(
                bar.get_x()+bar.get_width()/2,
                bar.get_height(),
                f"{bar.get_height():.4f}",
                ha="center",
                va="bottom"
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    plt.show()


data = pd.read_excel("StudentGradeDataSet.csv.xls")

print("\nFirst Five Rows")
print(data.head())

print("\nDataset Shape")
print(data.shape)

print("\nInformation")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())

print("\nStatistics")
print(data.describe())


plt.figure(figsize=(8,6))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()



fig, axes = plt.subplots(2,3,figsize=(16,9))

axes = axes.flatten()

for i,col in enumerate(data.columns):

    axes[i].hist(
    data[col],
    bins=10,
    color="skyblue",
    edgecolor="black",
    alpha=0.7
)

    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

if len(data.columns)<len(axes):
    axes[-1].axis("off")

plt.tight_layout()
plt.show()


fig,axes=plt.subplots(
    1,
    len(data.columns),
    figsize=(15,4)
)

for i,col in enumerate(data.columns):

    sns.boxplot(
        y=data[col],
        ax=axes[i]
    )

    axes[i].set_title(col)

plt.tight_layout()
plt.show()


features=[
    "SEM 1",
    "SEM 2",
    "SEM 3",
    "SEM 4"
]

x_all=data[features]

y=data["SEM 5"]


best_feature=x_all.corrwith(y).abs().idxmax()

print("\nBest Feature:",best_feature)



x_simple=data[[best_feature]]

simple_model,\
x_test_s,\
y_test_s,\
pred_s,\
metrics_s=train_and_evaluate(

    x_simple,
    y,
    "Simple Linear Regression"

)

plot_chart(

    "scatter_line",

    x_test_s,

    y_test_s,

    f"Simple LR ({best_feature})",

    best_feature,

    "SEM 5",

    pred_s

)

plot_chart(

    "metrics_bar",

    metrics_s,

    None,

    "Simple LR Performance",

    "Metrics",

    "Value"

)

multi_model,\
x_test_m,\
y_test_m,\
pred_m,\
metrics_m=train_and_evaluate(

    x_all,

    y,

    "Multiple Linear Regression"

)

plot_chart(

    "actual_vs_pred",

    y_test_m,

    pred_m,

    "Actual vs Predicted",

    "Actual Marks",

    "Predicted Marks"

)

plot_chart(

    "metrics_bar",

    metrics_m,

    None,

    "Multiple LR Performance",

    "Metrics",

    "Value"

)


print("\nPredict SEM 5 Marks")

s1=float(input("Enter SEM 1 Marks : "))
s2=float(input("Enter SEM 2 Marks : "))
s3=float(input("Enter SEM 3 Marks : "))
s4=float(input("Enter SEM 4 Marks : "))

user_input=pd.DataFrame({

    "SEM 1":[s1],

    "SEM 2":[s2],

    "SEM 3":[s3],

    "SEM 4":[s4]

})

prediction=multi_model.predict(user_input)

print("\nPredicted SEM 5 Marks =",round(prediction[0],2))