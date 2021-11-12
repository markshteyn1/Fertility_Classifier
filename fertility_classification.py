import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import os


def numerize_data(dataset):
    for c in dataset.columns:
        col = dataset[c].tolist()
        if all(isinstance(i, (float, int)) for i in col):
            continue

        values = Counter(col)
        for i in range(len(col)):
            counter = 0
            for v in values:
                if col[i] == v:
                    col[i] = counter
                counter += 1
        dataset[c] = col
    return dataset


def translate(word, og_list, replace_list):
    for i in range(len(replace_list)):
        word = word.replace(og_list[i], str(replace_list[i]))
    try:
        checker_var = float(word)
        return word
    except ValueError:
        print("Please Enter the information correctly...\tCapitalization does not matter... but spelling does.")
        return translate(input().upper(), og_list, replace_list)


def gather_user_data():
    os.system('cls')
    print("What is the current season?")
    season = translate(input().upper(), ['WINTER', 'SPRING', 'SUMMER', 'FALL'], [-1, -0.33, 0.33, 1])

    print("How old are you?")
    age = float(input())

    print("Have you had any of the following diseases: Chicken Pox, Measles, Mumps, Polio? (Yes or No)")
    diseases = translate(input().upper(), ['YES', 'NO'], [0, 1])

    print("Have you experienced any serious trauma?")
    trauma = translate(input().upper(), ['YES', 'NO'], [0, 1])

    print("Have you ever gone through surgery? (Recently, A while ago, Never)")
    surgery = translate(input().upper(), ['RECENTLY', 'A WHILE AGO', 'NEVER'], [-1, 0, 1])

    print("Frequency of Alcohol Consumption? (Several Times a Day, Everyday, Several Times a week, once a week, "
          "occasionally/never)")
    alcohol = translate(input().upper(),
                        ['SEVERAL TIMES A DAY', 'EVERYDAY', 'SEVERAL TIMES A WEEK', 'ONCE A WEEK', 'OCCASIONALLY',
                         'NEVER', 'OCCASIONALLY/NEVER'], [0, 0, 0, 1, 1, 1, 1])
    print("How often do you smoke? (Never, Occasionally, daily)")
    smoking_habits = translate(input().upper(), ['NEVER', 'OCCASIONALLY', 'DAILY'], [-1, 0, 1])

    print("How many hours do you spend sitting per day? (0-16)")
    hrs = float(input())
    hours_sitting = 0 if hrs < 7 else 1

    datapoint = [season, age, diseases, trauma, surgery, alcohol, smoking_habits, hours_sitting]
    for i in range(len(datapoint)):
        datapoint[i] = float(datapoint[i])
    return datapoint

def create_tree():
    csv = pd.read_csv("fertility.csv")
    data = csv[
        ['season', 'age', 'childhood diseases', 'trauma', 'surgery', 'frequency of alcohol consumption',
         'smoking habit', 'number of hours spent sitting per day']]
    data = numerize_data(data)
    labels = numerize_data(csv[['diagnosis']])
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels)

    max_score = 0
    tree = DecisionTreeClassifier()
    for i in range(1, 100):
        tree_i = DecisionTreeClassifier(max_depth=i)
        tree_i.fit(train_data, train_labels)
        score = tree_i.score(test_data, test_labels)
        if max_score < score:
            max_score = score
            tree = tree_i
    return tree
 