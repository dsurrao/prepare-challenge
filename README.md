# about
code submission for [PREPARE](https://www.drivendata.org/competitions/300/competition-nih-alzheimers-sdoh-2/page/928/) challenge.

# virtual env
python -m venv venv
source venv/bin/activate     
pip install -r requirements.txt

# data
This program reads data from a `data` directory. It expects the following files:
- train_features.csv
- train_labels.csv
- test_features.csv
- submission_format.csv

It writes it's output to `submission.csv`.

Submission format:
- uid (str): Unique identifier for the individual
- year (int): Year the individual received the score
- composite_score (int): Predicted composite score for the individual in the given year

uid,year,composite_score
aoid,2016,150
afui,2016,275
aqoe,2021,200