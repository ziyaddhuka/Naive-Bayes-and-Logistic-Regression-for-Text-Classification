# Code for NB and LR
Implementation of Bag of Words model, Bernoulli model, MultiNomial and Discrete Naive Bayes Model
Implementation of MCAP and SGD Logistic Regression

Paper referred for Naive Bayes part
http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf


# If you are getting errors or not getting the output in PART 1 then try PART 2

## IMP: Please give the path till the dataset folder. The code takes the data from the train and test folder inside the path 
# -------------PART 1-------------
> pip install -r requirements.txt
## Run the files with filepath as arguments
## for e.g.
> python sgd_classifier.py data/hw1/

## You can specify the seed for the sgd_classifier and logistic_regression after the filepath for e.g. if we want to have seed as 100
> python sgd_classifier.py data/hw1/ 100

# -------------PART 2-------------
# Steps to run the code... commands are tested in linux.. you can apply alternative commands for windows/MacOS
## Step 1 creating a virtual environment to run the code so that it does not conflicts with other instaled packages on the machine
> python3 -m venv my_env
## Step 2 if the above gives error then make sure your python version is 3.6 or above and install the venv package. If no error move to Step 3
	### for linux and MacOS
	> python3 -m pip install --user virtualenv
	### for windows
	> py -m pip install --user virtualenv

## Step 3 activate the environment
> source my_env/bin/activate
> pip install --upgrade pip 

## Step 2 use requirements.txt file to install required packages
> pip install -r requirements.txt

## Run the files with filepath as arguments
## for e.g.
> python sgd_classifier.py data/hw1/

## after running the above you'll see the output printed on the screen and also written to output.txt in the current working folder

### once done with grading of the code you can deactivate the environment and delete it
> deactivate
> rm -r my_env
