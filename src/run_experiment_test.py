import wandb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train(config=None):
   # Initialize a new wandb run
    with wandb.init(config=config):
        #If called by wandb.agent, as below,
        #this config will be set by Sweep Controller
        config = wandb.config
        x = np.random.randn(config["n_samples"], 1).astype(np.float32)
        y = (x + np.random.randn(config["n_samples"], 1)).ravel()
        y = y > np.median(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        # Create a new Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=config["n_estimators"])
        rf.fit(x_train, y_train)
        # Log the model
        #wandb.log({"model": rf})
        # Log the test set accuracy
        wandb.log({"test_accuracy": rf.score(x_test, y_test)})

if __name__ == "__main__":
    #config = {"n_samples": 10, "n_estimators": 10}
    #train(config)
    train()