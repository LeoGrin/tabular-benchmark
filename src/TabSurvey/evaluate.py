from utils.io_utils import get_predictions_from_file, save_results_to_file
from utils.parser import get_given_parameters_parser
from utils.scorer import get_scorer

import numpy as np


def main(args):
    print("Evaluate model " + args.model_name)

    predictions = get_predictions_from_file(args)
    scorer = get_scorer(args)

    for pred in predictions:
        # [:,0] is the truth and [:,1:] are the prediction probabilities

        truth = pred[:, 0]
        out = pred[:, 1:]
        pred_label = np.argmax(out, axis=1)

        scorer.eval(truth, pred_label, out)

    result = scorer.get_results()
    print(result)

    save_results_to_file(args, result)


if __name__ == "__main__":

    # Also load the best parameters
    parser = get_given_parameters_parser()
    arguments = parser.parse_args()
    print(arguments)

    main(arguments)
