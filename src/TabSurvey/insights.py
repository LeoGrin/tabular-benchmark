from TabSurvey.models import str2model
from utils.load_data import load_data
from utils.parser import get_given_parameters_parser


def get_size(args):
    # print("Calculating model size...")
    X, y = load_data(args)

    # Some models need to be fitted for one step before the size can be calculated
    args.epochs = 1

    model_name = str2model(args.model_name)

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)

    model.fit(X, y, X, y)

    try:
        size = model.get_model_size()
        print(f"Total Trainable Parameters of %s: %.3fK" % (args.model_name, size/1000))
    except NotImplementedError:
        print("Size calculation not implemented for " + args.model_name)


if __name__ == "__main__":
    parser = get_given_parameters_parser()
    arguments = parser.parse_args()
    print(arguments)

    get_size(arguments)
