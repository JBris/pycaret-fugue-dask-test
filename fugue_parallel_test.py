from pycaret.classification import *

def main():
    # load dataset
    from pycaret.datasets import get_data
    diabetes = get_data('income')

    # init setup
    
    clf1 = setup(data = diabetes, target = 'sex', n_jobs = 1)

    # import parallel back-end
    from pycaret.parallel import FugueBackend

    # compare models
    best = compare_models(parallel = FugueBackend("dask"))

if __name__ == "__main__":
    main()