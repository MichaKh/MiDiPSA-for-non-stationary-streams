import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

REPORT_DIR = 'C:\Users\micha\PycharmProjects\MiDiPSA\Incremental Evaluation\\'


class EvaluationPlotter:

    def __init__(self, evaluation_dir):
        self.evaluation_dir = evaluation_dir

    def read_evaluation_files(self, eval_dir, dataset, k=None, eps=None, l=None, estimator=None):
        i = 0
        combined_df = pd.DataFrame()
        if os.path.exists(eval_dir):
            filter_match = '{0}{1}_{2}_{3}_{4}_{5}.csv'.format(eval_dir, dataset, k, eps, l, estimator)
            for f in glob.glob(filter_match):
                i += 1
                dataset_name, k, eps, l, estimator = f.split('_')
                df = pd.read_csv(f)
                if not df.columns[0] in combined_df:
                    combined_df['time'] = df.iloc[:, 0].values
                combined_df[k] = df.iloc[:, -1].values
        return combined_df

    def plot(self, df, dataset_name, k, eps, l, estimator, yscale):
        sampling_freq = 100
        for i in range(1, len(df.columns)):
            x_val = y_val = []
            x_val = df.iloc[:, 0]
            y_val = df.iloc[:, i]

            filtered = lowess(y_val, x_val, is_sorted=True, frac=float(sampling_freq) / len(x_val), it=3)
            plt.plot(filtered[:, 0], filtered[:, 1], linewidth=1, label='k={0}'.format(df.columns[i]))
            # plt.plot(filtered[np.argmax(filtered[:, 1])][0], max(filtered[:, 1]), 'x')
            # plt.plot(x_val, y_val)
            # plt.plot(x_val, y_val, 'or')
        if yscale and yscale == 'log':
            plt.yscale('log', basey=10)
        plt.ylabel(estimator.split('.')[0])
        plt.xlabel(df.columns[0])
        plt.title('{0}: (eps={2})'.format(dataset_name.split('\\')[-1], k, eps, l))
        plt.legend()
        plt.grid(True, lw=0.5, ls='--', c='.75')
        plt.margins(0.005)
        plt.show()
        # plt.savefig('books_read', format='pdf')

    def plot_evaluation(self,
                        eval_dir,
                        eval_file_path=None,
                        group_by='k',
                        estimator='MSE Info Loss',
                        dataset=None,
                        k=50,
                        eps=0.01,
                        l=2,
                        yscale=None):
        combined_df = pd.DataFrame()
        if not os.path.exists(eval_dir) or os.listdir(eval_dir) == []:
            print("Could not locate evaluation files in specified directory: {0}".format(eval_dir))
        else:
            if group_by == 'k':
                combined_df = self.read_evaluation_files(eval_dir, dataset, k='*', eps=eps, l=l, estimator=estimator)
            elif group_by == 'eps':
                combined_df = self.read_evaluation_files(eval_dir, dataset, k=k, eps='*', l=l, estimator=estimator)
            elif group_by == 'l':
                combined_df = self.read_evaluation_files(eval_dir, dataset, k=k, eps='*', l=l, estimator=estimator)
            self.plot(combined_df, dataset_name=dataset, k=k, eps=eps, l=l, estimator=estimator, yscale=yscale)

