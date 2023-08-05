
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from src.data.datamodules import ContrailsDatamodule


def main(args):
    pattern_ind = re.compile(r'pl_(?P<loss_name>.+), \[\'(?P<path>.+)\'\]\: (?P<loss_value>\d\.\d+)')
    pattern_tot = re.compile(r'total pl, \[\'(?P<path>.+)\'\]\: (?P<loss_value>\d\.\d+)')
    
    # Extract the data
    with open(args.file, 'r') as f:
        content = f.read()
    data = \
        [m.groupdict() for m in pattern_ind.finditer(content)] + \
        [m.groupdict() for m in pattern_tot.finditer(content)]

    # Convert the data to pandas DataFrame
    df = pd.DataFrame(data)

    # Convert the data type
    df['loss_value'] = df['loss_value'].astype(float)

    # Fill NaN
    df['loss_name'].fillna('total', inplace=True)

    # Pivot the DataFrame
    df = df.pivot(index='path', columns='loss_name', values='loss_value')

    # Sort by 'dice' loss
    df.sort_values(by='dice', inplace=True)

    # Plot distplot
    fig, _ = plt.subplots(1, 1, figsize=(5, 5))
    sns.histplot(df['dice'], kde=False, bins=20, ax=fig.axes[0])
    fig.savefig('train_loss_hist_dice.png', bbox_inches='tight')

    fig, _ = plt.subplots(1, 1, figsize=(5, 5))
    sns.histplot(df['bce'], kde=False, bins=20, ax=fig.axes[0])
    fig.savefig('train_loss_hist_bce.png', bbox_inches='tight')

    # Print some stats

    # Count samplew with max loss
    print('Number of samples with max dice loss: {}'.format((df['dice'] >= args.threshold).sum()))

    # Print all samples with max loss
    print('Samples with max dice loss:')
    print(df[df['dice'] >= args.threshold])

    # Save to csv
    df[df['dice'] >= args.threshold].to_csv('train_loss_per_sample.csv')

    # Imshow all the samples with max loss and save to pdf

    # Get config, create datamodule
    with open('/workspace/contrails/run/configs/common.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['data']['init_args']['data_dirs'] = [
        Path(d) for d in config['data']['init_args']['data_dirs']
    ]
    config['data']['init_args']['cache_dir'] = Path(config['data']['init_args']['cache_dir'])
    
    datamodule = ContrailsDatamodule(**config['data']['init_args'])
    datamodule.setup()
    datamodule.train_dataset.transform = None
    
    # Get the samples with max loss
    with PdfPages('samples_with_max_loss.pdf') as pdf:
        pathes = set(df[df['dice'] >= args.threshold].index.tolist())
        for item in tqdm(datamodule.train_dataset):
            if item['path'] not in pathes:
                continue
        
            # Plot the image
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(item['image'])
            axes[0].axis('off')
            axes[0].set_title('Image')
            axes[1].imshow(item['mask'])            
            axes[1].axis('off')
            axes[1].set_title('Mask')
            # Prediction path is './preds/{record_id}_4_xkgWQAQleb.png'
            record_id = item['path'].split('/')[-1]
            pred_path = f'./preds/{record_id}_4_3CG6AevzBY.png'
            axes[2].imshow(plt.imread(pred_path))
            axes[2].axis('off')
            axes[2].set_title('Prediction')
            pdf.savefig(fig, bbox_inches='tight')

            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--threshold', type=float, default=0.1)
    args = parser.parse_args()

    main(args)
