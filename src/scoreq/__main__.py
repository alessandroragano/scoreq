import click
from scoreq import Scoreq

@click.command()
@click.argument('data_domain', type=click.Choice(['natural', 'synthetic']))
@click.argument('mode', type=click.Choice(['nr', 'ref']))
@click.argument('test_path', type=click.Path(exists=True))
@click.option('--ref_path', type=click.Path(exists=True), help='Path to the reference audio file (required in "ref" mode)')
@click.option('--device', type=str, default=None, help='Specify device, cuda or cpu. Automatically set cuda if None and GPU is detected')
def main(data_domain, mode, test_path, device=None, ref_path=None):
    """Audio quality assessment using SCOREQ"""

    # Check if 'ref_path' is provided in 'ref' mode
    if mode == 'ref' and ref_path is None:
        raise click.UsageError("Error: --ref_path is required in 'ref' mode")

    # If 'mode' is 'nr', set 'ref_path' to None explicitly
    if mode == 'nr':
        ref_path = None
    
    scoreq_model = Scoreq(device, data_domain, mode)
    scoreq_score = scoreq_model.predict(test_path, ref_path)

if __name__ == '__main__':
    main()