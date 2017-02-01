import click

from dpt.tools import config as cfg
from dpt.framework import KerasFramework, TensorflowFramework, TensorflowStdFramework

FRAMEWORKS = {
    'keras': KerasFramework,
    'tf': TensorflowFramework,
    'tfr': TensorflowStdFramework,
}


@click.command()
@click.argument('framework')
@click.argument('mode')
def main(framework, mode):

    fw = FRAMEWORKS[framework](cfg)
    fw.execute(mode)

    if framework in ['tf', 'tfr']:
        fw.shutdown()  # (Sgmt fault) Bug in tensorflow 1.0.0rc0 ?


if __name__ == '__main__':
    main()
