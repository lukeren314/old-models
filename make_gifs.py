from PIL import Image
from pathlib import Path
samples_dir = './samples'
output_dir = './gifs'


def _create_necessary_directories(paths: [Path]) -> None:
    for directory in paths:
        if not Path(directory).exists():
            Path(directory).mkdir()


if __name__ == '__main__':
    _create_necessary_directories([samples_dir, output_dir])
    for child in Path(samples_dir).iterdir():
        if child.is_dir():
            images = []
            for image_file in child.iterdir():
                images.append(Image.open(str(image_file)))
            images[0].save(str(Path(output_dir))+'/'+child.name+".gif",
                           save_all=True, append_images=images[1:], optimize=False)
    print('Done!')
