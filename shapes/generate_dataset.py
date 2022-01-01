import os
import argparse

import h5py
import imageio as imio
import tqdm


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Generates Shapes3D split from the official 3dshapes.h5 file '
            'and A.txt and B.txt split files.'))
    parser.add_argument('--h5_path',  type=str, required=True,
                        help='a path to the 3dshapes.h5 file.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Folder to store A/ and B/ image folders.')
    parser.add_argument('--split_file_dir', type=str, required=True,
                        help='Folder containing A.txt and B.txt files.')
    parser.add_argument('--take_first', type=int, default=None,
                        help='Generate only first X images (for testing).')
    args = parser.parse_args()

    fh5 = h5py.File(args.h5_path, 'r')
    dom_fns = {}
    for domain in ['A', 'B']:
        os.makedirs(os.path.join(args.output_folder, domain), exist_ok=True)
        with open(os.path.join(args.split_file_dir, domain + '.txt')) as f:
            dom_fns[domain] = [x.strip() for x in f.readlines()]

    shared = set(dom_fns['A']).intersection(set(dom_fns['B']))

    for domain, fns in dom_fns.items():
        for i, fn in enumerate(tqdm.tqdm(set(fns) - shared)):
            if args.take_first is not None and i >= args.take_first:
                break
            idx = int(fn.split('.')[0])
            img = fh5['images'][idx]
            imio.imsave(os.path.join(args.output_folder, domain, fn), img)

if __name__ == "__main__":
    main()