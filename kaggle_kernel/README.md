The sole purpose of this repository is to use it in the kaggle competition - Jigsaw Agile Community Rules Classification

Having a repository will allow faster development locally and push code as a kernel/dataset for usage in kaggle notebooks

## Creating your own package (Optional)

You can push the jupyter notebook jigsaw_rules.ipynb here as kernel to clone and save the repo for further use

Configure kaggle api in your system and update the kernel-metadta.json to your profile in this directory

You can use the following command to update from your remote changes (github) to kaggle

    kaggle kernels push -p kaggle_kernel

## Usage of package

Add the following kernel notebook (or your own) as an input to your main notebook : https://www.kaggle.com/code/weedoo/jigsaw-rules?scriptVersionId=265275295

This package should be updated through the api to keep the latest remote version

Copy the package to working directory

    !cp -r /kaggle/input/jigsaw-rules/jigsaw_kaggle /kaggle/working/jigsaw_kaggle

Change directory to the package source

    cd /kaggle/working/jigsaw_kaggle/src

Additionally, add the package source to sys path

    import sys
    package_parent_dir = "/kaggle/working/jigsaw_kaggle/src"
    sys.path.insert(0, package_parent_dir)

Done! you can use the jigsaw_rules as a package now!
We do it this way to apply instant changes if necessary.

## Making Instant Changes

Use %%writefile to make changes on the go, useful for iterative experiments or debugging before making remote changes to the repo

    %%writefile jigsaw_rules/configs.py
    # Paste the code here and make changes on top of it

All the main modules in this repo are provided as scripts and can be started with a subprcess through the kaggle notebook

    !python -m jigsaw_rules.any_module

Using the subprocess will make the OS handle the clean ups like clearning up the GPU vRAM or any garbage collection related to that module.

## Warnings

It is advisable to not import jigsaw_rules directly into the jupyter notebook as it caches the package and makes any instant changes after that redundant
