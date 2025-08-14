# Documentation Generation

## Index

- `_static`: Contains images, css, js for Sphinx to look at
- `_templates`: Contains html layout for custom Sphinx design
- `build`: Contains the generated documentation
- `devel`: Contains `*.rst` files for the Developer's Guide
- `examples`: DIPY application showcases. Add any tutorial here
- `examples_built`: Keep it empty. Only for example generation
- `release_notes`: Contains all API changes / PRs, issues resolved for a specific release
- `sphinxext`: Sphinx custom plugins
- `theory`: Diffusion theory + FAQ files
- `tools`: Scripts to generate some parts of the documentation, like the API

## Doc generation steps:

### Installing requirements

```bash
$ pip install -U -r doc-requirements.txt
```

### Generate all the Documentation

The recommended way is via `spin`, which supports selective builds:

```bash
# Full build with all examples
spin docs

# Skip example execution (fast, RST/API only)
spin docs --no-plot

# Build only one tutorial (runs its code)
spin docs reconst_csa

# Build only one tutorial without running it
spin docs reconst_csa --no-plot

# Build multiple tutorials
spin docs reconst_csa reconst_dti

# Clean before building
spin docs --clean
```

Tutorial names are matched as regex substrings against each example file path,
so `reconst_csa` matches any script whose path contains that string.  Multiple
names are joined with `|` (union match).

Alternatively you can call `make` directly:

#### Under Linux and macOS

```bash
$ make -C . clean && make -C . html
```

#### Under Windows

```bash
$ ./make.bat clean
$ ./make.bat html
```
