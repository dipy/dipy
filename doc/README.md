# Documentation Generation

## Index

- `_static`: Contains images, css, js for Sphinx to look at
- `_templates`: Contains html layout for custom Sphinx design
- `build`: Contains the generated documentation
- `devel`: Contains `*.rst` files for the Developer's Guide
- `examples`: DIPY application showcases. Add any tutorial here
- `examples_built`: Keep it empty. Only for example generation
- `releases_notes`: Contains all API changes / PRs, issues resolved for a specific release
- `sphinxext`: Sphinx custom plugins
- `theory`: Diffusion theory + FAQ files
- `tools`: Scripts to generate some parts of the documentation, like the API

## Doc generation steps:

### Installing requirements

```bash
$ pip install -U -r doc-requirements.txt
```

### Generate all the Documentation

#### Under Linux and OSX

```bash
$ make -C . clean && make -C . html
```

#### Under Windows

```bash
$ ./make.bat clean
$ ./make.bat html
```
