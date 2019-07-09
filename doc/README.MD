# Documentation Generation

## Index

-   ``Devel``: Contains *.rst files for Developer Guide
-   ``examples``: Dipy app showcases. Add any tutorial here
-   ``examples_built``: keep it empty. Only for examples generation
-   ``releases_notes``: Contains all API changes / PR-issues resolve for a specific release
-   ``sphinx_ext``: Sphinx custom plugins
-   ``theory``: Diffusion theory + FAQ files
-   ``tools``: Scripts to generate some part of the documentation like API 
-   ``build``: Contains the generated documentation
-   ``_static``: Contains images, css, js for Sphinx to look at 
-   ``_templates``: Contains html layout for custom sphinx design

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