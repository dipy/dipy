#!/bin/bash
# Upload website to gh-pages
USAGE="$0 <html_dir> <project-name> [<organization-name>]"
HTML_DIR=$1
if [ -z "$HTML_DIR" ]; then
    echo $USAGE
    exit 1
fi
if [ ! -e "$HTML_DIR/index.html" ]; then
    echo "$HTML_DIR does not contain an index.html"
    exit 1
fi
if [ -d "$HTML_DIR/.git" ]; then
    echo "$HTML_DIR already contains a .git directory"
    exit 1
fi
PROJECT=$2
if [ -z "$PROJECT" ]; then
    echo $USAGE
    exit 1
fi
ORGANIZATION=$3
if [ -z "$ORGANIZATION" ]; then
    ORGANIZATION=nipy
fi
upstream_repo="https://github.com/$ORGANIZATION/$PROJECT"
cd $HTML_DIR
git init
git checkout -b gh-pages
git add *
# A nojekyll file is needed to tell github that this is *not* a jekyll site:
touch .nojekyll
git add .nojekyll
git commit -a -m "Documentation build - no history"
git remote add origin $upstream_repo
git push origin gh-pages --force
rm -rf .git  # Yes
