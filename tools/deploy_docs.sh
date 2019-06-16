#!/bin/bash

# This script is used by Travis to automatically build and push the docs to the
# gh-pages branch of the datascience repo.

set -e # exit with nonzero exit code if anything fails

# Only build docs on master branch
if [[ $TRAVIS_PULL_REQUEST == true ]] ; then #&& $TRAVIS_BRANCH == "master" ]] ; then

  # https://docs.travis-ci.com/user/gui-and-headless-browsers/#Using-xvfb-to-Run-Tests-That-Require-a-GUI
  # sam: The DISPLAY env variable needs to be set for matplotlib charts to be
  # generated on Travis. I don't know why just matplotlib.use doesn't work.
  export DISPLAY=:99.0
  # sh -e /etc/init.d/xvfb start
  # sleep 3 # give xvfb some time to start

  echo "-- building docs --"

  make docs

  echo "-- pushing docs --"


  git config --global user.name "Travis CI"
  git config --global user.email "travis@travis.com"

  commitHash=`git rev-parse HEAD`

  make deploy_docs \
    DEPLOY_DOCS_MESSAGE="Generated by commit $commitHash, pushed by Travis build $TRAVIS_BUILD_NUMBER." \
    GH_REMOTE="https://${GH_TOKEN}@${GH_REPO}"
fi
