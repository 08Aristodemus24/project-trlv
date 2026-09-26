#!/bin/bash

# If $1 is not provided, USERNAME defaults to 'guest'
COMMIT_MSG="${2:-guest}"
BRANCH="${1:-master}"

git add .
git commit -m "$COMMIT_MSG"
git push origin $BRANCH
