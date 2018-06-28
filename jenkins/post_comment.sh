#!/bin/bash
set -ex

ERRORS_FILENAME=$WORKSPACE/erros.log

# This script is kicked with $1=0 
# only if "bash build_and_test.sh" has exited successfully.
if [[ $1 -eq 0 ]]; then
  BODY="Test(${BUILD_DISPLAY_NAME}, ${GIT_COMMIT}) passed in $(uname -n)."
else
  BODY="Test(${BUILD_DISPLAY_NAME}, ${GIT_COMMIT}) failed in $(uname -n). 
\`\`\`$(cat ${ERRORS_FILENAME})\`\`\`"
fi


# BODY may contain [",\]. Escape it with jq's raw input option (-R).
# jq puts the escaped string from stdin onto the place of ".".
echo "${BODY}" |
  jq -sR "{ 
    \"commit_id\": \"${ghprbActualCommit}\",
    \"body\": . ,
    \"event\": \"COMMENT\"
  }" |
  curl -XPOST "https://api.github.com/repos/fixstars/clpy/pulls/${ghprbPullId}/reviews?access_token=${access_token}" -d @- 
