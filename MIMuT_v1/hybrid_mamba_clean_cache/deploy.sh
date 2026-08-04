#!/usr/bin/env bash
#

set -ex

REMOTE="ssh://root@muscriptor.kyutai.org"

: "${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token before deploying}"

docker -H "${REMOTE}" compose -f swarm.yml build --push

docker -H "${REMOTE}" stack deploy \
    --with-registry-auth \
    -c swarm.yml muscriptor

