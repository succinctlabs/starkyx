#!/bin/sh

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

set -eu

if ! cargo fmt --all -- --check
then
    echo "${RED}There are some code style issues.${NC}"
    echo "${RED}Run cargo fmt first.${NC}"
    exit 1
fi

if ! RUSTFLAGS="-Copt-level=3 -Cdebug-assertions -Coverflow-checks=y -Cdebuginfo=0 -Cprefer-dynamic=y" cargo clippy --all-features --all-targets -p starky -- -D warnings -A incomplete-features
then
    echo "${RED}There are some clippy issues.${NC}"
    exit 1
fi

if ! RUSTFLAGS="-Copt-level=3 -Cdebug-assertions -Coverflow-checks=y -Cdebuginfo=0 -Cprefer-dynamic=y" cargo test -p starky
then
    echo "${RED}There are some test issues.${NC}"
    exit 1
fi

echo "${GREEN}Succesfully executed precommit hook.${NC}"

exit 0