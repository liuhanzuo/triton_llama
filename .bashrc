cd() {
    builtin cd "$@" || return
    if [[ -f ".conda-env" ]]; then
        conda activate "$(cat .conda-env)"
    fi
}
